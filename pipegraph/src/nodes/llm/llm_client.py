# -*- coding: utf-8 -*-
"""
LLM Client — OpenAI-compatible wrapper supporting Ollama and OpenRouter.

Provider selection (in priority order):
  1. ``provider`` param passed to ``LLMClient.__init__()`` or ``LLMClient.create()``
  2. ``state.config["llm_provider"]`` (per-call runtime override for ablation)
  3. ``config.json → llm.provider`` (global default)

OpenRouter configuration:
  - base_url   : config.json → openrouter.base_url  (default: https://openrouter.ai/api/v1)
  - models     : config.json → openrouter.models.<role>
  - api_key    : env var OPENROUTER_API_KEY  (never commit to config.json)
  - headers    : HTTP-Referer and X-Title are injected automatically

Ollama configuration:
  - base_url   : config.json → llm.base_url  (default: http://localhost:11434/v1)
  - models     : config.json → llm.models.<role> / llm.model
  - api_key    : config.json → llm.api_key  (optional, Ollama accepts empty)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger("LLMClient")

# Module-level config cache (loaded once per process)
_CONFIG_CACHE: Optional[Dict[str, Any]] = None

# OpenRouter constants
_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
_OPENROUTER_APP_NAME = "PipeGraph-Anonymisation"
_OPENROUTER_SITE_URL = "https://github.com/Fireman4740/Anonymisation"


def _resolve_ollama_base_url(url: str) -> str:
    """
    In WSL2 NAT mode, localhost refers to the WSL VM, not the Windows host.
    Replace localhost with the Windows host IP (default gateway from /proc/net/route).
    """
    if "localhost" not in url and "127.0.0.1" not in url:
        return url
    if not os.path.exists("/proc/net/route"):
        return url

    try:
        with open("/proc/net/route") as f:
            next(f)  # skip header
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3 and parts[1] == "00000000":
                    gw_bytes = bytes.fromhex(parts[2])
                    host_ip = ".".join(str(b) for b in reversed(gw_bytes))
                    resolved = url.replace("localhost", host_ip).replace("127.0.0.1", host_ip)
                    logger.debug(f"WSL2 NAT — Ollama base_url rewritten: {url} → {resolved}")
                    return resolved
    except Exception:
        pass
    return url


# ---------------------------------------------------------------------------
# JSON extraction helper — used by _extract_content for reasoning models
# ---------------------------------------------------------------------------

def _find_json_block(text: str) -> Optional[str]:
    """
    Search for the most plausible JSON array or object in *text*.

    Strategy (in priority order):
      1. A fenced ```json … ``` block
      2. The LAST balanced [ … ] or { … } in the text (likely the answer after reasoning)
      3. None if nothing found

    Returns the extracted JSON string or None.
    """
    if not text:
        return None

    # 1. Fenced code block
    fence = re.search(r"```(?:json)?\s*(\[[\s\S]*?\]|\{[\s\S]*?\})\s*```", text)
    if fence:
        return fence.group(1).strip()

    # 2. Last balanced JSON block — search from the end
    #    We search for the last [ or { and try json.loads
    for start_char, end_char in [("[", "]"), ("{", "}")]:
        last_end = text.rfind(end_char)
        if last_end == -1:
            continue
        # Walk backward to find the matching opener
        depth = 0
        for i in range(last_end, -1, -1):
            if text[i] == end_char:
                depth += 1
            elif text[i] == start_char:
                depth -= 1
            if depth == 0:
                candidate = text[i : last_end + 1].strip()
                try:
                    json.loads(candidate)
                    return candidate
                except (json.JSONDecodeError, ValueError):
                    break  # malformed — try next type

    return None


# ---------------------------------------------------------------------------
# Approximate context-window sizes (in *tokens*) for common OpenRouter models.
# Used to estimate a safe max-characters for prompt truncation.
# Conservative estimates — better to under-fill than overflow.
# ---------------------------------------------------------------------------
_MODEL_CONTEXT_WINDOWS: Dict[str, int] = {
    # Ollama local models
    "gemma4:26b": 131_072,
    "gemma4:12b": 131_072,
    "gemma4:4b": 131_072,
    # OpenAI
    "openai/gpt-oss-20b": 8_192,
    "openai/gpt-4o-mini": 128_000,
    "openai/gpt-4o": 128_000,
    "openai/gpt-4-turbo": 128_000,
    # Meta Llama
    "meta-llama/llama-3.3-70b-instruct": 131_072,
    "meta-llama/llama-3.3-70b-instruct:free": 131_072,
    "meta-llama/llama-3.1-8b-instruct:free": 131_072,
    "meta-llama/llama-3.1-70b-instruct": 131_072,
    # Google
    "google/gemma-3-27b-it": 96_000,
    "google/gemma-3-27b-it:free": 96_000,
    "google/gemma-3-12b-it:free": 96_000,
    "google/gemini-flash-1.5": 1_000_000,
    "google/gemini-2.0-flash-001": 1_000_000,
    "google/gemini-2.5-flash-preview": 1_000_000,
    # Anthropic
    "anthropic/claude-3.5-sonnet": 200_000,
    "anthropic/claude-3-haiku": 200_000,
    # Mistral
    "mistralai/mistral-small-24b-instruct-2501:free": 32_768,
    # DeepSeek
    "deepseek/deepseek-chat-v3-0324:free": 131_072,
    "deepseek/deepseek-chat:free": 131_072,
    # Qwen (thinking models — content extraction handled by _extract_content)
    "qwen/qwen3-vl-235b-a22b-thinking": 32_768,
    "qwen/qwen3-235b-a22b": 40_960,
    "qwen/qwen3-235b-a22b-2507": 40_960,
}

# Models known to use "reasoning tokens" — they consume part of max_tokens for
# chain-of-thought before the actual answer.  We multiply max_tokens by this
# factor to give them enough budget for both reasoning AND the answer.
_REASONING_MODELS: Dict[str, float] = {
    # Ollama local reasoning models
    "gemma4:26b": 4.0,   # thinking model: ~6K chars reasoning before answer
    "gemma4:12b": 4.0,
    "gemma4:4b": 4.0,
    # OpenRouter
    "openai/gpt-oss-20b": 3.0,
    "deepseek/deepseek-r1": 3.0,
    "deepseek/deepseek-r1:free": 3.0,
    "qwen/qwen3-vl-235b-a22b-thinking": 3.0,
}


def _reasoning_multiplier(model: str) -> float:
    """Return the max_tokens multiplier for reasoning models, or 1.0 for normal models."""
    mult = _REASONING_MODELS.get(model)
    if mult:
        return mult
    # Check partial match (e.g. model with :free suffix)
    base = model.split(":")[0]
    for pattern, m in _REASONING_MODELS.items():
        if base == pattern.split(":")[0]:
            return m
    # Heuristic: models with "thinking" or "-r1" in the name are likely reasoning
    lower = model.lower()
    if "thinking" in lower or "-r1" in lower:
        return 3.0
    return 1.0

# Conservative chars-per-token ratio for estimation (1 token ≈ 3.5 chars for English)
_CHARS_PER_TOKEN = 3.5


def estimate_max_prompt_chars(model: str, reserved_output_tokens: int = 4096) -> int:
    """
    Estimate the maximum number of *characters* we can safely put in the prompt
    for a given model, leaving room for `reserved_output_tokens` of output and
    ~500 tokens of system/prompt template overhead.

    For reasoning models, the reserved output budget is multiplied by the
    reasoning multiplier (e.g. 3x for gpt-oss-20b) because reasoning tokens
    count toward max_tokens.

    Falls back to a conservative 6K-char limit for unknown models.
    """
    # Find context window — try exact match, then prefix match
    ctx_tokens = _MODEL_CONTEXT_WINDOWS.get(model)
    if ctx_tokens is None:
        for pattern, tokens in _MODEL_CONTEXT_WINDOWS.items():
            if model.startswith(pattern.split(":")[0]):
                ctx_tokens = tokens
                break
    if ctx_tokens is None:
        ctx_tokens = 8_192  # conservative default for unknown models

    # Account for reasoning overhead: reasoning tokens eat into context budget
    mult = _reasoning_multiplier(model)
    effective_output_tokens = int(reserved_output_tokens * mult)

    prompt_overhead_tokens = 500  # system prompt + template text
    available_tokens = ctx_tokens - effective_output_tokens - prompt_overhead_tokens
    available_tokens = max(available_tokens, 1_000)  # minimum floor

    return int(available_tokens * _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# Global rate-limiter for OpenRouter API calls.
# Uses a simple token-bucket per-model approach with X-RateLimit awareness.
# ---------------------------------------------------------------------------

class _RateLimiter:
    """Thread-safe rate limiter that respects OpenRouter's rate limit headers."""

    # Maximum time a single wait() call can block (seconds)
    _MAX_WAIT: float = 120.0

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # model → earliest time we're allowed to send the next request
        self._next_allowed: Dict[str, float] = {}
        # default minimum interval between requests to the same model (seconds)
        self._default_interval: float = 0.5

    def wait(self, model: str) -> None:
        """Block until it's safe to send a request for the given model."""
        with self._lock:
            now = time.monotonic()
            earliest = self._next_allowed.get(model, 0.0)
            if now < earliest:
                wait_time = earliest - now
            else:
                wait_time = 0.0
            # Cap wait to prevent unbounded blocking (e.g. bad reset_timestamp)
            wait_time = min(wait_time, self._MAX_WAIT)
            # Reserve our slot — next caller must wait at least _default_interval after us
            self._next_allowed[model] = max(earliest, now) + self._default_interval

        if wait_time > 0:
            logger.debug(f"Rate limiter: waiting {wait_time:.1f}s for {model}")
            time.sleep(wait_time)

    def report_rate_limit(self, model: str, reset_timestamp_ms: Optional[int] = None,
                          remaining: Optional[int] = None) -> None:
        """
        Update rate-limit info from OpenRouter response headers.
        Called after a 429 error to back off correctly.
        """
        with self._lock:
            now = time.monotonic()
            if reset_timestamp_ms is not None:
                # Convert Unix ms timestamp to monotonic wait time
                reset_in_seconds = max(0, (reset_timestamp_ms / 1000) - time.time())
                # Cap to prevent absurd waits from malformed timestamps
                reset_in_seconds = min(reset_in_seconds, self._MAX_WAIT)
                # Add small buffer
                self._next_allowed[model] = now + reset_in_seconds + 1.0
                logger.debug(
                    f"Rate limiter: model {model} rate-limited, "
                    f"reset in {reset_in_seconds:.0f}s"
                )
            elif remaining is not None and remaining == 0:
                # No reset info, but we know we're at the limit — wait 60s
                self._next_allowed[model] = now + 60.0
            else:
                # Generic backoff
                self._next_allowed[model] = now + 10.0


# Singleton rate limiter
_rate_limiter = _RateLimiter()


def _config_path() -> str:
    """Resolve the absolute path to pipegraph/config.json from this file's location."""
    here = os.path.dirname(os.path.abspath(__file__))
    # nodes/llm/ (1)→ nodes/ (2)→ src/ (3)→ pipegraph/
    return os.path.abspath(os.path.join(here, "../../../config.json"))


def load_full_config() -> Dict[str, Any]:
    """Load and cache the full config.json. Cache is per-process."""
    global _CONFIG_CACHE
    if _CONFIG_CACHE is not None:
        return _CONFIG_CACHE

    path = _config_path()
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                _CONFIG_CACHE = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
            _CONFIG_CACHE = {}
    else:
        logger.warning(f"config.json not found at {path}")
        _CONFIG_CACHE = {}

    return _CONFIG_CACHE or {}


def _load_llm_section() -> Dict[str, Any]:
    return load_full_config().get("llm", {})


def _resolve_provider(provider_override: Optional[str] = None) -> str:
    """
    Return the effective provider name (lowercase).
    Priority: explicit param > config.json → llm.provider
    """
    if provider_override:
        return provider_override.lower()
    cfg = _load_llm_section()
    return cfg.get("provider", "ollama").lower()


def _get_openrouter_api_key() -> str:
    """
    Retrieve the OpenRouter API key from the environment.
    Raises ValueError with a clear message if not set.
    """
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        # Fallback : charger le .env via chemin absolu résolu depuis ce fichier
        # (fonctionne même si le CWD n'est pas la racine du projet)
        try:
            here = os.path.dirname(os.path.abspath(__file__))
            # nodes/llm → nodes → src → pipegraph → project_root
            dotenv_path = os.path.abspath(os.path.join(here, "../../../../.env"))
            if os.path.exists(dotenv_path):
                # Lire brute-force sans dépendance python-dotenv
                with open(dotenv_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith("OPENROUTER_API_KEY=") and not line.startswith("#"):
                            key = line.split("=", 1)[1].strip().strip('"').strip("'")
                            break
        except Exception:
            pass
    if not key:
        # Dernier recours : pydantic LLMEnvSettings (env_file relatif au CWD)
        try:
            from src.config.settings import LLMEnvSettings  # type: ignore
            env_settings = LLMEnvSettings()
            if env_settings.has_openrouter_key:
                key = env_settings.openrouter_key
        except Exception:
            pass
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY is not set. "
            "Add it to your .env file: OPENROUTER_API_KEY=sk-or-v1-..."
        )
    return key


def _resolve_model(
    role: str,
    provider: str,
    cfg: Dict[str, Any],
    model_override: Optional[str] = None,
) -> str:
    """
    Resolve the effective model name.

    Priority chain (first non-None wins):
      1. model_override  (from state.config at runtime — ablation / CLI)
      2. config.json per-role  (openrouter.models.<role> / llm.models.<role>)
      3. config.json global    (openrouter.model / llm.model)
      4. Hard-coded default    (google/gemma-3-27b-it)
    """
    _DEFAULT_MODEL = "google/gemma-3-27b-it"

    # 1. runtime override (highest priority)
    if model_override:
        return model_override

    # 2 & 3. config.json — per-role then global
    section_key = "openrouter" if provider == "openrouter" else "llm"
    section = cfg.get(section_key, {})

    per_role = section.get("models", {}).get(role)
    if per_role:
        return per_role

    global_model = section.get("model")
    if global_model:
        return global_model

    return _DEFAULT_MODEL


class LLMClient:
    """
    Thin wrapper around the OpenAI client supporting Ollama and OpenRouter.

    Usage:
        # Use provider from config.json (default)
        client = LLMClient(role="audit")

        # Explicitly select OpenRouter
        client = LLMClient(role="audit", provider="openrouter")

        # From state config (for ablation / runtime override)
        client = LLMClient.create(role="detect", state_config=state["config"])

        response = client.chat([{"role": "user", "content": "Hello"}])
        data = LLMClient.extract_json(response)
    """

    def __init__(self, role: str = "detect", provider: Optional[str] = None, model_override: Optional[str] = None):
        """
        Args:
            role: One of "detect", "audit", "paraphrase".
                  Selects the model from the appropriate config section.
            provider: "openrouter", "ollama", or None (reads from config.json).
            model_override: Force a specific model, bypassing config.json and .env.
                  Useful for per-run ablation (state.config["llm_model"] / ["llm_model_detect"]).
        """
        self.role = role
        self._client = None  # lazy init
        self._extra_headers: Dict[str, str] = {}

        effective_provider = _resolve_provider(provider)
        self.provider = effective_provider

        cfg = load_full_config()

        if effective_provider == "openrouter":
            or_cfg = cfg.get("openrouter", {})
            self.base_url: str = or_cfg.get("base_url", _OPENROUTER_BASE_URL)
            self.model: str = _resolve_model(role, effective_provider, cfg, model_override)
            self.timeout: float = float(or_cfg.get("timeout_seconds", 60))
            self.retry_count: int = int(or_cfg.get("retry_count", 2))
            self.supports_json_format: bool = bool(or_cfg.get("supports_response_format", True))
            # OpenRouter required headers
            self._extra_headers = {
                "HTTP-Referer": _OPENROUTER_SITE_URL,
                "X-Title": _OPENROUTER_APP_NAME,
            }
            # Warn if using a small-context reasoning model (fundamentally broken)
            mult = _reasoning_multiplier(self.model)
            ctx = _MODEL_CONTEXT_WINDOWS.get(self.model, 8_192)
            if mult > 1.0 and ctx <= 16_000:
                logger.warning(
                    f"⚠️  Model {self.model} is a REASONING model with only "
                    f"{ctx} tokens context. Reasoning consumes most of the "
                    f"output budget, leaving little for the actual answer. "
                    f"Consider using a non-reasoning model like "
                    f"google/gemma-3-27b-it or deepseek/deepseek-chat-v3-0324:free."
                )
            logger.debug(f"LLMClient configured for OpenRouter — model={self.model}")
        else:
            # Ollama (or any other OpenAI-compatible server)
            llm_cfg = cfg.get("llm", {})
            self.base_url = _resolve_ollama_base_url(
                llm_cfg.get("base_url", "http://localhost:11434/v1")
            )
            self.model = _resolve_model(role, effective_provider, cfg, model_override)
            self.timeout = float(llm_cfg.get("timeout_seconds", 90))
            self.retry_count = int(llm_cfg.get("retry_count", 1))
            self.supports_json_format = bool(llm_cfg.get("supports_response_format", False))
            logger.debug(f"LLMClient configured for {effective_provider} — model={self.model}")

    @classmethod
    def create(cls, role: str = "detect", state_config: Optional[Dict[str, Any]] = None) -> "LLMClient":
        """
        Factory that resolves provider and model from state.config at runtime.
        Use this in LLM nodes for ablation study / per-call override.

        Supported state.config keys:
          - ``llm_provider``          : "openrouter" | "ollama"  (selects the endpoint)
          - ``llm_model``             : model string for all roles   (e.g. "openai/gpt-4o-mini")
          - ``llm_model_detect``      : model override for detection role only
          - ``llm_model_audit``       : model override for audit role only
          - ``llm_model_paraphrase``  : model override for paraphrase role only

        Args:
            role: "detect" | "audit" | "paraphrase"
            state_config: state["config"] dict
        """
        provider_override: Optional[str] = None
        model_override: Optional[str] = None
        if state_config:
            provider_override = state_config.get("llm_provider")
            # Role-specific model takes precedence over global llm_model
            model_override = (
                state_config.get(f"llm_model_{role}")
                or state_config.get("llm_model")
            )
        return cls(role=role, provider=provider_override, model_override=model_override or None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_client(self):
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore

                if self.provider == "openrouter":
                    api_key = _get_openrouter_api_key()
                else:
                    cfg = _load_llm_section()
                    api_key = cfg.get("api_key") or "no-key-required"

                kwargs: Dict[str, Any] = {
                    "base_url": self.base_url,
                    "api_key": api_key,
                    "timeout": self.timeout,
                    "max_retries": self.retry_count,
                }
                if self._extra_headers:
                    kwargs["default_headers"] = self._extra_headers

                self._client = OpenAI(**kwargs)
                logger.debug(
                    f"LLM client initialised — provider={self.provider}, "
                    f"base_url={self.base_url}, model={self.model}"
                )
            except ImportError:
                logger.error("openai package not installed. Run: pip install openai>=1.0.0")
                raise
        return self._client

    # ------------------------------------------------------------------
    # Content extraction (handles "thinking" models)
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_content(resp: Any, model: str = "") -> Optional[str]:
        """
        Extract the usable text content from a chat completion response.

        "Thinking" / reasoning models (gpt-oss-20b, qwen3-*-thinking, deepseek-r1…)
        may put the reasoning trace in ``message.reasoning_content`` and the actual
        answer in ``message.content``.  We ALWAYS prefer ``message.content``.

        If ``content`` is empty BUT ``reasoning_content`` contains a valid JSON
        structure, we extract it.  We NEVER return raw reasoning text — that would
        pollute downstream JSON parsing and produce garbage.
        """
        if not resp.choices:
            return None

        msg = resp.choices[0].message
        content: Optional[str] = getattr(msg, "content", None)
        reasoning: Optional[str] = (
            getattr(msg, "reasoning_content", None)
            or getattr(msg, "reasoning", None)
        )

        # ── 1. Standard content — always preferred ──
        if content and content.strip():
            # Strip <think>…</think> wrapper if present (some models inline reasoning)
            stripped = re.sub(
                r"<think>.*?</think>\s*", "", content, flags=re.DOTALL
            ).strip()
            if stripped:
                return stripped
            # The entire content was inside <think> — try extracting JSON from it
            think_match = re.search(r"<think>(.*?)</think>", content, flags=re.DOTALL)
            if think_match:
                inner = think_match.group(1).strip()
                json_in_think = _find_json_block(inner)
                if json_in_think:
                    logger.info(
                        f"Extracted JSON from <think> block in content "
                        f"({model}, {len(json_in_think)} chars)"
                    )
                    return json_in_think

        # ── 2. Content is empty — model used all output tokens on reasoning ──
        if reasoning and isinstance(reasoning, str) and reasoning.strip():
            # ONLY extract structured JSON from reasoning — never return raw text
            json_block = _find_json_block(reasoning)
            if json_block:
                logger.info(
                    f"Reasoning model ({model}): extracted JSON from "
                    f"reasoning_content ({len(reasoning)} chars reasoning "
                    f"→ {len(json_block)} chars JSON)"
                )
                return json_block

            # No usable JSON in reasoning — treat as empty response
            logger.warning(
                f"Reasoning model ({model}): content is empty and no JSON "
                f"found in reasoning_content ({len(reasoning)} chars). "
                f"Model likely spent all output tokens on reasoning. "
                f"Consider increasing max_tokens or using a non-reasoning model."
            )
            return None

        return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    # Default retry / concurrency settings (overridable via config.json → openrouter)
    _MAX_RETRIES: int = 3
    _RETRY_BASE_DELAY: float = 1.0   # seconds — grows exponentially (1s, 2s, 4s …)
    _RETRY_MAX_DELAY: float = 30.0
    _DEFAULT_WORKERS: int = 2         # parallel requests for chat_batch (conservative for free models)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> Optional[str]:
        """
        Send a chat completion request and return the assistant's message text.

        Retries up to ``_MAX_RETRIES`` times with exponential back-off on
        transient failures (timeout, rate-limit 429, server errors 5xx, empty
        responses).  On persistent empty responses, attempts the fallback model
        if configured.  Returns ``None`` only if all attempts fail.
        """
        cfg = load_full_config()
        section = cfg.get("openrouter", {}) if self.provider == "openrouter" else cfg.get("llm", {})
        max_retries = int(section.get("retry_count", section.get("max_retries", self._MAX_RETRIES)))
        base_delay = float(section.get("retry_base_delay", self._RETRY_BASE_DELAY))
        max_delay = float(section.get("retry_max_delay", self._RETRY_MAX_DELAY))

        # Models to try: primary, then fallback if configured and different
        fallback_model = section.get("fallback_model")
        models_to_try = [self.model]
        if fallback_model and fallback_model != self.model:
            models_to_try.append(fallback_model)

        last_error: Optional[Exception] = None

        for model_idx, current_model in enumerate(models_to_try):
            if model_idx > 0:
                logger.info(
                    f"LLM ({self.role}/{self.provider}) switching to fallback model: {current_model}"
                )

            for attempt in range(1, max_retries + 1):
                try:
                    # --- Rate limiter: wait if needed ---
                    if self.provider == "openrouter":
                        _rate_limiter.wait(current_model)

                    client = self._get_client()

                    # For reasoning models, multiply max_tokens so the model
                    # has enough budget for BOTH reasoning AND the actual answer.
                    effective_max_tokens = max_tokens
                    mult = _reasoning_multiplier(current_model)
                    is_reasoning = mult > 1.0
                    if is_reasoning:
                        # Cap at model's context window to avoid errors
                        ctx = _MODEL_CONTEXT_WINDOWS.get(current_model, 8_192)
                        effective_max_tokens = min(int(max_tokens * mult), ctx - 500)
                        logger.debug(
                            f"Reasoning model {current_model}: "
                            f"max_tokens {max_tokens} → {effective_max_tokens} (×{mult})"
                        )

                    kwargs: Dict[str, Any] = {
                        "model": current_model,
                        "messages": messages,
                        "temperature": temperature,
                        "max_tokens": effective_max_tokens,
                    }
                    if self.supports_json_format:
                        kwargs["response_format"] = {"type": "json_object"}

                    resp = client.chat.completions.create(**kwargs)

                    # --- Detailed diagnostics ---
                    finish_reason = getattr(resp.choices[0], "finish_reason", "unknown") if resp.choices else "no_choices"
                    content = self._extract_content(resp, current_model)
                    usage = getattr(resp, "usage", None)

                    if usage:
                        logger.debug(
                            f"LLM ({self.role}/{self.provider}/{current_model}) "
                            f"usage: prompt={getattr(usage, 'prompt_tokens', '?')}, "
                            f"completion={getattr(usage, 'completion_tokens', '?')}, "
                            f"finish_reason={finish_reason}"
                        )

                    # Treat empty / whitespace-only responses as transient failures
                    if not content or not content.strip():
                        logger.warning(
                            f"LLM ({self.role}/{self.provider}) empty response "
                            f"(model={current_model}, attempt {attempt}/{max_retries}, "
                            f"finish_reason={finish_reason})"
                        )
                        # If finish_reason is "length", the context is too long — no point retrying same model
                        if finish_reason == "length":
                            logger.warning(
                                f"LLM response truncated (finish_reason=length). "
                                f"Context may be too long for model {current_model}."
                            )
                            break  # Try fallback model instead
                        last_error = RuntimeError(f"empty response (finish_reason={finish_reason})")
                        if attempt < max_retries:
                            delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                            time.sleep(delay)
                            continue
                        break  # Exhausted retries for this model, try fallback

                    logger.debug(
                        f"LLM ({self.role}/{self.provider}) → {len(content)} chars "
                        f"(model={current_model}, attempt {attempt}, finish_reason={finish_reason})"
                    )
                    return content

                except Exception as e:
                    last_error = e
                    err_str = str(e)

                    # Non-retriable errors — fail immediately
                    if "OPENROUTER_API_KEY" in err_str:
                        logger.error(f"❌ OpenRouter API key manquante — {e}")
                        return None
                    if "404" in err_str or "model_not_found" in err_str.lower():
                        logger.error(
                            f"❌ Model not found: {current_model} — {e}"
                        )
                        break  # Try fallback model
                    # 400 Bad Request — non-retriable (wrong params, etc.)
                    if "400" in err_str:
                        logger.error(
                            f"❌ Bad request for {current_model}: {e}"
                        )
                        break  # Try fallback model

                    # --- Extract rate-limit info from 429 errors ---
                    if "429" in err_str and self.provider == "openrouter":
                        reset_ts: Optional[int] = None
                        remaining: Optional[int] = None
                        try:
                            # OpenRouter includes metadata with rate-limit headers
                            # The openai SDK's APIStatusError has a .response attribute
                            resp_obj = getattr(e, "response", None)
                            if resp_obj is not None:
                                json_fn = getattr(resp_obj, "json", None)
                                if callable(json_fn):
                                    err_body = json_fn()  # type: ignore[call-arg]
                                    if not isinstance(err_body, dict):
                                        err_body = {}
                                    headers_dict = (
                                        err_body.get("error", {})
                                        .get("metadata", {})
                                        .get("headers", {})
                                    )
                                    if isinstance(headers_dict, dict):
                                        if "X-RateLimit-Reset" in headers_dict:
                                            reset_ts = int(headers_dict["X-RateLimit-Reset"])
                                        if "X-RateLimit-Remaining" in headers_dict:
                                            remaining = int(headers_dict["X-RateLimit-Remaining"])
                        except Exception:
                            pass
                        _rate_limiter.report_rate_limit(current_model, reset_ts, remaining)

                    # Rate limiting (429) / server errors (5xx) → retry
                    is_retriable = any(code in err_str for code in ("429", "500", "502", "503", "504", "timeout", "Timeout"))
                    if not is_retriable and attempt == 1:
                        # For unknown errors, retry once just to be safe
                        is_retriable = True

                    if is_retriable and attempt < max_retries:
                        delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                        logger.warning(
                            f"LLM call failed (model={current_model}, "
                            f"attempt {attempt}/{max_retries}, "
                            f"retry in {delay:.1f}s): {type(e).__name__}: {e}"
                        )
                        time.sleep(delay)
                        continue

                    logger.warning(
                        f"LLM call failed after {attempt} attempt(s) "
                        f"(role={self.role}, provider={self.provider}, "
                        f"model={current_model}): "
                        f"{type(e).__name__}: {e}"
                    )
                    break  # Try fallback model

        # All models exhausted
        logger.warning(
            f"LLM exhausted all models and retries "
            f"(role={self.role}, models={models_to_try}): {last_error}"
        )

        # Cross-provider fallback: if primary provider is unreachable, try fallback_provider
        fallback_provider_name = section.get("fallback_provider")
        if fallback_provider_name and fallback_provider_name != self.provider and last_error is not None:
            is_conn_error = "Connection error" in str(last_error) or "ConnectionError" in type(last_error).__name__
            if is_conn_error:
                logger.info(
                    f"LLM ({self.role}) provider '{self.provider}' unreachable — "
                    f"falling back to '{fallback_provider_name}'"
                )
                try:
                    fallback_client = LLMClient(role=self.role, provider=fallback_provider_name)
                    return fallback_client.chat(
                        messages, temperature=temperature, max_tokens=max_tokens
                    )
                except Exception as fb_err:
                    logger.warning(
                        f"LLM fallback provider '{fallback_provider_name}' also failed: {fb_err}"
                    )

        return None

    # ------------------------------------------------------------------
    # Batch / parallel API
    # ------------------------------------------------------------------

    def chat_batch(
        self,
        message_sets: Sequence[List[Dict[str, str]]],
        temperature: float = 0.1,
        max_tokens: int = 2048,
        max_workers: Optional[int] = None,
    ) -> List[Optional[str]]:
        """
        Send multiple chat completion requests **in parallel** using a
        thread-pool.  Each request benefits from the same retry logic as
        ``chat()``.

        Args:
            message_sets: A sequence of message lists (one per request).
            temperature:  Sampling temperature.
            max_tokens:   Max tokens per response.
            max_workers:  Thread-pool size.  Defaults to
                          config.json → openrouter.max_workers (or 8).

        Returns:
            A list of response strings (or None for failures), in the same
            order as ``message_sets``.
        """
        if max_workers is None:
            cfg = load_full_config()
            section = cfg.get("openrouter", {}) if self.provider == "openrouter" else cfg.get("llm", {})
            max_workers = int(section.get("max_workers", self._DEFAULT_WORKERS))

        n = len(message_sets)
        if n == 0:
            return []
        # For single request, skip overhead
        if n == 1:
            return [self.chat(message_sets[0], temperature=temperature, max_tokens=max_tokens)]

        results: List[Optional[str]] = [None] * n

        logger.info(
            f"LLM batch: {n} requests, {max_workers} workers "
            f"(provider={self.provider}, model={self.model})"
        )

        def _worker(idx: int) -> Tuple[int, Optional[str]]:
            return idx, self.chat(
                message_sets[idx],
                temperature=temperature,
                max_tokens=max_tokens,
            )

        with ThreadPoolExecutor(max_workers=min(max_workers, n)) as pool:
            futures = {pool.submit(_worker, i): i for i in range(n)}
            for future in as_completed(futures):
                try:
                    # Timeout per-future to prevent indefinite hang
                    idx, response = future.result(timeout=180)
                    results[idx] = response
                except TimeoutError:
                    idx = futures[future]
                    logger.warning(f"LLM batch item {idx} timed out after 180s")
                    results[idx] = None
                except Exception as exc:
                    idx = futures[future]
                    logger.warning(f"LLM batch item {idx} failed: {exc}")
                    results[idx] = None

        success = sum(1 for r in results if r is not None)
        logger.info(f"LLM batch complete: {success}/{n} successful")
        return results

    # ------------------------------------------------------------------
    # JSON extraction & Repair
    # ------------------------------------------------------------------

    @staticmethod
    def _repair_json(text: str) -> str:
        """
        Attempt to automatically repair a truncated or malformed JSON string.
        (e.g., handles truncated output from small OSS models)
        """
        # Remove trailing space/newline
        text = text.strip()
        
        # Simple stack-based parsing to close unclosed strings, objects and arrays.
        stack = []
        in_string = False
        escape = False
        
        for char in text:
            if in_string:
                if escape:
                    escape = False
                elif char == '\\':
                    escape = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == '{':
                    stack.append('}')
                elif char == '[':
                    stack.append(']')
                elif char in ('}', ']'):
                    if stack and stack[-1] == char:
                        stack.pop()
        
        repaired = text
        if in_string:
            repaired += '"'
            
        # Optional: remove trailing comma before closing if we just terminated a string
        repaired = re.sub(r',\s*$', '', repaired)
        
        while stack:
            repaired += stack.pop()
            
        return repaired

    @staticmethod
    def extract_json(text: Optional[str]) -> Any:
        """
        Robustly extract JSON from an LLM response that may be wrapped in
        markdown code fences, contain extra prose, or be truncated.
        Returns the parsed Python object, or None on failure.
        """
        if not text:
            return None

        stripped = text.strip()

        # Helper to try parsing and repairing
        def try_parse(json_str: str) -> Any:
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # Attempt to repair
                try:
                    repaired = LLMClient._repair_json(json_str)
                    return json.loads(repaired)
                except json.JSONDecodeError:
                    return None

        # 1) Direct parse
        result = try_parse(stripped)
        if result is not None:
            return result

        # 2) Extract from ```json ... ``` or ``` ... ``` blocks
        m = re.search(r"```(?:json)?\s*([\s\S]+?)```", stripped)
        if m:
            result = try_parse(m.group(1).strip())
            if result is not None:
                return result

        # 3) Extact with potentially unclosed ``` or missing markdown
        # e.g., ```json\n{ "test": ... EOF
        m = re.search(r"```(?:json)?\s*([\s\S]+)", stripped)
        if m:
            result = try_parse(m.group(1).strip())
            if result is not None:
                return result

        # 4) Find the first complete (or partial that we can repair) {...} or [...] block
        # Use simple finding of first opening brace/bracket
        first_brace = stripped.find('{')
        first_bracket = stripped.find('[')
        
        start_idx = -1
        if first_brace != -1 and first_bracket != -1:
            start_idx = min(first_brace, first_bracket)
        else:
            start_idx = max(first_brace, first_bracket)
            
        if start_idx != -1:
            raw_substr = stripped[start_idx:]
            # We try greedily to parse it, repair will close everything left open
            result = try_parse(raw_substr)
            if result is not None:
                return result

        logger.debug(f"extract_json: could not parse response: {text[:300]}")
        return None

    def is_available(self) -> bool:
        """Quick connectivity check — returns True if the server responds."""
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False
