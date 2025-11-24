import os
import json
import re
from copy import deepcopy
from typing import Any, Dict, Optional

import requests

try:  # pragma: no cover - import principal via package src.config
	from ..config.config_loader import load_config as _load_config  # type: ignore
except Exception:  # pragma: no cover
	try:  # fallback lorsque le paquet n'est pas résolu (exécution depuis racine)
		from src.config.config_loader import load_config as _load_config  # type: ignore
	except Exception:  # pragma: no cover
		try:
			from config_loader import load_config as _load_config  # type: ignore
		except Exception:  # pragma: no cover
			_load_config = None  # type: ignore

__all__ = ["JSONParseError", "OpenRouterClient", "load_llm_settings"]


_DEFAULTS_BY_PROVIDER: Dict[str, Dict[str, Any]] = {
	"lmstudio": {
		"base_url": "http://10.10.153.169:1234/v1",
		"retry_count": 1,
		"fallback_model": None,
		"api_key_env": None,
		"require_api_key": False,
		"supports_response_format": False,
		"allow_fallback": False,
		"timeout_seconds": 90,
		"headers": {},
		"models": {
			"detect": "openai/gpt-oss-20b",
			"paraphrase": "openai/gpt-oss-20b",
			"audit": "openai/gpt-oss-20b",
		},
	},
 "openrouter": {
		"base_url": "https://openrouter.ai/api/v1",
		"retry_count": 1,
		"fallback_model": "qwen/qwen3-30b-a3b-instruct-2507",
		"api_key_env": "OPENROUTER_API_KEY",
		"require_api_key": True,
		"supports_response_format": True,
		"allow_fallback": True,
		"timeout_seconds": 60,
		"headers": {},
		"models": {},
	},
	
	"ollama": {
		"base_url": "http://127.0.0.1:11434/v1",
		"retry_count": 1,
		"fallback_model": None,
		"api_key_env": None,
		"require_api_key": False,
		"supports_response_format": False,
		"allow_fallback": False,
		"timeout_seconds": 60,
		"headers": {},
		"models": {},
	},
}


def load_llm_settings(force_reload: bool = False) -> Dict[str, Any]:
	"""Charge la configuration LLM (openrouter/local) en fusionnant les sections pertinentes."""

	cfg: Dict[str, Any] = {}
	if _load_config:
		try:
			cfg = _load_config(force_reload=force_reload)
		except Exception:
			cfg = {}

	llm_section = cfg.get("llm") if isinstance(cfg.get("llm"), dict) else {}
	openrouter_section = cfg.get("openrouter") if isinstance(cfg.get("openrouter"), dict) else {}

	provider = str(
		llm_section.get("provider")
		or openrouter_section.get("provider")
		or "openrouter"
	).lower()

	defaults = deepcopy(_DEFAULTS_BY_PROVIDER.get(provider, _DEFAULTS_BY_PROVIDER["openrouter"]))

	# Ordre: defaults -> openrouter (fallback) -> llm (override)
	merged: Dict[str, Any] = {**defaults}
	merged.update(openrouter_section)
	merged.update(llm_section)
	merged["provider"] = provider

	# Normalisation types
	merged["retry_count"] = int(merged.get("retry_count", defaults["retry_count"]) or defaults["retry_count"])
	fallback_model = merged.get("fallback_model")
	merged["fallback_model"] = fallback_model if fallback_model else None
	if merged.get("api_key_env") in {"", None}:
		merged["api_key_env"] = defaults.get("api_key_env")
	merged["require_api_key"] = bool(merged.get("require_api_key", defaults["require_api_key"]))
	merged["supports_response_format"] = bool(merged.get("supports_response_format", defaults["supports_response_format"]))
	merged["allow_fallback"] = bool(merged.get("allow_fallback", defaults["allow_fallback"]))
	try:
		merged["timeout_seconds"] = int(merged.get("timeout_seconds", defaults["timeout_seconds"]))
	except Exception:
		merged["timeout_seconds"] = defaults["timeout_seconds"]

	headers = merged.get("headers") if isinstance(merged.get("headers"), dict) else {}
	merged["headers"] = {str(k): str(v) for k, v in headers.items()}

	models = merged.get("models") if isinstance(merged.get("models"), dict) else {}
	merged["models"] = {str(k): str(v) for k, v in models.items() if isinstance(v, str) and v}

	api_key = merged.get("api_key")
	merged["api_key"] = str(api_key) if isinstance(api_key, str) and api_key else None

	return merged


class JSONParseError(RuntimeError):
	"""Erreur spécifique lorsque le modèle ne renvoie pas de JSON exploitable."""


class OpenRouterClient:
	"""Client OpenAI-compatible configuré via openrouter/local providers.

	- Gère parsing JSON robuste (code fences, tool_calls...)
	- Supporte plusieurs fournisseurs (OpenRouter, LM Studio, Ollama) via config.json
	- Permet de désactiver l'autorisation et `response_format` selon les capacités du backend
	"""

	CHAT_COMPLETIONS_PATH = "/chat/completions"

	def __init__(
		self,
		api_key: Optional[str] = None,
		base_url: Optional[str] = None,
		retry_count: int = 1,
		fallback_model: Optional[str] = None,
		*,
		provider: str = "openrouter",
		api_key_env: Optional[str] = None,
		require_api_key: Optional[bool] = None,
		extra_headers: Optional[Dict[str, str]] = None,
		supports_response_format: Optional[bool] = None,
		default_timeout: Optional[int] = None,
		allow_fallback: Optional[bool] = None,
	) -> None:
		self.provider = (provider or "openrouter").lower()
		defaults = _DEFAULTS_BY_PROVIDER.get(self.provider, _DEFAULTS_BY_PROVIDER["openrouter"])

		self.base_url = (base_url or defaults["base_url"]).rstrip("/")
		self.retry_count = max(1, int(retry_count or defaults["retry_count"]))
		self.fallback_model = fallback_model if fallback_model else defaults.get("fallback_model")

		self._explicit_key = api_key
		self.api_key_env = api_key_env if api_key_env is not None else defaults.get("api_key_env")
		self._requires_api_key = (
			require_api_key if require_api_key is not None else bool(defaults.get("require_api_key", False))
		)

		self.extra_headers = dict(extra_headers or defaults.get("headers", {}))
		self.supports_response_format = (
			supports_response_format if supports_response_format is not None else bool(defaults.get("supports_response_format", True))
		)
		self.default_timeout = int(default_timeout or defaults.get("timeout_seconds", 60))
		self.allow_fallback = (
			bool(allow_fallback) if allow_fallback is not None else bool(defaults.get("allow_fallback", False))
		)

		self.config_models: Dict[str, str] = {}

	# ----------------- Propriétés -----------------
	@property
	def api_key(self) -> Optional[str]:
		if self._explicit_key:
			return self._explicit_key
		if self.api_key_env:
			return os.getenv(self.api_key_env)
		return None

	@property
	def requires_api_key(self) -> bool:
		return self._requires_api_key

	# ----------------- Helpers Parsing -----------------
	def _strip_code_fences(self, s: str) -> str:
		if not s:
			return s
		s = s.strip()
		if s.startswith("```") and s.endswith("```"):
			s = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", s)
			s = re.sub(r"\n?```$", "", s)
			return s.strip()
		return s

	def _extract_fenced_block(self, s: str) -> Optional[str]:
		m = re.search(r"```json\s*(.*?)```", s, flags=re.DOTALL | re.IGNORECASE)
		if m:
			return m.group(1).strip()
		m = re.search(r"```\s*(.*?)```", s, flags=re.DOTALL)
		if m:
			return m.group(1).strip()
		return None

	def _extract_balanced(self, s: str) -> Optional[str]:
		def find_balanced(txt: str, open_ch: str, close_ch: str) -> Optional[str]:
			in_str = False
			esc = False
			quote = ''
			depth = 0
			start_idx = -1
			best_span = None
			for i, ch in enumerate(txt):
				if esc:
					esc = False
					continue
				if ch == '\\':
					esc = True
					continue
				if in_str:
					if ch == quote:
						in_str = False
					continue
				else:
					if ch in ('"', "'"):
						in_str = True
						quote = ch
						continue
				if ch == open_ch and not in_str:
					if depth == 0:
						start_idx = i
					depth += 1
				elif ch == close_ch and not in_str and depth > 0:
					depth -= 1
					if depth == 0 and start_idx != -1:
						span = (start_idx, i + 1)
						if best_span is None or (span[1] - span[0]) > (best_span[1] - best_span[0]):
							best_span = span
			if best_span:
				return txt[best_span[0]:best_span[1]]
			return None

		obj = find_balanced(s, '{', '}')
		arr = find_balanced(s, '[', ']')
		return obj or arr

	def _remove_trailing_commas(self, s: str) -> str:
		return re.sub(r",\s*([}\]])", r"\1", s)

	def _parse_possible_json(self, content: str) -> Dict[str, Any]:
		if content is None:
			raise JSONParseError("Empty content")
		text = str(content).strip()
		try:
			return json.loads(text)
		except Exception:
			pass

		fb = self._extract_fenced_block(text)
		if fb:
			try:
				return json.loads(self._remove_trailing_commas(self._strip_code_fences(fb)))
			except Exception:
				pass

		cand = self._extract_balanced(text)
		if cand:
			try:
				return json.loads(cand)
			except Exception:
				repaired = self._remove_trailing_commas(self._strip_code_fences(cand))
				try:
					return json.loads(repaired)
				except Exception:
					pass

		m = re.search(r"\{[\s\S]*\}", text)
		if m:
			blob = self._remove_trailing_commas(self._strip_code_fences(m.group(0)))
			try:
				return json.loads(blob)
			except Exception:
				pass
		raise JSONParseError(f"Model did not return JSON: {text[:240]}")

	# ----------------- Appels réseau -----------------
	def _single_call(
		self,
		system_prompt: str,
		user_prompt: str,
		model: str,
		temperature: float,
		max_tokens: int,
		timeout: int,
	) -> Dict[str, Any]:
		key = self.api_key
		if self._requires_api_key and not key:
			missing = self.api_key_env or "OPENROUTER_API_KEY"
			raise RuntimeError(f"API key required for provider '{self.provider}' (expected env var {missing})")

		url = f"{self.base_url}{self.CHAT_COMPLETIONS_PATH}"
		headers = {"Content-Type": "application/json", "Accept": "application/json"}
		headers.update(self.extra_headers)
		if key:
			headers.setdefault("Authorization", f"Bearer {key}")

		payload: Dict[str, Any] = {
			"model": model,
			"messages": [
				{"role": "system", "content": system_prompt},
				{"role": "user", "content": user_prompt},
			],
			"temperature": temperature,
			"max_tokens": max_tokens,
		}
		if self.supports_response_format:
			payload["response_format"] = {"type": "json_object"}

		try:
			resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
			resp.raise_for_status()
			data = resp.json()
		except requests.RequestException as exc:
			raise JSONParseError(f"LLM call failed: {exc}") from exc

		try:
			choice0 = (data.get("choices") or [{}])[0]
			msg = choice0.get("message") or {}
		except Exception as exc:  # pragma: no cover - structure invalide
			raise JSONParseError(f"Unexpected response structure: {data}") from exc

		tool_calls = msg.get("tool_calls") or []
		for tc in tool_calls:
			try:
				fn = (tc.get("function") or {})
				args = fn.get("arguments")
				if isinstance(args, str) and args.strip():
					return self._parse_possible_json(args)
			except Exception:
				continue

		content = msg.get("content") or ""
		return self._parse_possible_json(content)

	def call_json(
		self,
		system_prompt: str,
		user_prompt: str,
		model: str,
		temperature: float = 0.2,
		max_tokens: int = 1200,
		timeout: Optional[int] = None,
		allow_fallback: bool = True,
	) -> Dict[str, Any]:
		"""Appel avec retries + fallback pour obtenir un JSON valide."""

		last_error: Optional[Exception] = None
		effective_timeout = timeout or self.default_timeout

		for _ in range(self.retry_count):
			try:
				return self._single_call(
					system_prompt=system_prompt,
					user_prompt=user_prompt,
					model=model,
					temperature=temperature,
					max_tokens=max_tokens,
					timeout=effective_timeout,
				)
			except JSONParseError as exc:
				last_error = exc
				continue

		if allow_fallback and self.allow_fallback and self.fallback_model and self.fallback_model != model:
			try:
				return self._single_call(
					system_prompt=system_prompt,
					user_prompt=user_prompt,
					model=self.fallback_model,
					temperature=temperature,
					max_tokens=max_tokens,
					timeout=effective_timeout,
				)
			except JSONParseError as exc:
				last_error = exc

		if last_error:
			raise last_error
		raise JSONParseError("Unknown error without captured exception during call_json")

	# ----------------- Construction via config -----------------
	@classmethod
	def from_config(cls, force_reload: bool = False) -> "OpenRouterClient":
		settings = load_llm_settings(force_reload=force_reload)
		import sys
		print(f"[OpenRouterClient] Loaded settings:", file=sys.stderr)
		print(f"  provider: {settings.get('provider')}", file=sys.stderr)
		print(f"  base_url: {settings.get('base_url')}", file=sys.stderr)
		print(f"  require_api_key: {settings.get('require_api_key')}", file=sys.stderr)
		client = cls(
			api_key=settings.get("api_key"),
			base_url=settings.get("base_url"),
			retry_count=settings.get("retry_count", 1),
			fallback_model=settings.get("fallback_model"),
			provider=settings.get("provider", "openrouter"),
			api_key_env=settings.get("api_key_env"),
			require_api_key=settings.get("require_api_key"),
			extra_headers=settings.get("headers"),
			supports_response_format=settings.get("supports_response_format"),
			default_timeout=settings.get("timeout_seconds"),
			allow_fallback=settings.get("allow_fallback"),
		)
		client.config_models = dict(settings.get("models", {}))
		return client

