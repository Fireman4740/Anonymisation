from __future__ import annotations

import hashlib
import json
import re
import random
import threading
import time
import urllib.error
import urllib.request
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Iterator, List, Optional

from atlas_anno.config import load_config
from atlas_anno.io import serialize
from atlas_anno.records import llm_run_meta_from_dict
from atlas_anno.schemas import LLMRunMeta, PromptSpec
from atlas_anno.settings import AtlasSettings
from atlas_anno.storage import append_llm_run, load_llm_cache_entry, save_llm_cache_entry


class LLMError(RuntimeError):
    pass


class RetryableLLMError(LLMError):
    pass


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)


def _strip_think_blocks(text: str) -> str:
    """Retire les blocs <think>…</think> produits par certains modèles de raisonnement."""
    return _THINK_RE.sub("", text).strip()


def _extract_fence_block(text: str) -> str | None:
    """Extrait le contenu du premier bloc ``` ou ```json si présent."""
    match = _FENCE_RE.search(text)
    return match.group(1).strip() if match else None


def _parse_json_payload(text: str) -> Any:
    """Parse JSON de façon robuste :
    1. Retire les blocs <think>…</think>.
    2. Extrait les fences ```json … ```.
    3. Tente json.loads direct.
    4. Extrait le premier bloc {…} / […] et re-tente.
    5. Utilise json_repair comme dernier recours.
    Lève LLMError si aucune tentative ne réussit.
    """
    if not text or not text.strip():
        raise LLMError("empty llm response")

    # Étape 1 : retirer les blocs de raisonnement
    cleaned = _strip_think_blocks(text)

    # Étape 2 : extraire la fence si présente
    fenced = _extract_fence_block(cleaned)
    candidate = fenced if fenced else cleaned

    # Étape 3 : parse direct
    candidate = candidate.strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    # Étape 4 : slice sur le premier { ou [
    start = -1
    for ch in ("{["):
        pos = candidate.find(ch)
        if pos != -1 and (start == -1 or pos < start):
            start = pos
    end = max(candidate.rfind("}"), candidate.rfind("]"))
    if start != -1 and end != -1 and end > start:
        sliced = candidate[start : end + 1]
        try:
            return json.loads(sliced)
        except json.JSONDecodeError:
            pass
    else:
        sliced = candidate

    # Étape 5 : json_repair — uniquement si le texte contient au moins une structure JSON
    if "{" in candidate or "[" in candidate:
        try:
            import json_repair  # type: ignore[import]
            target = sliced if sliced else candidate
            repaired = json_repair.loads(target)
            # json_repair peut renvoyer None, une chaîne vide ou le texte brut sur échec
            if repaired is not None and repaired != "" and repaired != target:
                return repaired
        except ImportError:
            pass
        except Exception:
            pass

    raise LLMError(f"response does not contain parsable JSON: {candidate[:200]!r}")


# Alias conservé pour les callers internes existants
_extract_json_block = _parse_json_payload


def _extract_message_text(payload: Dict[str, Any]) -> str:
    try:
        choice = payload["choices"][0]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("openrouter response missing choices") from exc

    if not isinstance(choice, dict):
        raise LLMError("openrouter choice payload is invalid")

    message = choice.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            text = content.strip()
            if text:
                return text
        elif isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                    continue
                if not isinstance(item, dict):
                    continue
                text_value = item.get("text")
                if isinstance(text_value, str):
                    parts.append(text_value)
                    continue
                content_value = item.get("content")
                if isinstance(content_value, str):
                    parts.append(content_value)
            text = "".join(parts).strip()
            if text:
                return text

        refusal = message.get("refusal")
        if isinstance(refusal, str) and refusal.strip():
            raise LLMError(f"openrouter refusal: {refusal.strip()}")

    text_choice = choice.get("text")
    if isinstance(text_choice, str) and text_choice.strip():
        return text_choice.strip()

    finish_reason = choice.get("finish_reason") or choice.get("native_finish_reason") or "unknown"
    raise LLMError(f"openrouter response missing text content (finish_reason={finish_reason})")


@dataclass(frozen=True)
class _RequestOutcome:
    payload: Dict[str, Any] | None
    attempt_count: int
    queue_wait_ms: int
    error: LLMError | None = None


@dataclass
class OpenRouterClient:
    settings: AtlasSettings
    runtime_overrides: Dict[str, Any] = field(default_factory=dict)

    _semaphores: ClassVar[Dict[tuple[str, int], threading.BoundedSemaphore]] = {}
    _semaphore_lock: ClassVar[threading.Lock] = threading.Lock()

    def enabled(self) -> bool:
        return self.settings.llm_enabled

    def _runtime_config(self) -> Dict[str, Any]:
        return load_config().defaults.get("llm", {}).get("runtime", {})

    def _runtime_value(self, key: str, default: Any) -> Any:
        if key in self.runtime_overrides:
            return self.runtime_overrides[key]
        return self._runtime_config().get(key, default)

    def _cache_enabled(self) -> bool:
        return bool(self._runtime_value("cache_enabled", True))

    def _max_retries(self) -> int:
        return int(load_config().defaults.get("llm", {}).get("max_retries", 2))

    def _repair_retries(self) -> int:
        return int(load_config().defaults.get("llm", {}).get("repair_retries", 1))

    def _backoff_initial_seconds(self) -> float:
        return float(self._runtime_value("backoff_initial_seconds", 2))

    def _backoff_max_seconds(self) -> float:
        return float(self._runtime_value("backoff_max_seconds", 30))

    def _model_limit(self, model: str) -> int:
        if model == self.settings.atlas_model_creative:
            return max(1, int(self._runtime_value("creative_workers", 8)))
        return max(1, int(self._runtime_value("reasoning_workers", 12)))

    def _semaphore_key(self, model: str) -> tuple[str, int]:
        return (model, self._model_limit(model))

    def _semaphore(self, model: str) -> threading.BoundedSemaphore:
        key = self._semaphore_key(model)
        with self._semaphore_lock:
            if key not in self._semaphores:
                self._semaphores[key] = threading.BoundedSemaphore(key[1])
            return self._semaphores[key]

    @contextmanager
    def _acquire_model_slot(self, model: str) -> Iterator[int]:
        semaphore = self._semaphore(model)
        started = time.perf_counter()
        semaphore.acquire()
        queue_wait_ms = int((time.perf_counter() - started) * 1000)
        try:
            yield queue_wait_ms
        finally:
            semaphore.release()

    def _json_response_format_enabled(self) -> bool:
        return bool(self._runtime_value("json_response_format", True))

    def _perform_http_request(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        use_json_format: bool = False,
    ) -> Dict[str, Any]:
        if not self.enabled():
            raise LLMError("OPENROUTER_API_KEY is not configured")

        body: Dict[str, Any] = {
            "model": model,
            "temperature": temperature,
            "messages": messages,
        }
        if use_json_format:
            body["response_format"] = {"type": "json_object"}

        payload = json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            self.settings.openrouter_base_url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.settings.openrouter_api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.settings.http_timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            if exc.code in {408, 429, 500, 502, 503, 504}:
                raise RetryableLLMError(f"openrouter http error {exc.code}: {message}") from exc
            # 400/422 avec response_format peuvent indiquer que le modèle ne supporte
            # pas le paramètre → relancer sans response_format (dégradation gracieuse).
            if use_json_format and exc.code in {400, 422}:
                raise RetryableLLMError(
                    f"openrouter http error {exc.code} (response_format may be unsupported): {message}"
                ) from exc
            raise LLMError(f"openrouter http error {exc.code}: {message}") from exc
        except urllib.error.URLError as exc:
            raise RetryableLLMError(f"openrouter network error: {exc}") from exc

        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RetryableLLMError("openrouter returned invalid JSON") from exc

    def _sleep_backoff(self, attempt_index: int) -> None:
        base = min(self._backoff_max_seconds(), self._backoff_initial_seconds() * (2 ** attempt_index))
        jitter = random.uniform(0.0, max(0.1, base * 0.25))
        time.sleep(min(self._backoff_max_seconds(), base + jitter))

    def _request_with_retries(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.0,
        use_json_format: bool = False,
    ) -> _RequestOutcome:
        total_attempts = 0
        total_queue_wait_ms = 0
        last_error: LLMError | None = None
        max_retries = self._max_retries()
        # Dégradation gracieuse : désactiver response_format après un premier échec 400/422.
        _use_json_fmt = use_json_format

        for attempt in range(max_retries + 1):
            with self._acquire_model_slot(model) as queue_wait_ms:
                total_queue_wait_ms += queue_wait_ms
                total_attempts += 1
                try:
                    payload = self._perform_http_request(
                        model, messages, temperature=temperature, use_json_format=_use_json_fmt
                    )
                    return _RequestOutcome(payload=payload, attempt_count=total_attempts, queue_wait_ms=total_queue_wait_ms)
                except RetryableLLMError as exc:
                    last_error = exc
                    # Si l'erreur vient de response_format non supporté, désactiver pour les retries suivants.
                    if _use_json_fmt and "response_format may be unsupported" in str(exc):
                        _use_json_fmt = False
                    if attempt >= max_retries:
                        break
            self._sleep_backoff(attempt)

        if last_error is None:
            last_error = LLMError("openrouter request failed")
        return _RequestOutcome(payload=None, attempt_count=total_attempts, queue_wait_ms=total_queue_wait_ms, error=last_error)

    def _cache_key(self, step_name: str, prompt_spec: PromptSpec, model: str, user_prompt: str, temperature: float) -> str:
        payload = f"{step_name}\n{model}\n{prompt_spec.version}\n{temperature:.3f}\n{user_prompt}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_hit_meta(
        self,
        *,
        step_name: str,
        model: str,
        prompt_spec: PromptSpec,
        cached_meta: LLMRunMeta,
    ) -> LLMRunMeta:
        return LLMRunMeta(
            step_name=step_name,
            model=model,
            prompt_version=prompt_spec.version,
            llm_used=True,
            fallback_used=False,
            retry_count=0,
            attempt_count=0,
            queue_wait_ms=0,
            cache_hit=True,
            validation_errors=[],
            latency_ms=0,
            estimated_cost=0.0,
            raw_response_excerpt=cached_meta.raw_response_excerpt,
        )

    def _append_run(self, prompt_spec: PromptSpec, meta: LLMRunMeta) -> None:
        append_llm_run({"step_name": meta.step_name, "prompt_name": prompt_spec.prompt_name, **meta.__dict__})

    def _load_cached_value(
        self,
        *,
        step_name: str,
        prompt_spec: PromptSpec,
        model: str,
        user_prompt: str,
        temperature: float,
        validator: Callable[[Any], Any],
    ) -> tuple[Any, LLMRunMeta] | None:
        if not self._cache_enabled():
            return None
        payload = load_llm_cache_entry(step_name, self._cache_key(step_name, prompt_spec, model, user_prompt, temperature))
        if not payload:
            return None
        try:
            cached_result = validator(payload["result"])
            cached_meta = llm_run_meta_from_dict(payload["llm_run"])
        except Exception:
            return None
        meta = self._cache_hit_meta(step_name=step_name, model=model, prompt_spec=prompt_spec, cached_meta=cached_meta)
        self._append_run(prompt_spec, meta)
        return cached_result, meta

    def _save_cached_value(
        self,
        *,
        step_name: str,
        prompt_spec: PromptSpec,
        model: str,
        user_prompt: str,
        temperature: float,
        result: Any,
        meta: LLMRunMeta,
    ) -> None:
        if not self._cache_enabled() or not meta.llm_used or meta.fallback_used or meta.cache_hit:
            return
        save_llm_cache_entry(
            step_name,
            self._cache_key(step_name, prompt_spec, model, user_prompt, temperature),
            {
                "result": serialize(result),
                "llm_run": serialize(meta),
            },
        )

    def generate_text(self, prompt: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
        selected_model = model or self.settings.atlas_model_creative
        outcome = self._request_with_retries(selected_model, [{"role": "user", "content": prompt}], temperature=temperature)
        if outcome.error is not None or outcome.payload is None:
            raise outcome.error or LLMError("openrouter response missing text")
        return _extract_message_text(outcome.payload)

    def generate_json(self, prompt: str, model: Optional[str] = None) -> Any:
        text = self.generate_text(prompt, model=model or self.settings.atlas_model_reasoning, temperature=0.0)
        return _extract_json_block(text)

    def rewrite_text(self, text: str, instruction: str) -> str:
        prompt = f"{instruction}\n\nTexte:\n{text}"
        return self.generate_text(prompt, model=self.settings.atlas_model_creative, temperature=0.1)

    def score_candidates(self, text: str, candidates: List[Dict[str, Any]], aux_knowledge: Dict[str, Any] | None = None) -> Any:
        prompt = (
            "Classe les candidats suivants pour une attaque de re-identification closed-world.\n"
            "Retourne strictement un JSON avec les champs top_k et rationale.\n\n"
            f"Texte:\n{text}\n\n"
            f"Connaissance auxiliaire:\n{json.dumps(aux_knowledge or {}, ensure_ascii=False, indent=2)}\n\n"
            f"Candidats:\n{json.dumps(candidates, ensure_ascii=False, indent=2)}"
        )
        return self.generate_json(prompt, model=self.settings.atlas_model_reasoning)

    def complete_text(
        self,
        *,
        step_name: str,
        prompt_spec: PromptSpec,
        user_prompt: str,
        model: Optional[str] = None,
        temperature: float = 0.2,
        fallback_text: Optional[str] = None,
    ) -> tuple[str, LLMRunMeta]:
        selected_model = model or self.settings.atlas_model_creative
        cached = self._load_cached_value(
            step_name=step_name,
            prompt_spec=prompt_spec,
            model=selected_model,
            user_prompt=user_prompt,
            temperature=temperature,
            validator=lambda payload: str(payload),
        )
        if cached is not None:
            return cached

        started = time.perf_counter()
        validation_errors: List[str] = []
        used_llm = False
        fallback_used = False
        output = fallback_text or ""
        raw_excerpt = ""
        attempt_count = 0
        queue_wait_ms = 0

        if not self.enabled():
            fallback_used = True
            validation_errors.append("OPENROUTER_API_KEY missing")
        else:
            outcome = self._request_with_retries(
                selected_model,
                [
                    {"role": "system", "content": prompt_spec.system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            attempt_count = outcome.attempt_count
            queue_wait_ms = outcome.queue_wait_ms
            if outcome.error is not None or outcome.payload is None:
                fallback_used = True
                validation_errors.append(str(outcome.error))
            else:
                try:
                    raw_excerpt = json.dumps(outcome.payload, ensure_ascii=False)[:400]
                    output = _extract_message_text(outcome.payload)
                    used_llm = True
                except (LLMError, KeyError, IndexError, TypeError) as exc:
                    validation_errors.append(str(exc))
                    fallback_used = True

        latency_ms = int((time.perf_counter() - started) * 1000)
        meta = LLMRunMeta(
            step_name=step_name,
            model=selected_model,
            prompt_version=prompt_spec.version,
            llm_used=used_llm,
            fallback_used=fallback_used,
            retry_count=max(0, attempt_count - 1),
            attempt_count=attempt_count,
            queue_wait_ms=queue_wait_ms,
            cache_hit=False,
            validation_errors=validation_errors,
            latency_ms=latency_ms,
            estimated_cost=_estimate_cost(user_prompt, output, selected_model),
            raw_response_excerpt=raw_excerpt,
        )
        self._append_run(prompt_spec, meta)
        if not output and fallback_text is not None:
            output = fallback_text
        self._save_cached_value(
            step_name=step_name,
            prompt_spec=prompt_spec,
            model=selected_model,
            user_prompt=user_prompt,
            temperature=temperature,
            result=output,
            meta=meta,
        )
        return output, meta

    def complete_json(
        self,
        *,
        step_name: str,
        prompt_spec: PromptSpec,
        user_prompt: str,
        model: Optional[str],
        validator: Callable[[Any], Any],
        fallback_value: Any,
        temperature: float = 0.0,
        allow_fallback: bool = True,
    ) -> tuple[Any, LLMRunMeta]:
        """Génère une réponse JSON validée par *validator*.

        Si *allow_fallback* est False et que toutes les tentatives échouent,
        le champ ``error`` de LLMRunMeta est mis à True et *fallback_value*
        n'est PAS utilisée comme résultat (None est renvoyé).
        Le caller doit impérativement vérifier ``meta.error`` dans ce cas.
        """
        selected_model = model or self.settings.atlas_model_reasoning
        cached = self._load_cached_value(
            step_name=step_name,
            prompt_spec=prompt_spec,
            model=selected_model,
            user_prompt=user_prompt,
            temperature=temperature,
            validator=validator,
        )
        if cached is not None:
            return cached

        started = time.perf_counter()
        validation_errors: List[str] = []
        used_llm = False
        fallback_used = False
        generation_error = False
        result: Any = fallback_value
        raw_excerpt = ""
        attempt_count = 0
        queue_wait_ms = 0
        repair_messages: List[Dict[str, str]] = [
            {"role": "system", "content": prompt_spec.system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        use_json_fmt = self._json_response_format_enabled()

        if not self.enabled():
            fallback_used = True
            generation_error = not allow_fallback
            validation_errors.append("OPENROUTER_API_KEY missing")
        else:
            for repair_index in range(self._repair_retries() + 1):
                outcome = self._request_with_retries(
                    selected_model,
                    repair_messages,
                    temperature=temperature,
                    use_json_format=use_json_fmt,
                )
                attempt_count += outcome.attempt_count
                queue_wait_ms += outcome.queue_wait_ms
                # Désactiver response_format pour les passes de réparation suivantes
                # si le modèle a signalé qu'il n'est pas supporté.
                if outcome.error and "response_format may be unsupported" in str(outcome.error):
                    use_json_fmt = False
                if outcome.error is not None or outcome.payload is None:
                    validation_errors.append(str(outcome.error))
                    break
                try:
                    raw_excerpt = json.dumps(outcome.payload, ensure_ascii=False)[:400]
                    content = _extract_message_text(outcome.payload)
                    parsed = _parse_json_payload(content)
                    result = validator(parsed)
                    used_llm = True
                    fallback_used = False
                    generation_error = False
                    break
                except (LLMError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError) as exc:
                    err_msg = str(exc)
                    validation_errors.append(f"repair#{repair_index}: {err_msg}")
                    if repair_index >= self._repair_retries():
                        break
                    # Construire un message de réparation contextuel pour le prochain tour.
                    repair_hint = (
                        f"\n\nTA RÉPONSE PRÉCÉDENTE ÉTAIT INVALIDE ({err_msg[:200]}).\n"
                        "Corrige-la et retourne UNIQUEMENT un objet JSON valide.\n"
                        "Premier caractère : `{`  \u2014  dernier caractère : `}`.\n"
                        "Aucun markdown, aucune fence ```, aucun commentaire, aucune virgule traînante."
                    )
                    repair_messages = [
                        {"role": "system", "content": prompt_spec.system_prompt},
                        {"role": "user", "content": user_prompt + repair_hint},
                    ]

            else:
                # La boucle s'est terminée normalement sans break (ne devrait pas arriver)
                pass

            if not used_llm:
                if allow_fallback:
                    fallback_used = True
                else:
                    fallback_used = False
                    generation_error = True
                    result = None

        latency_ms = int((time.perf_counter() - started) * 1000)
        meta = LLMRunMeta(
            step_name=step_name,
            model=selected_model,
            prompt_version=prompt_spec.version,
            llm_used=used_llm,
            fallback_used=fallback_used,
            error=generation_error,
            retry_count=max(0, attempt_count - 1),
            attempt_count=attempt_count,
            queue_wait_ms=queue_wait_ms,
            cache_hit=False,
            validation_errors=validation_errors,
            latency_ms=latency_ms,
            estimated_cost=_estimate_cost(
                user_prompt,
                json.dumps(serialize(result) if result is not None else {}, ensure_ascii=False),
                selected_model,
            ),
            raw_response_excerpt=raw_excerpt,
        )
        self._append_run(prompt_spec, meta)
        if used_llm and not generation_error:
            self._save_cached_value(
                step_name=step_name,
                prompt_spec=prompt_spec,
                model=selected_model,
                user_prompt=user_prompt,
                temperature=temperature,
                result=result,
                meta=meta,
            )
        return result, meta


def _estimate_cost(prompt_text: str, output_text: str, model: str) -> float:
    config = load_config()
    llm_config = config.defaults.get("llm", {})
    pricing = llm_config.get("pricing_usd_per_1k_tokens", {})
    configured_reasoning_model = str(llm_config.get("models", {}).get("reasoning", "aion-labs/aion-2.0"))
    prompt_tokens = max(1, len(prompt_text) // 4)
    output_tokens = max(1, len(output_text) // 4)
    if model == configured_reasoning_model:
        input_rate = float(pricing.get("reasoning_input", 0.0))
        output_rate = float(pricing.get("reasoning_output", 0.0))
    else:
        input_rate = float(pricing.get("creative_input", 0.0))
        output_rate = float(pricing.get("creative_output", 0.0))
    return round((prompt_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate, 6)
