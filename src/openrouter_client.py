import os
import json
import re
from typing import Any, Dict, Optional

import requests


class JSONParseError(RuntimeError):
    """Erreur spécifique lorsque le modèle ne renvoie pas de JSON exploitable."""
    pass


class OpenRouterClient:
    """OpenRouter chat client avec parsing JSON robuste + retry/fallback.

    Stratégie parsing (dans l'ordre):
      1. Tentative JSON directe.
      2. Extraction bloc ```json ... ``` ou ``` ... ```.
      3. Extraction de l'objet/array le plus long équilibré (braces/brackets) dans le texte.
      4. Réparations légères (suppression fences, virgules traînantes).
      5. Recherche finale d'un objet { ... } quelconque.
    Si tool_calls présent avec arguments fonction JSON, on tente d'abord dessus.

    Stratégie robustesse:
      - Jusqu'à `retry_count` tentatives sur le modèle principal en cas d'erreur de parsing.
      - Puis 1 tentative sur `fallback_model` (par défaut openai/gpt-4o-mini) si activé et différent.
      - Lève JSONParseError si aucune tentative ne retourne un JSON valide.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        retry_count: int = 1,
        fallback_model: str = "openai/gpt-4o-mini",
    ):
        self._explicit_key = api_key
        self.base_url = base_url.rstrip("/")
        self.retry_count = max(1, retry_count)
        self.fallback_model = fallback_model

    # ----------------- Propriétés -----------------
    @property
    def api_key(self) -> Optional[str]:
        return self._explicit_key or os.getenv("OPENROUTER_API_KEY")

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
        # 1 - direct
        try:
            return json.loads(text)
        except Exception:
            pass
        # 2 - fenced block
        fb = self._extract_fenced_block(text)
        if fb:
            try:
                return json.loads(self._remove_trailing_commas(self._strip_code_fences(fb)))
            except Exception:
                pass
        # 3 - balanced
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
        # 4 - recherche générique objet
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
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY not set – disable LLM features or provide a key")

        url = f"{self.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
        }
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        try:
            choice0 = (data.get("choices") or [{}])[0]
            msg = choice0.get("message") or {}
        except Exception as e:
            raise JSONParseError(f"Unexpected response structure: {data}") from e

        # tool_calls prioritaire
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
        timeout: int = 60,
        allow_fallback: bool = True,
    ) -> Dict[str, Any]:
        """Appel avec retries + fallback pour obtenir un JSON valide."""
        last_error: Optional[Exception] = None
        for _ in range(self.retry_count):
            try:
                return self._single_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except JSONParseError as e:
                last_error = e
                continue

        if allow_fallback and self.fallback_model and self.fallback_model != model:
            try:
                return self._single_call(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    model=self.fallback_model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=timeout,
                )
            except JSONParseError as e:
                last_error = e

        if last_error:
            raise last_error
        raise JSONParseError("Unknown error without captured exception during call_json")