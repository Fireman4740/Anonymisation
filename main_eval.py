#!/usr/bin/env python3
"""API Flask d'anonymisation (point d'entrée).

Endpoints:
  GET  /health -> {status: ok}
  GET  /       -> métadonnées
  POST /anonymize {text, level?, scope_id?, secret_salt?, ner_results?, openrouter_models?}
"""
from __future__ import annotations

import os
import time
import typing as t
from importlib import import_module
from flask import Flask, request, jsonify

_ANON_FN: t.Callable[..., dict] | None = None


def _load() -> t.Callable[..., dict]:
    global _ANON_FN
    if _ANON_FN is not None:
        return _ANON_FN
    for name in ("orchestrator", "src.orchestrator", "src.validator.orchestrator"):
        try:
            mod = import_module(name)
            fn = getattr(mod, "anonymize_text", None)
            if callable(fn):
                _ANON_FN = fn  # type: ignore
                return _ANON_FN
        except Exception:
            continue
    raise RuntimeError("anonymize_text introuvable")


def anonymize_call(
    text: str,
    level: str = "L2",
    scope_id: str | None = None,
    secret_salt: str = "default_secret",
    ner_results: t.List[dict] | None = None,
    openrouter_models: t.Dict[str, str] | None = None,
) -> dict:
    fn = _load()
    scope = scope_id or f"SCOPE-{int(time.time()*1000)%1_000_000}"
    return fn(
        value=text,
        scope_id=scope,
        secret_salt=secret_salt,
        level=level,
        openrouter_models=openrouter_models,
        ner_results=ner_results or [],
    )


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():  # type: ignore
        return jsonify({"status": "ok"})

    @app.get("/")
    def root():  # type: ignore
        return jsonify({
            "service": "anonymization-api",
            "version": "0.1.0",
            "endpoints": ["GET /health", "POST /anonymize"],
            "default_level": "L2",
        })

    @app.post("/anonymize")
    def anonymize():  # type: ignore
        t0 = time.time()
        if not request.is_json:
            return jsonify({"error": "Corps JSON requis"}), 400
        data = request.get_json(silent=True) or {}
        text = data.get("text")
        if not isinstance(text, str) or not text.strip():
            return jsonify({"error": "Champ 'text' manquant ou vide"}), 400
        level = str(data.get("level", "L2"))
        secret_salt = str(data.get("secret_salt", os.getenv("ANON_SECRET", "default_secret")))
        scope_id = data.get("scope_id")
        ner_results = data.get("ner_results") or []
        if not isinstance(ner_results, list):
            return jsonify({"error": "'ner_results' doit être une liste"}), 400
        models = data.get("openrouter_models")
        if models is not None and not isinstance(models, dict):
            return jsonify({"error": "'openrouter_models' doit être un objet"}), 400
        try:
            res = anonymize_call(
                text=text,
                level=level,
                scope_id=scope_id,
                secret_salt=secret_salt,
                ner_results=ner_results,
                openrouter_models=models,
            )
        except Exception as e:
            return jsonify({"error": f"Echec anonymisation: {e}"}), 500
        dt = int((time.time() - t0) * 1000)
        return jsonify({**res, "timings_ms": {"total": dt}})

    return app


def main():  # pragma: no cover
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":  # pragma: no cover
    main()
