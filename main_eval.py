#!/usr/bin/env python3
"""API Flask d'anonymisation (point d'entrée).

Endpoints:
  GET  /health -> {status: ok}
  GET  /       -> métadonnées
  POST /anonymize {text, level?, scope_id?, secret_salt?, ner_results?, openrouter_models?, overrides?}

Mise à jour:
 - Support champ JSON `overrides` passé directement à l'orchestrateur.
 - Par défaut (si aucun override explicite) active le preset GLiNER "best" (ensemble pondéré)
   en injectant la liste des modèles si `GLINER_PRESET` n'est pas déjà défini.
"""
from __future__ import annotations

import os
import time
import typing as t
from importlib import import_module
from flask import Flask, request, jsonify

_ANON_FN: t.Callable[..., dict] | None = None
# Liste des modèles du preset "best" (doit rester alignée avec src/ner_ensemble.py)
BEST_GLINER_MODELS = [
    "EmergentMethods/gliner_medium_news-v2.1",
    "numind/NuNER_Zero-span",
    "urchade/gliner_large-v2.1",
    "urchade/gliner_multi-v2.1",
]


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
    overrides: t.Dict[str, t.Any] | None = None,
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
        overrides=overrides,
    )


def _inject_default_gliner_best(overrides: dict | None) -> dict | None:
    """Ajoute les modèles GLiNER "best" si aucun override gliner fourni et que
    GLINER_PRESET n'est pas défini (on ne force pas si l'utilisateur configure via env).
    """
    if overrides is None:
        overrides = {}
    # Ne rien faire si l'utilisateur a déjà précisé des modèles ou a explicitement désactivé GLiNER
    if any(k in overrides for k in ("gliner_models",)) or overrides.get("ner_use_gliner") is False:
        return overrides or None
    if os.getenv("GLINER_PRESET"):
        return overrides or None  # Respecte config externe
    # Injection liste best (run_gliner utilisera le weighting via env GLINER_WEIGHTING=1 par défaut)
    overrides.setdefault("ner_use_gliner", True)
    overrides.setdefault("gliner_models", BEST_GLINER_MODELS)
    # On peut laisser d'autres sources actives; possibilité d'ajouter des modèles via overrides si besoin
    return overrides or None


def create_app() -> Flask:
    app = Flask(__name__)

    @app.get("/health")
    def health():  # type: ignore
        return jsonify({"status": "ok"})

    @app.get("/")
    def root():  # type: ignore
        return jsonify({
            "service": "anonymization-api",
            "version": "0.2.0",
            "endpoints": ["GET /health", "POST /anonymize"],
            "default_level": "L2",
            "gliner_default_preset": "best (auto-injecté si non configuré)",
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
        overrides = data.get("overrides")
        if overrides is not None and not isinstance(overrides, dict):
            return jsonify({"error": "'overrides' doit être un objet"}), 400
        overrides = _inject_default_gliner_best(overrides)
        try:
            res = anonymize_call(
                text=text,
                level=level,
                scope_id=scope_id,
                secret_salt=secret_salt,
                ner_results=ner_results,
                openrouter_models=models,
                overrides=overrides,
            )
        except Exception as e:
            return jsonify({"error": f"Echec anonymisation: {e}"}), 500
        dt = int((time.time() - t0) * 1000)
        return jsonify({**res, "timings_ms": {"total": dt}, "applied_overrides": overrides or {}})

    return app


def main():  # pragma: no cover
    app = create_app()
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    debug = os.getenv("FLASK_DEBUG", "0") == "1"
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":  # pragma: no cover
    main()
