"""Minimal FastAPI server to expose the anonymisation pipeline."""
from __future__ import annotations

import json
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.orchestrator import anonymize_text
from src.core.policy import preset
from src.services.detection.detection import create_detection_service
from src.services.generalization.generalizer import GeneralizationService

DEFAULT_LEVEL = os.getenv("PIPELINE_DEFAULT_LEVEL", "L0")
DEFAULT_SECRET = os.getenv("PIPELINE_SECRET_SALT", "change_me")
DEFAULT_SCOPE_PREFIX = os.getenv("PIPELINE_SCOPE_PREFIX", "scope")
DEFAULT_OVERRIDES = os.getenv("PIPELINE_DEFAULT_OVERRIDES", "{}")

# --- GLOBAL SERVICES CACHE ---
_SERVICES = {
    "detection": None,
    "generalization": None
}

def _load_default_overrides(raw: str) -> Dict[str, Any]:
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        pass
    return {}

PIPELINE_OVERRIDES = _load_default_overrides(DEFAULT_OVERRIDES)

class AnonymizeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Texte à anonymiser")
    scope_id: Optional[str] = Field(None, description="Scope personnalisé")
    level: Optional[str] = Field(None, description="Override du niveau L0/L1/L2")
    secret_salt: Optional[str] = Field(None, description="Override du secret HMAC")
    overrides: Dict[str, Any] = Field(default_factory=dict, description="Overrides ponctuels")
    ner_results: Optional[List[Dict[str, Any]]] = Field(
        default=None,
        description="Résultats NER externes compatibles",
    )

class AnonymizeResponse(BaseModel):
    anonymized_text: str
    audit: Dict[str, Any]
    evaluation: Dict[str, Any]
    policy: Dict[str, Any]

class HealthResponse(BaseModel):
    status: str
    default_level: str

def _generate_scope_id() -> str:
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:6]
    return f"{DEFAULT_SCOPE_PREFIX}-{timestamp}-{suffix}"

def _merge_overrides(request_overrides: Dict[str, Any]) -> Dict[str, Any]:
    merged = PIPELINE_OVERRIDES.copy()
    merged.update(request_overrides or {})
    return merged

# --- LIFESPAN MANAGER ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load services once
    print("[API] Loading pipeline services...")
    try:
        policy = preset(DEFAULT_LEVEL)
        # On crée les services une seule fois ici
        _SERVICES["detection"] = create_detection_service(policy, PIPELINE_OVERRIDES)
        _SERVICES["generalization"] = GeneralizationService(policy)
        print("[API] Pipeline services loaded successfully.")
    except Exception as e:
        print(f"[API] Error loading services: {e}")
    
    yield
    
    # Shutdown: Clean up if needed
    _SERVICES.clear()

app = FastAPI(
    title="Anonymisation Pipeline API",
    description="Expose le pipeline via une API REST minimale",
    version="1.0.0",
    lifespan=lifespan,
)

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def healthcheck() -> HealthResponse:
    return HealthResponse(status="ok", default_level=DEFAULT_LEVEL)

@app.post("/anonymize", response_model=AnonymizeResponse, tags=["anonymisation"])
def anonymize(payload: AnonymizeRequest) -> AnonymizeResponse:
    scope_id = payload.scope_id or _generate_scope_id()
    level = payload.level or DEFAULT_LEVEL
    secret = payload.secret_salt or DEFAULT_SECRET
    
    # Utiliser les services pré-chargés si on est sur le niveau par défaut
    # Sinon, on recrée (mais grâce aux caches dans les classes, c'est rapide et sans fuite)
    detection_service = _SERVICES["detection"] if level == DEFAULT_LEVEL else None
    generalization_service = _SERVICES["generalization"] if level == DEFAULT_LEVEL else None

    try:
        result = anonymize_text(
            payload.text,
            scope_id=scope_id,
            secret_salt=secret,
            level=level,
            overrides=_merge_overrides(payload.overrides),
            ner_results=payload.ner_results,
            detection_service=detection_service,
            generalization_service=generalization_service,
        )
        return AnonymizeResponse(**result)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("scripts.api_server:app", host=host, port=port, reload=os.getenv("API_RELOAD", "0") == "1")