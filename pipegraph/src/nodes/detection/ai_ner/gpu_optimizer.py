# -*- coding: utf-8 -*-
"""
NER GPU Optimizer - Optimisations pour GPU puissant (24GB VRAM)
FIX: Ajout de caching global pour éviter les fuites de mémoire.
"""
from __future__ import annotations

import os
import json
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Optional imports
try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    torch = None

try:
    from gliner import GLiNER  # type: ignore
    _GLINER_AVAILABLE = True
except ImportError:
    _GLINER_AVAILABLE = False
    GLiNER = None

try:
    from transformers import pipeline as hf_pipeline, AutoTokenizer, AutoModelForTokenClassification  # type: ignore
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    hf_pipeline = AutoTokenizer = AutoModelForTokenClassification = None

# Import du module NER existant
try:
    from .ensemble import (
        split_sentences,
        _normalize_gliner_label,
        _GLINER_PRESETS,
        _GLINER_MODEL_WEIGHTS,
        GLINER_ALL_LABELS,
    )
except ImportError:
    # Fallback si import relatif échoue (ex: exécution directe)
    from ensemble import (  # type: ignore
        split_sentences,
        _normalize_gliner_label,
        _GLINER_PRESETS,
        _GLINER_MODEL_WEIGHTS,
        GLINER_ALL_LABELS,
    )

# Configuration par défaut
_GPU_CONFIG = {
    "enabled": False,
    "vram_gb": 24,
    "batch_size": 32,
    "max_parallel_models": 3,
    "use_fp16": True,
    "use_torch_compile": False,
    "gliner_preset": "best",
    "prefetch_models": True,
    "optimization_level": "high",  # low, medium, high
}

_DEBUG = os.getenv("NER_GPU_DEBUG", "0").lower() in {"1", "true", "yes"}

# --- CACHE GLOBAL POUR ÉVITER LES FUITES MÉMOIRE ---
_OPTIMIZED_MODEL_CACHE: Dict[Tuple[str, str, bool, bool], OptimizedGLiNERModel] = {}


def _log(msg: str) -> None:
    if _DEBUG:
        print(f"[ner_gpu_optimizer] {msg}")


def load_gpu_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Charge la configuration GPU depuis config.json ou les variables d'env."""
    config = dict(_GPU_CONFIG)
    
    if config_path is None:
        # Chercher config.json à la racine du projet ou dans config/
        # Ici on suppose qu'on est dans src/nodes/detection/ai_ner/
        # On remonte de 4 niveaux pour la racine pipegraph
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
        config_path = os.path.join(base_dir, "config.json")
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                full_config = json.load(f)
                if "ner_gpu" in full_config:
                    config.update(full_config["ner_gpu"])
            _log(f"Configuration GPU chargée depuis {config_path}")
        except Exception as e:
            _log(f"Erreur lors du chargement de config.json : {e}")
    
    if os.getenv("NER_GPU_ENABLED"):
        config["enabled"] = os.getenv("NER_GPU_ENABLED", "0").lower() in {"1", "true", "yes"}
    if os.getenv("NER_GPU_BATCH_SIZE"):
        try:
            config["batch_size"] = int(os.getenv("NER_GPU_BATCH_SIZE", "32"))
        except ValueError:
            pass
    if os.getenv("NER_GPU_VRAM_GB"):
        try:
            config["vram_gb"] = int(os.getenv("NER_GPU_VRAM_GB", "24"))
        except ValueError:
            pass
    if os.getenv("NER_GPU_COMPILE"):
        config["use_torch_compile"] = os.getenv("NER_GPU_COMPILE", "0").lower() in {"1", "true", "yes"}
    if os.getenv("NER_GPU_PARALLEL_MODELS"):
        try:
            config["max_parallel_models"] = int(os.getenv("NER_GPU_PARALLEL_MODELS", "3"))
        except ValueError:
            pass
    
    return config


GPU_CONFIG = load_gpu_config()


def auto_tune_batch_size(vram_gb: int, model_size: str = "medium") -> int:
    size_factor = {
        "small": 1.5,
        "medium": 1.0,
        "large": 0.6,
    }.get(model_size, 1.0)
    
    if vram_gb >= 24:
        base = 64
    elif vram_gb >= 16:
        base = 48
    elif vram_gb >= 12:
        base = 32
    elif vram_gb >= 8:
        base = 16
    else:
        base = 8
    
    return max(8, int(base * size_factor))


class OptimizedGLiNERModel:
    """Wrapper pour GLiNER avec optimisations GPU."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        use_fp16: bool = True,
        use_compile: bool = False,
    ):
        if not _GLINER_AVAILABLE:
            raise ImportError("GLiNER not available")
        
        self.model_name = model_name
        self.device = device
        self.use_fp16 = use_fp16
        self.use_compile = use_compile
        
        _log(f"Chargement de {model_name} sur {device} (FP16={use_fp16}, compile={use_compile})")
        
        self.model = GLiNER.from_pretrained(model_name)
        
        if hasattr(self.model, "model"):
            target = self.model.model
        else:
            target = self.model
        
        if hasattr(target, "to"):
            target.to(device)
        
        if use_fp16 and device == "cuda":
            try:
                target.half()
                _log(f"FP16 activé pour {model_name}")
            except Exception as e:
                _log(f"Échec FP16 pour {model_name}: {e}")
        
        if use_compile and _TORCH_AVAILABLE and hasattr(torch, "compile"):
            try:
                if hasattr(target, "token_rep_layer"):
                    target.token_rep_layer = torch.compile(target.token_rep_layer, mode="reduce-overhead")
                _log(f"torch.compile activé pour {model_name}")
            except Exception as e:
                _log(f"torch.compile échoué pour {model_name}: {e}")
    
    def predict_batch(
        self,
        texts: List[str],
        labels: List[str],
        threshold: float = 0.35,
    ) -> List[List[Dict[str, Any]]]:
        if not texts:
            return []
        
        # _log(f"Inférence batch de {len(texts)} textes avec {self.model_name}")
        
        results = []
        with torch.inference_mode() if _TORCH_AVAILABLE else _nullcontext():
            for text in texts:
                try:
                    ents = self.model.predict_entities(text, labels, threshold=threshold) or []
                    results.append(ents)
                except Exception as e:
                    _log(f"Erreur inférence pour texte: {e}")
                    results.append([])
        
        return results


class ParallelNERPipeline:
    """Pipeline NER parallélisé pour GPU puissant."""
    
    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        labels: Optional[List[str]] = None,
        device: str = "cuda",
        batch_size: int = 32,
        max_parallel_models: int = 3,
        use_fp16: bool = True,
        use_compile: bool = False,
        threshold: float = 0.35,
    ):
        self.device = device
        self.batch_size = batch_size
        self.max_parallel_models = max_parallel_models
        self.threshold = threshold
        self.labels = labels or GLINER_ALL_LABELS[:20]
        
        if model_names is None:
            model_names = _GLINER_PRESETS.get("best", ["urchade/gliner_medium-v2.1"])
        
        _log(f"Initialisation de {len(model_names)} modèles : {model_names}")
        
        self.models: List[OptimizedGLiNERModel] = []
        for name in model_names:
            try:
                # --- FIX: Utilisation du cache ---
                cache_key = (name, device, use_fp16, use_compile)
                if cache_key in _OPTIMIZED_MODEL_CACHE:
                    model = _OPTIMIZED_MODEL_CACHE[cache_key]
                    # _log(f"Modèle {name} récupéré du cache")
                else:
                    model = OptimizedGLiNERModel(
                        name,
                        device=device,
                        use_fp16=use_fp16,
                        use_compile=use_compile,
                    )
                    _OPTIMIZED_MODEL_CACHE[cache_key] = model
                
                self.models.append(model)
            except Exception as e:
                _log(f"Échec chargement {name}: {e}")
        
        _log(f"Pipeline initialisé avec {len(self.models)} modèles")
    
    def _process_single_model(
        self,
        model: OptimizedGLiNERModel,
        chunks: List[Tuple[int, int, str]],
    ) -> Dict[Tuple[int, int, str], float]:
        texts = [chunk[2] for chunk in chunks]
        batch_results = model.predict_batch(texts, self.labels, self.threshold)
        
        votes: Dict[Tuple[int, int, str], float] = {}
        weight = _GLINER_MODEL_WEIGHTS.get(model.model_name, 1.0)
        
        for (chunk_start, chunk_end, _), ents in zip(chunks, batch_results):
            for ent in ents:
                start, end = ent.get("start"), ent.get("end")
                if start is None or end is None:
                    if ent.get("text"):
                        idx = chunks[0][2].find(ent["text"])
                        if idx != -1:
                            start, end = idx, idx + len(ent["text"])
                
                if not isinstance(start, int) or not isinstance(end, int) or end <= start:
                    continue
                
                label = _normalize_gliner_label(str(ent.get("label") or ent.get("type")))
                if not label:
                    continue
                
                abs_start = chunk_start + start
                abs_end = chunk_start + end
                
                key = (abs_start, abs_end, label)
                votes[key] = votes.get(key, 0.0) + weight
        
        return votes
    
    def predict(self, text: str) -> List[Dict[str, Any]]:
        if not self.models:
            return []
        
        sent_spans = split_sentences(text)
        chunks = [(s, e, text[s:e]) for s, e in sent_spans]
        
        # _log(f"Traitement de {len(chunks)} chunks avec {len(self.models)} modèles")
        
        all_votes: Dict[Tuple[int, int, str], float] = {}
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=self.max_parallel_models) as executor:
            futures = {
                executor.submit(self._process_single_model, model, chunks): model
                for model in self.models
            }
            
            for future in as_completed(futures):
                model = futures[future]
                try:
                    votes = future.result()
                    for key, vote in votes.items():
                        all_votes[key] = all_votes.get(key, 0.0) + vote
                except Exception as e:
                    _log(f"Erreur avec modèle {model.model_name}: {e}")
        
        elapsed = time.time() - start_time
        # _log(f"Inférence terminée en {elapsed:.2f}s")
        
        results = [
            {
                "start": s,
                "end": e,
                "entity_group": lab,
                "votes": v,
            }
            for (s, e, lab), v in all_votes.items()
        ]
        
        results.sort(key=lambda x: (x["start"], x["end"]))
        return results
    
    def warm_up(self) -> None:
        _log("Warm-up des modèles...")
        dummy_text = "John Doe works at Acme Corp."
        _ = self.predict(dummy_text)
        _log("Warm-up terminé")


try:
    from contextlib import nullcontext as _nullcontext
except ImportError:
    from contextlib import contextmanager
    @contextmanager
    def _nullcontext():  # type: ignore
        yield


def create_optimized_pipeline(
    config: Optional[Dict[str, Any]] = None,
) -> Optional[ParallelNERPipeline]:
    if config is None:
        config = GPU_CONFIG
    
    if not config.get("enabled"):
        _log("Mode GPU désactivé")
        return None
    
    if not _TORCH_AVAILABLE or not torch.cuda.is_available():
        _log("PyTorch ou CUDA non disponible")
        return None
    
    if not _GLINER_AVAILABLE:
        _log("GLiNER non disponible")
        return None
    
    batch_size = config.get("batch_size", 32)
    vram_gb = config.get("vram_gb", 24)
    
    if config.get("optimization_level") == "high":
        batch_size = auto_tune_batch_size(vram_gb, "medium")
    
    preset = config.get("gliner_preset", "best")
    model_names = _GLINER_PRESETS.get(preset, _GLINER_PRESETS["balanced"])
    
    pipeline = ParallelNERPipeline(
        model_names=model_names,
        device="cuda",
        batch_size=batch_size,
        max_parallel_models=config.get("max_parallel_models", 3),
        use_fp16=config.get("use_fp16", True),
        use_compile=config.get("use_torch_compile", False),
    )
    
    if config.get("prefetch_models"):
        pipeline.warm_up()
    
    return pipeline


__all__ = [
    "OptimizedGLiNERModel",
    "ParallelNERPipeline",
    "create_optimized_pipeline",
    "load_gpu_config",
    "auto_tune_batch_size",
    "GPU_CONFIG",
]
