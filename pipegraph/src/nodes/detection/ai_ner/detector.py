from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .ensemble import run_gliner, warm_up_models
from .gpu_optimizer import create_optimized_pipeline, GPU_CONFIG

# Imports optionnels pour les autres librairies IA (Legacy / Alternatives)
try:
    from schwifty import IBAN
except ImportError:
    IBAN = None

try:
    import phonenumbers
except ImportError:
    phonenumbers = None

try:
    from flair.models import SequenceTagger
    from flair.data import Sentence

    _FLAIR_AVAILABLE = True
except ImportError:
    SequenceTagger = None
    Sentence = None
    _FLAIR_AVAILABLE = False

try:
    import spacy

    _SPACY_AVAILABLE = True
except ImportError:
    spacy = None
    _SPACY_AVAILABLE = False

logger = logging.getLogger("AINerDetector")

# Use centralized thread-safe model cache for Flair/Spacy
try:
    from src.utils.model_cache import get_model_cache

    _MODEL_CACHE = get_model_cache("flair_spacy")
    _USE_NEW_CACHE = True
except ImportError:
    # Fallback to simple dict cache
    _MODEL_CACHE = {"flair": None, "spacy": None}
    _USE_NEW_CACHE = False


@dataclass
class AIEntity:
    start: int
    end: int
    value: str
    etype: str
    source: str
    score: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": self.start,
            "end": self.end,
            "value": self.value,
            "type": self.etype,
            "source": self.source,
            "score": self.score,
        }


class AINerDetector:
    """
    Détecteur d'entités basé sur l'IA (NER).
    Utilise GLiNER (via ensemble ou GPU optimizer).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.provider = self.config.get("provider", "gliner")  # gliner, flair, spacy
        self.gpu_pipeline = None
        self.use_gpu = self.config.get("use_gpu", False)

        # --- GLiNER Setup ---
        if self.provider == "gliner":
            # Tentative d'initialisation du pipeline GPU optimisé si demandé
            if self.use_gpu:
                try:
                    # On peut passer une config spécifique au GPU optimizer
                    gpu_conf = self.config.get("gpu_config", GPU_CONFIG)
                    gpu_conf["enabled"] = True
                    self.gpu_pipeline = create_optimized_pipeline(gpu_conf)
                    if self.gpu_pipeline:
                        logger.info("✅ Pipeline NER GPU optimisé chargé.")
                    else:
                        logger.warning(
                            "⚠️ Pipeline GPU demandé mais non chargé (fallback CPU/Standard)."
                        )
                except Exception as e:
                    logger.error(f"Erreur init GPU pipeline: {e}")

            # Warmup si nécessaire (pour le mode standard)
            if not self.gpu_pipeline and self.config.get("warmup", True):
                try:
                    warm_up_models(gliner_preset=self.config.get("preset", "balanced"))
                except Exception as e:
                    logger.warning(f"Warmup failed: {e}")

        # --- Flair Setup ---
        elif self.provider == "flair":
            self._load_flair()

        # --- Spacy Setup ---
        elif self.provider == "spacy":
            self._load_spacy()

    def _load_flair(self):
        """Charge le modèle Flair (Singleton via cache centralisé)."""
        if not _FLAIR_AVAILABLE:
            logger.error("Flair n'est pas installé.")
            return

        cache_key = "flair:ner-french"

        # Try to get from cache (new API)
        if _USE_NEW_CACHE:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                self.flair_tagger = cached
                return
        else:
            # Fallback dict API
            if _MODEL_CACHE.get("flair") is not None:
                self.flair_tagger = _MODEL_CACHE["flair"]
                return

        # Load model
        try:
            logger.info("Chargement du modèle Flair (ner-french)...")
            model = SequenceTagger.load("flair/ner-french")

            # Store in cache
            if _USE_NEW_CACHE:
                _MODEL_CACHE.put(cache_key, model)
            else:
                _MODEL_CACHE["flair"] = model

            self.flair_tagger = model
            logger.info("Modèle Flair chargé.")
        except Exception as e:
            logger.error(f"Erreur chargement Flair: {e}")
            self.flair_tagger = None

    def _load_spacy(self):
        """Charge le modèle Spacy (Singleton via cache centralisé)."""
        if not _SPACY_AVAILABLE:
            logger.error("Spacy n'est pas installé.")
            return

        model_name = self.config.get("spacy_model", "fr_core_news_sm")
        cache_key = f"spacy:{model_name}"

        # Try to get from cache (new API)
        if _USE_NEW_CACHE:
            cached = _MODEL_CACHE.get(cache_key)
            if cached is not None:
                self.spacy_nlp = cached
                return
        else:
            # Fallback dict API
            if _MODEL_CACHE.get("spacy") is not None:
                self.spacy_nlp = _MODEL_CACHE["spacy"]
                return

        # Load model
        try:
            logger.info(f"Chargement du modèle Spacy ({model_name})...")
            try:
                model = spacy.load(model_name)
            except OSError:
                # Fallback
                fallback = "fr_core_news_md"
                logger.warning(f"Modèle {model_name} introuvable, essai avec {fallback}...")
                model = spacy.load(fallback)
                cache_key = f"spacy:{fallback}"

            # Store in cache
            if _USE_NEW_CACHE:
                _MODEL_CACHE.put(cache_key, model)
            else:
                _MODEL_CACHE["spacy"] = model

            self.spacy_nlp = model
            logger.info("Modèle Spacy chargé.")
        except Exception as e:
            logger.error(f"Erreur chargement Spacy: {e}")
            self.spacy_nlp = None

    def detect(self, text: str) -> List[AIEntity]:
        """Exécute la détection NER selon le provider configuré."""
        logger.debug(
            f"Début de l'analyse AI-NER (Provider: {self.provider}, GPU: {self.use_gpu})..."
        )
        entities = []

        try:
            if self.provider == "gliner":
                entities = self._detect_gliner(text)
            elif self.provider == "flair":
                entities = self._detect_flair(text)
            elif self.provider == "spacy":
                entities = self._detect_spacy(text)
            else:
                logger.warning(f"Provider inconnu: {self.provider}")

        except Exception as e:
            logger.error(f"Erreur lors de la détection NER ({self.provider}): {e}")

        logger.debug(f"Fin de l'analyse AI-NER. {len(entities)} entités trouvées.")
        return entities

    def _detect_gliner(self, text: str) -> List[AIEntity]:
        entities = []
        if self.gpu_pipeline:
            # Mode GPU Optimisé
            logger.debug("Utilisation du pipeline GPU optimisé")
            raw_results = self.gpu_pipeline.predict(text)
        else:
            # Mode Standard (CPU ou GPU simple via ensemble.py)
            preset = self.config.get("preset", "balanced")
            threshold = self.config.get("threshold", 0.35)
            logger.debug(
                f"Utilisation du modèle standard (Preset: {preset}, Threshold: {threshold})"
            )
            raw_results = run_gliner(text, preset=preset, threshold=threshold)

        logger.debug(f"{len(raw_results)} résultats bruts GLiNER")

        # Conversion des résultats
        for res in raw_results:
            score = res.get("votes", res.get("score", 1.0))
            entities.append(
                AIEntity(
                    start=res["start"],
                    end=res["end"],
                    value=text[res["start"] : res["end"]],
                    etype=res["entity_group"],
                    source="gliner",
                    score=score,
                )
            )
        return entities

    def _detect_flair(self, text: str) -> List[AIEntity]:
        entities = []
        if not self.flair_tagger or not Sentence:
            return []

        try:
            logger.debug("Prédiction Flair...")
            sentence = Sentence(text)
            self.flair_tagger.predict(sentence)
            spans = sentence.get_spans("ner")
            logger.debug(f"{len(spans)} spans trouvés par Flair")
            for span in spans:
                entities.append(
                    AIEntity(
                        start=span.start_position,
                        end=span.end_position,
                        value=span.text,
                        etype=span.tag.upper(),
                        source="flair",
                        score=float(span.score or 0.0),
                    )
                )
        except Exception as e:
            logger.error(f"Erreur Flair predict: {e}")
        return entities

    def _detect_spacy(self, text: str) -> List[AIEntity]:
        entities = []
        if not self.spacy_nlp:
            return []

        try:
            logger.debug("Prédiction Spacy...")
            doc = self.spacy_nlp(text)
            logger.debug(f"{len(doc.ents)} entités trouvées par Spacy")
            for ent in doc.ents:
                entities.append(
                    AIEntity(
                        start=ent.start_char,
                        end=ent.end_char,
                        value=ent.text,
                        etype=ent.label_.upper(),
                        source="spacy",
                        score=1.0,  # Spacy ne donne pas de score de confiance par défaut facilement accessible
                    )
                )
        except Exception as e:
            logger.error(f"Erreur Spacy predict: {e}")
        return entities
