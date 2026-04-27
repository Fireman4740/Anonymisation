from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

from .ensemble import get_gliner_labels, run_gliner, warm_up_models
from .gpu_optimizer import create_optimized_pipeline, GPU_CONFIG
from src.utils.entity_utils import normalize_entity_profile

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
        self._default_gliner_profile = self._resolve_gliner_profile(self.config)

        # Lecture du sous-config gliner (support YAML nested: ai_ner.gliner.*)
        self._gliner_sub = self.config.get("gliner", {})
        self.use_gpu = self._gliner_sub.get("use_gpu", self.config.get("use_gpu", False))

        # --- GLiNER Setup ---
        if self.provider == "gliner":
            # Tentative d'initialisation du pipeline GPU optimisé si demandé
            if self.use_gpu:
                try:
                    # Merge des defaults GPU avec les overrides YAML
                    yaml_gpu_conf = self._gliner_sub.get(
                        "gpu_config", self.config.get("gpu_config", {})
                    )
                    gpu_conf = {**GPU_CONFIG, **yaml_gpu_conf}
                    gpu_conf["enabled"] = True  # forcé car use_gpu=True
                    gpu_conf["threshold"] = self.config.get("threshold", 0.35)
                    gpu_conf["label_profile"] = self._default_gliner_profile
                    gpu_conf["labels"] = get_gliner_labels(
                        profile=self._default_gliner_profile,
                        preset=self._gliner_sub.get(
                            "preset", self.config.get("preset", "balanced")
                        ),
                    )
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
                    _warmup_preset = self._gliner_sub.get(
                        "preset", self.config.get("preset", "balanced")
                    )
                    warm_up_models(gliner_preset=_warmup_preset)
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

    def detect(self, text: str, config_override: Optional[Dict[str, Any]] = None) -> List[AIEntity]:
        """
        Exécute la détection NER selon le provider configuré.

        Args:
            text: Texte à analyser.
            config_override: Surcharges runtime pour l'ablation (priorité sur self.config).
                Clés supportées :
                  - ``gliner_preset``   : "fast"|"balanced"|"accuracy"|"best"|"full"|"pii"
                  - ``gliner_models``   : List[str] — liste explicite de modèles GLiNER
                  - ``gliner_threshold``: float — seuil de confiance GLiNER
                  - ``gliner_label_profile`` / ``entity_profile`` : "pii"|"news_ner"|"conll2003"|"hybrid"
                  - ``gliner_labels``   : List[str] — labels GLiNER explicites
                  - ``ner_provider``    : "gliner"|"flair"|"spacy" — provider à la volée
        """
        eff = {**self.config, **(config_override or {})}
        provider = eff.get("ner_provider", self.provider)

        logger.debug(
            f"Début de l'analyse AI-NER (Provider: {provider}, GPU: {self.use_gpu})..."
        )
        entities = []

        try:
            if provider == "gliner":
                entities = self._detect_gliner(text, eff, config_override or {})
            elif provider == "flair":
                entities = self._detect_flair(text)
            elif provider == "spacy":
                entities = self._detect_spacy(text)
            else:
                logger.warning(f"Provider inconnu: {provider}")

        except Exception as e:
            logger.error(f"Erreur lors de la détection NER ({provider}): {e}")

        logger.debug(f"Fin de l'analyse AI-NER. {len(entities)} entités trouvées.")
        return entities

    def _resolve_gliner_profile(self, eff_config: Optional[Dict[str, Any]] = None) -> str:
        eff = eff_config or self.config
        _gliner_sub = eff.get("gliner", {})
        profile = (
            eff.get("gliner_label_profile")
            or eff.get("entity_profile")
            or _gliner_sub.get("label_profile")
            or eff.get("label_profile")
            or "pii"
        )
        return normalize_entity_profile(profile) or "pii"

    def _detect_gliner(
        self,
        text: str,
        eff_config: Optional[Dict[str, Any]] = None,
        runtime_override: Optional[Dict[str, Any]] = None,
    ) -> List[AIEntity]:
        """Détection GLiNER avec paramètres effectifs (self.config + overrides runtime)."""
        eff = eff_config or self.config
        entities = []
        _gliner_sub = eff.get("gliner", {})
        preset = (
            eff.get("gliner_preset")
            or _gliner_sub.get("preset")
            or self.config.get("preset", "balanced")
        )
        threshold = float(
            eff.get("gliner_threshold")
            or eff.get("threshold")
            or self.config.get("threshold", 0.35)
        )
        custom_models = eff.get("gliner_models") or None
        custom_labels = eff.get("gliner_labels") or None
        label_profile = self._resolve_gliner_profile(eff)

        override_keys = {
            "gliner_preset",
            "gliner_models",
            "gliner_threshold",
            "gliner_label_profile",
            "entity_profile",
            "gliner_labels",
        }
        has_runtime_gliner_override = any(k in (runtime_override or {}) for k in override_keys)

        if self.gpu_pipeline and not has_runtime_gliner_override:
            # Mode GPU Optimisé
            logger.debug("Utilisation du pipeline GPU optimisé")
            raw_results = self.gpu_pipeline.predict(text)
        else:
            # Mode Standard — les overrides runtime s'appliquent ici
            logger.debug(
                f"Utilisation du modèle standard "
                f"(Preset: {preset}, Threshold: {threshold}, "
                f"Custom models: {custom_models}, Label profile: {label_profile}, "
                f"Custom labels: {custom_labels})"
            )
            raw_results = run_gliner(
                text,
                preset=preset,
                threshold=threshold,
                model_names=custom_models,
                labels=custom_labels,
                label_profile=label_profile,
            )

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
