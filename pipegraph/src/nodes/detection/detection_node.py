import sys
import os
import re
from typing import Dict, Any
import yaml

from src.state import PipelineState
from .deterministic.detector import DeterministicDetector
from .ai_ner.detector import AINerDetector
from src.utils.span_utils import merge_entity_lists
from src.utils.entity_utils import normalize_entity_profile, normalize_entity_type
import concurrent.futures

_SHORT_ACRONYM_RE = re.compile(r"^[A-Z](?:[A-Z0-9]|[.$&/'-])+$")


def _passes_ai_length_filter(entity: Dict[str, Any], min_len: int) -> bool:
    value = str(entity.get("value") or entity.get("text") or "").strip()
    length = int(entity.get("end", 0)) - int(entity.get("start", 0))
    if length >= min_len:
        return True
    if length < 2 or not value:
        return False
    return bool(_SHORT_ACRONYM_RE.fullmatch(value))

class DetectionNode:
    def __init__(self):
        self._pipegraph_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))

        # Chargement de la configuration globale
        config_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../config/pipeline_config.yaml"))
        self.global_config = {}
        try:
            if os.path.exists(config_path):
                with open(config_path, "r") as f:
                    self.global_config = yaml.safe_load(f) or {}
                print(f"✅ Configuration chargée depuis {config_path}")
            else:
                print(f"⚠️ Fichier de config introuvable: {config_path}")
        except Exception as e:
            print(f"❌ Erreur chargement config: {e}")

        # Initialisation du détecteur déterministe
        try:
            # On utilise le chemin défini dans la config globale ou une valeur par défaut
            # Note: la config est sous pipeline.nodes.detection.*
            yaml_det_config = self.global_config.get("pipeline", {}).get("nodes", {}).get("detection", {})
            det_cfg = yaml_det_config.get("deterministic", {}) if isinstance(yaml_det_config, dict) else {}

            # Compat: accepter plusieurs clés (ancienne vs nouvelle)
            patterns_path = (
                det_cfg.get("patterns_config_path")
                or det_cfg.get("patterns_file")
                or "config/patterns_config.yaml"
            )

            # Résolution robuste: si relatif, on le resolve depuis la racine pipegraph/
            if isinstance(patterns_path, str) and not os.path.isabs(patterns_path):
                patterns_path = os.path.abspath(os.path.join(self._pipegraph_root, patterns_path))

            print(f"ℹ️ Patterns config (deterministic): {patterns_path} (exists={os.path.exists(str(patterns_path))})")

            self.det_detector = DeterministicDetector(config_path=str(patterns_path))
            print("✅ DeterministicDetector chargé.")
        except Exception as e:
            print(f"❌ Erreur DeterministicDetector: {e}")
            self.det_detector = None

        # Initialisation du détecteur IA (NER)
        try:
            # Configuration depuis le fichier YAML
            yaml_det_config = self.global_config.get("pipeline", {}).get("nodes", {}).get("detection", {})
            ai_config = yaml_det_config.get("ai_ner", {}) if isinstance(yaml_det_config, dict) else {}
            
            # Fallback si la config est vide
            if not ai_config:
                ai_config = {
                    "use_gpu": False,
                    "preset": "balanced",
                    "threshold": 0.35
                }
            
            self.ai_detector = AINerDetector(config=ai_config)
            print("✅ AINerDetector chargé.")
        except Exception as e:
            print(f"❌ Erreur AINerDetector: {e}")
            self.ai_detector = None

    def __call__(self, state: PipelineState) -> Dict[str, Any]:
        """
        Exécute la détection (Déterministe + IA) sur le texte.
        Peut exécuter en parallèle si configuré.
        """
        print("--- Node: Detection (Hybrid) ---")
        
        # Config runtime (state)
        state_config = state.get("config", {})
        
        # Config statique (YAML)
        yaml_det_config = self.global_config.get("pipeline", {}).get("nodes", {}).get("detection", {})
        gliner_yaml_cfg = yaml_det_config.get("ai_ner", {}).get("gliner", {})
        
        # Vérification globale du noeud
        # Priorité: State > YAML > Default True
        node_enabled = state_config.get("enable_detection", yaml_det_config.get("enabled", True))
        
        if not node_enabled:
            print("🚫 Detection Node désactivé.")
            return {"entities": []}

        text = state["text"]
        entities_found = []
        
        # Mode d'exécution: "parallel" ou "serial"
        # Priorité: State > YAML > Default "serial"
        exec_mode = state_config.get("detection_mode", yaml_det_config.get("execution_mode", "serial"))
        print(f"🔍 Mode d'exécution: {exec_mode}")
        
        # Activation des sous-modules
        # Priorité: State > YAML > Default True
        enable_det = state_config.get("enable_deterministic", yaml_det_config.get("deterministic", {}).get("enabled", True))
        enable_ai = state_config.get("enable_ai", yaml_det_config.get("ai_ner", {}).get("enabled", True))
        
        det_results = []
        ai_results = []

        # ---------------------------------------------------------------
        # Préparation des overrides NER pour l'ablation
        # Ces clés dans state.config permettent de modifier le comportement
        # NER à la volée sans changer le YAML (utile pour run_ablation.py)
        # ---------------------------------------------------------------
        _NER_OVERRIDE_KEYS = (
            "gliner_preset",
            "gliner_models",
            "gliner_threshold",
            "entity_profile",
            "gliner_label_profile",
            "gliner_labels",
            "ner_provider",
        )
        ner_config_override: Dict[str, Any] = {
            k: state_config[k] for k in _NER_OVERRIDE_KEYS if k in state_config
        }

        entity_profile = normalize_entity_profile(
            state_config.get("entity_profile")
            or state_config.get("gliner_label_profile")
            or gliner_yaml_cfg.get("label_profile")
            or yaml_det_config.get("entity_profile")
            or "pii"
        )

        if exec_mode == "parallel":
            print("🚀 Lancement des détecteurs en parallèle...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                if self.det_detector and enable_det:
                    print("   -> Soumission DeterministicDetector")
                    futures[executor.submit(self.det_detector.detect, text)] = "deterministic"
                
                if self.ai_detector and enable_ai:
                    print("   -> Soumission AINerDetector")
                    futures[executor.submit(
                        self.ai_detector.detect, text, ner_config_override or None
                    )] = "ai"
                
                for future in concurrent.futures.as_completed(futures):
                    source = futures[future]
                    try:
                        res = future.result()
                        if source == "deterministic":
                            det_results = res
                            print(f"   ✅ DeterministicDetector terminé: {len(res)} entités")
                        else:
                            ai_results = res
                            print(f"   ✅ AINerDetector terminé: {len(res)} entités")
                    except Exception as e:
                        print(f"❌ Erreur dans le thread {source}: {e}")
        else:
            # Série
            print("🚀 Lancement des détecteurs en série...")
            if self.det_detector and enable_det:
                print("   -> Exécution DeterministicDetector...")
                det_results = self.det_detector.detect(text)
                print(f"   ✅ DeterministicDetector terminé: {len(det_results)} entités")
            elif not enable_det:
                print("   🚫 DeterministicDetector désactivé (config).")
            
            if self.ai_detector and enable_ai:
                print("   -> Exécution AINerDetector...")
                ai_results = self.ai_detector.detect(text, ner_config_override or None)
                print(f"   ✅ AINerDetector terminé: {len(ai_results)} entités")
            elif not enable_ai:
                print("   🚫 AINerDetector désactivé (config).")

        # Conversion + normalisation centralisée des types d'entités
        def _to_dict_normalized(ent) -> Dict[str, Any]:
            d = ent.to_dict()
            d["type"] = normalize_entity_type(d.get("type", ""), profile=entity_profile)
            return d

        det_dicts = [_to_dict_normalized(e) for e in det_results]
        ai_dicts  = [_to_dict_normalized(e) for e in ai_results]

        # Filtre de consensus : seuil de vote minimum.
        # Par défaut 1.0 = conserver toutes les détections d'au moins 1 modèle.
        # Augmenter (ex: 2.0) pour exiger le consensus de 2+ modèles (↓recall, ↑précision).
        # Configurable via state.config["ner_min_vote"].
        _MIN_VOTE = float(state_config.get("ner_min_vote", 1.0))
        # Filtre longueur minimum : éliminer les entités de moins de N chars
        _MIN_LEN = int(state_config.get("ner_min_len", 3))
        ai_dicts = [
            d for d in ai_dicts
            if float(d.get("score") or 0.0) >= _MIN_VOTE
            and _passes_ai_length_filter(d, _MIN_LEN)
        ]

        # Fusion intelligente avec déduplication et résolution des chevauchements
        # Les entités déterministes (source=regex) ont priorité sur l'IA (SOURCE_PRIORITY)
        entities_found = merge_entity_lists(
            det_dicts,
            ai_dicts,
            resolve_overlapping=True,
            strategy="priority_longest",
        )

        print(f"Entités trouvées: {len(entities_found)} après fusion (Det: {len(det_results)}, AI: {len(ai_results)})")

        return {"entities": entities_found}

