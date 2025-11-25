import sys
import os
from typing import Dict, Any
import yaml

from src.state import PipelineState
from .deterministic.detector import DeterministicDetector
from .ai_ner.detector import AINerDetector
import concurrent.futures

class DetectionNode:
    def __init__(self):
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
            patterns_path = self.global_config.get("detection", {}).get("deterministic", {}).get("patterns_file", "config/patterns_config.yaml")
            # Si le chemin est relatif, on essaie de le résoudre par rapport à la racine du projet si besoin
            # Pour l'instant on laisse tel quel car DeterministicDetector semble gérer les chemins relatifs
            self.det_detector = DeterministicDetector(config_path=patterns_path)
            print("✅ DeterministicDetector chargé.")
        except Exception as e:
            print(f"❌ Erreur DeterministicDetector: {e}")
            self.det_detector = None

        # Initialisation du détecteur IA (NER)
        try:
            # Configuration depuis le fichier YAML
            ai_config = self.global_config.get("detection", {}).get("ai_ner", {})
            
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

        if exec_mode == "parallel":
            print("🚀 Lancement des détecteurs en parallèle...")
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {}
                if self.det_detector and enable_det:
                    print("   -> Soumission DeterministicDetector")
                    futures[executor.submit(self.det_detector.detect, text)] = "deterministic"
                
                if self.ai_detector and enable_ai:
                    print("   -> Soumission AINerDetector")
                    futures[executor.submit(self.ai_detector.detect, text)] = "ai"
                
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
                ai_results = self.ai_detector.detect(text)
                print(f"   ✅ AINerDetector terminé: {len(ai_results)} entités")
            elif not enable_ai:
                print("   🚫 AINerDetector désactivé (config).")

        # Fusion des résultats
        # TODO: Stratégie de fusion plus intelligente (gestion des chevauchements)
        # Pour l'instant on concatène tout
        
        for ent in det_results:
            entities_found.append(ent.to_dict())
            
        for ent in ai_results:
            entities_found.append(ent.to_dict())

        print(f"Entités trouvées: {len(entities_found)} (Det: {len(det_results)}, AI: {len(ai_results)})")
        
        return {"entities": entities_found}


