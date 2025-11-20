#!/usr/bin/env python3
"""
Adaptateur pour le dataset TAB (Text Anonymization Benchmark).

Convertit les données TAB en format compatible avec le pipeline RUPTA.
Le dataset TAB contient des cas de la Cour Européenne des Droits de l'Homme
avec annotations détaillées (DIRECT/QUASI identifiers, confidential status).
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any

# Ajouter le répertoire parent au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_tab_dataset(file_path: str) -> List[Dict[str, Any]]:
    """Charge un fichier JSON du dataset TAB."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_applicant_name(task: str) -> str:
    """
    Extrait le nom de l'applicant depuis la tâche.
    
    Exemple: "Task: Annotate the document to anonymise the following person: Cengiz Polat"
    -> "Cengiz Polat"
    """
    if ":" in task and "person:" in task.lower():
        return task.split(":")[-1].strip()
    return "Unknown"


def extract_masked_entities(annotations: Dict[str, Any]) -> List[str]:
    """
    Extrait les entités à masquer (DIRECT ou QUASI identifiers).
    
    Returns:
        Liste de textes d'entités uniques
    """
    masked_entities = set()
    
    for annotator, ann_data in annotations.items():
        if "entity_mentions" in ann_data:
            for mention in ann_data["entity_mentions"]:
                # Ne garder que DIRECT et QUASI identifiers
                if mention.get("identifier_type") in ["DIRECT", "QUASI"]:
                    span_text = mention.get("span_text", "").strip()
                    if span_text:
                        masked_entities.add(span_text)
    
    return list(masked_entities)


def convert_tab_to_rupta_format(tab_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convertit les données TAB en format compatible RUPTA.
    
    Format RUPTA attendu:
    {
        "text": str,
        "people": str,  # Nom de la personne à protéger
        "label": str    # Occupation/attribut (pour utility)
    }
    """
    
    rupta_data = []
    
    for item in tab_data:
        # Extraire les informations de base
        doc_id = item.get("doc_id", "unknown")
        text = item.get("text", "")
        task = item.get("task", "")
        annotations = item.get("annotations", {})
        meta = item.get("meta", {})
        
        # Extraire le nom de l'applicant
        people = extract_applicant_name(task)
        
        # Extraire l'occupation depuis meta (si disponible)
        # Note: TAB ne contient pas toujours l'occupation explicite
        # On utilise "applicant" comme label générique
        label = meta.get("applicant", people)
        
        # Pour TAB, on peut aussi extraire le pays concerné
        countries = meta.get("respondent", []) if isinstance(meta.get("respondent"), list) else []
        
        rupta_item = {
            "doc_id": doc_id,
            "text": text,
            "people": people,
            "label": "applicant",  # Label générique pour TAB
            "dataset": "tab",
            "split": item.get("dataset_type", "unknown"),
            "meta": {
                "countries": countries,
                "masked_entities": extract_masked_entities(annotations),
                "original_annotations": len(annotations)
            }
        }
        
        rupta_data.append(rupta_item)
    
    return rupta_data


def create_evaluation_splits(tab_dir: Path, output_dir: Path):
    """
    Crée les fichiers d'évaluation train/dev/test au format RUPTA.
    """
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    splits = {
        "train": tab_dir / "echr_train.json",
        "dev": tab_dir / "echr_dev.json",
        "test": tab_dir / "echr_test.json"
    }
    
    stats = {}
    
    for split_name, file_path in splits.items():
        if not file_path.exists():
            print(f"⚠️  Fichier {file_path} introuvable, ignoré.")
            continue
        
        print(f"Traitement de {split_name}...")
        
        # Charger et convertir
        tab_data = load_tab_dataset(str(file_path))
        rupta_data = convert_tab_to_rupta_format(tab_data)
        
        # Sauvegarder au format JSONL (une ligne = un exemple)
        output_file = output_dir / f"{split_name}.jsonl"
        with open(output_file, 'w', encoding='utf-8') as f:
            for item in rupta_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        stats[split_name] = {
            "samples": len(rupta_data),
            "output_file": str(output_file)
        }
        
        print(f"  ✅ {len(rupta_data)} exemples -> {output_file}")
    
    # Sauvegarder les statistiques
    stats_file = output_dir / "conversion_stats.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Conversion terminée ! Statistiques : {stats_file}")
    
    return stats


def main():
    """Point d'entrée principal."""
    
    # Chemins
    tab_dir = Path(__file__).parent.parent / "Dataset" / "text-anonymization-benchmark"
    output_dir = Path(__file__).parent.parent / "Dataset" / "evaluation" / "TAB"
    
    print("="*60)
    print("CONVERSION DATASET TAB → FORMAT RUPTA")
    print("="*60)
    print(f"Source: {tab_dir}")
    print(f"Destination: {output_dir}")
    print()
    
    # Vérifier que le répertoire TAB existe
    if not tab_dir.exists():
        print(f"❌ Erreur: Répertoire {tab_dir} introuvable!")
        print("Assurez-vous que le dataset TAB est bien placé dans Dataset/text-anonymization-benchmark/")
        return 1
    
    # Convertir les fichiers
    stats = create_evaluation_splits(tab_dir, output_dir)
    
    # Afficher le résumé
    print("\n" + "="*60)
    print("RÉSUMÉ")
    print("="*60)
    
    total_samples = sum(s["samples"] for s in stats.values())
    print(f"Total échantillons: {total_samples}")
    
    for split_name, split_stats in stats.items():
        print(f"  - {split_name}: {split_stats['samples']} exemples")
    
    print("\n💡 Pour évaluer sur TAB:")
    print("  python scripts/eval_rupta_pipeline.py \\")
    print("      --dataset tab \\")
    print("      --split test \\")
    print("      --n_samples 50 \\")
    print("      --use_rupta \\")
    print("      --output results/eval_tab_test.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
