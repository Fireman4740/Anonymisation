import json
import os
import sys

def validate_dataset(file_path, fix=False):
    """
    Vérifie si les offsets 'start' et 'end' correspondent au champ 'text' dans 'original_text'.
    Si fix=True, corrige les offsets en cherchant le texte dans l'original.
    """
    if not os.path.exists(file_path):
        print(f"Erreur : Le fichier {file_path} n'existe pas.")
        return

    with open(file_path, 'r', encoding='utf-8') as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError as e:
            print(f"Erreur de lecture JSON : {e}")
            return

    modified = False
    stats = {"total": 0, "correct": 0, "fixed": 0, "errors": 0}

    for example in data.get("examples", []):
        id = example.get("id", "unknown")
        original_text = example.get("original_text", "")
        
        for i, ann in enumerate(example.get("annotations", [])):
            stats["total"] += 1
            start = ann.get("start")
            end = ann.get("end")
            expected_text = ann.get("text")
            
            if start is None or end is None or expected_text is None:
                # Si le texte est présent mais pas les offsets, on peut essayer de les trouver
                if expected_text and fix:
                    new_start = original_text.find(expected_text)
                    if new_start != -1:
                        ann["start"] = new_start
                        ann["end"] = new_start + len(expected_text)
                        stats["fixed"] += 1
                        modified = True
                        print(f"[{id}] Offset ajouté pour '{expected_text}'")
                    else:
                        stats["errors"] += 1
                        print(f"[{id}] ERREUR : Texte '{expected_text}' introuvable")
                continue
                
            actual_text = original_text[start:end]
            
            if actual_text == expected_text:
                stats["correct"] += 1
            else:
                if fix:
                    # Recherche de l'occurrence la plus proche du start original
                    new_start = -1
                    occurrences = []
                    curr = original_text.find(expected_text)
                    while curr != -1:
                        occurrences.append(curr)
                        curr = original_text.find(expected_text, curr + 1)
                    
                    if occurrences:
                        new_start = min(occurrences, key=lambda x: abs(x - start))
                        new_end = new_start + len(expected_text)
                        
                        print(f"[{id}] FIX : '{expected_text}' déplacé de {start}:{end} vers {new_start}:{new_end}")
                        ann["start"] = new_start
                        ann["end"] = new_end
                        stats["fixed"] += 1
                        modified = True
                    else:
                        stats["errors"] += 1
                        print(f"[{id}] CRITIQUE : '{expected_text}' absent de original_text !")
                else:
                    stats["errors"] += 1
                    print(f"[{id}] ERREUR : Offset incorrect pour '{expected_text}' (trouvé '{actual_text}')")

    print("\n--- Rapport de validation ---")
    print(f"Total annotations : {stats['total']}")
    print(f"Correctes         : {stats['correct']}")
    print(f"Corrigées         : {stats['fixed']}")
    print(f"Erreurs restantes : {stats['errors']}")

    if modified:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        print(f"\nFichier mis à jour : {file_path}")

if __name__ == "__main__":
    path = "eval/datasets/data/anonymization_dataset.json"
    if len(sys.argv) > 1:
        path = sys.argv[1]
    
    validate_dataset(path, fix=True)
