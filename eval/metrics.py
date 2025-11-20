import re

def calculate_overlap(span1, span2):
    """Calcule si deux intervalles se chevauchent."""
    start1, end1 = span1
    start2, end2 = span2
    return max(0, min(end1, end2) - max(start1, start2)) > 0

def compute_anonymization_metrics(predictions, ground_truth, strict=False):
    """
    Calcule Précision, Rappel et F2-Score.
    
    Args:
        predictions: Liste de tuples (start, end, type) détectés par ton API.
        ground_truth: Liste de tuples (start, end, type) du dataset (la vérité).
        strict: Si True, exige une correspondance exacte des indices. 
                Si False (recommandé pour l'anonymisation), accepte le chevauchement.
    """
    tp = 0  # Vrais Positifs (Entités correctement masquées)
    fp = 0  # Faux Positifs (Texte normal masqué par erreur - perte d'utilité)
    fn = 0  # Faux Négatifs (PII manquée - GRAVE !)

    # On fait une copie pour ne pas modifier les listes originales
    gt_copy = list(ground_truth)
    pred_copy = list(predictions)

    # Compter les Vrais Positifs (TP)
    for pred in list(pred_copy):
        match_found = False
        for gt in list(gt_copy):
            # Vérification de la position
            is_position_match = False
            if strict:
                is_position_match = (pred[0] == gt[0] and pred[1] == gt[1])
            else:
                is_position_match = calculate_overlap((pred[0], pred[1]), (gt[0], gt[1]))
            
            # On peut ajouter une vérification du type si nécessaire (ex: PER vs LOC)
            # Pour l'anonymisation pure, le type importe peu tant que c'est masqué.
            
            if is_position_match:
                tp += 1
                match_found = True
                gt_copy.remove(gt) # On ne compte pas deux fois la même vérité
                break
        
        if match_found:
            pred_copy.remove(pred)

    # Ce qui reste dans pred_copy sont des Faux Positifs (on a masqué pour rien)
    fp = len(pred_copy)
    # Ce qui reste dans gt_copy sont des Faux Négatifs (on a raté ces infos !)
    fn = len(gt_copy)

    # Calculs
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1 Score (Harmonique)
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # F2 Score (Pondère le Rappel 2x plus que la Précision - CRITIQUE pour l'anonymisation)
    beta = 2
    f2 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall) if ((beta**2 * precision) + recall) > 0 else 0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "f2": f2,
        "missed_entities": fn,
        "extra_redactions": fp
    }

def check_leakage(anonymized_text, original_text, ground_truth_spans):
    """
    Vérifie si le texte des entités sensibles est toujours présent dans la sortie.
    C'est l'ultime test de sécurité.
    """
    leaks = []
    for start, end, label in ground_truth_spans:
        sensitive_value = original_text[start:end]
        # Si la valeur sensible est toujours dans le texte anonymisé
        if sensitive_value in anonymized_text:
            # Attention: cela peut être un faux positif si le mot est commun (ex: "Paris")
            # Mais pour des noms ou ID, c'est une fuite.
            leaks.append((sensitive_value, label))
    return leaks