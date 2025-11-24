"""
Interface Streamlit pour l'analyse des erreurs d'anonymisation

Cette application permet de visualiser de manière intuitive les résultats
de l'évaluation du pipeline d'anonymisation, avec mise en évidence des erreurs
directement dans le texte.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union

import streamlit as st

# Ajouter le répertoire parent au path pour importer metrics
SCRIPT_DIR = Path(__file__).parent
EVAL_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(EVAL_DIR))

# Configuration de la page
st.set_page_config(
    page_title="Analyse des Erreurs d'Anonymisation",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constantes
REPORTS_DIR = EVAL_DIR / "evaluation" / "reports"


def load_available_reports() -> List[str]:
    """Charge la liste des rapports JSON disponibles."""
    if not REPORTS_DIR.exists():
        return []
    
    reports = []
    for file_path in REPORTS_DIR.glob("*_details.json"):
        dataset_name = file_path.stem.replace("report_", "").replace("_details", "")
        reports.append(dataset_name)
    
    return reports


def load_report_data(dataset_name: str) -> List[Dict[str, Any]]:
    """Charge les données d'un rapport JSON."""
    report_path = REPORTS_DIR / f"report_{dataset_name}_details.json"
    
    if not report_path.exists():
        st.error(f"Rapport introuvable: {report_path}")
        return []
    
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _has_overlap(span1: Tuple[int, int], span2: Tuple[int, int]) -> bool:
    """Vérifie si deux spans se chevauchent."""
    return max(0, min(span1[1], span2[1]) - max(span1[0], span2[0])) > 0


def _normalize_leak(leak: Union[Tuple, List, str]) -> str:
    """Normalise le format d'une fuite en extrayant le texte."""
    if isinstance(leak, (list, tuple)) and len(leak) >= 1:
        return leak[0] if isinstance(leak[0], str) else str(leak)
    return str(leak) if leak else ""


def _escape_html(text: str) -> str:
    """Échappe les caractères HTML."""
    return (text
            .replace('&', '&amp;')
            .replace('<', '&lt;')
            .replace('>', '&gt;')
            .replace('"', '&quot;')
            .replace("'", '&#39;')
            .replace('\n', '<br>'))


def highlight_text_with_entities(
    text: str,
    ground_truth: List[Tuple[int, int, str]],
    predictions: List[Tuple[int, int, str]],
    leaks: Union[List[Union[Tuple[str, str], str]], None] = None
) -> str:
    """
    Génère du HTML avec mise en évidence des entités:
    - Vert: True Positives (bien détectées)
    - Rouge: False Negatives (manquées)
    - Orange: False Positives (sur-détection)
    - Rose: Fuites
    """
    # Convertir les listes de tuples en ensembles pour comparaison
    gt_set = set(tuple(gt) for gt in ground_truth)
    pred_set = set(tuple(pred) for pred in predictions)
    
    # Calculer les TP, FP, FN
    tp = []
    fp = []
    fn = list(gt_set)
    
    for pred in predictions:
        pred_tuple = tuple(pred)
        match_found = False
        for gt in ground_truth:
            gt_tuple = tuple(gt)
            # Vérifier le chevauchement
            if _has_overlap((pred[0], pred[1]), (gt[0], gt[1])):
                tp.append(pred)
                if gt_tuple in fn:
                    fn.remove(gt_tuple)
                match_found = True
                break
        if not match_found:
            fp.append(pred)
    
    # Créer une liste de toutes les annotations avec leurs types
    annotations = []
    
    for entity in tp:
        annotations.append({
            'start': entity[0],
            'end': entity[1],
            'type': 'TP',
            'label': entity[2],
            'text': text[entity[0]:entity[1]]
        })
    
    for entity in fp:
        annotations.append({
            'start': entity[0],
            'end': entity[1],
            'type': 'FP',
            'label': entity[2],
            'text': text[entity[0]:entity[1]]
        })
    
    for entity in fn:
        annotations.append({
            'start': entity[0],
            'end': entity[1],
            'type': 'FN',
            'label': entity[2],
            'text': text[entity[0]:entity[1]]
        })
    
    # Ajouter les fuites
    if leaks:
        for leak in leaks:
            leak_text = _normalize_leak(leak)
            if leak_text:
                # Trouver toutes les occurrences de la fuite dans le texte
                start = 0
                while True:
                    idx = text.find(leak_text, start)
                    if idx == -1:
                        break
                    annotations.append({
                        'start': idx,
                        'end': idx + len(leak_text),
                        'type': 'LEAK',
                        'label': 'FUITE',
                        'text': leak_text
                    })
                    start = idx + len(leak_text)
    
    # Trier les annotations par position
    annotations.sort(key=lambda x: (x['start'], -x['end']))
    
    # Construire le HTML avec les annotations
    html_parts = []
    last_pos = 0
    
    # Fusionner les annotations qui se chevauchent
    merged_annotations = _merge_overlapping_annotations(annotations)
    
    for ann in merged_annotations:
        # Texte avant l'annotation
        if ann['start'] > last_pos:
            html_parts.append(_escape_html(text[last_pos:ann['start']]))
        
        # Annotation avec style
        color_map = {
            'TP': '#28a745',  # Vert
            'FN': '#dc3545',  # Rouge
            'FP': '#fd7e14',  # Orange
            'LEAK': '#e83e8c'  # Rose
        }
        
        label_map = {
            'TP': '✓',
            'FN': '✗ Manqué',
            'FP': '⚠ Sur-détection',
            'LEAK': '⚠ FUITE'
        }
        
        color = color_map.get(ann['type'], '#6c757d')
        label_prefix = label_map.get(ann['type'], '')
        
        html_parts.append(
            f'<mark style="background-color: {color}; color: white; padding: 2px 4px; '
            f'border-radius: 3px; font-weight: bold;" '
            f'title="{label_prefix} - {ann["label"]}">'
            f'{_escape_html(ann["text"])}'
            f'</mark>'
        )
        
        last_pos = ann['end']
    
    # Texte restant
    if last_pos < len(text):
        html_parts.append(_escape_html(text[last_pos:]))
    
    return ''.join(html_parts)


def _merge_overlapping_annotations(annotations: List[Dict]) -> List[Dict]:
    """Fusionne les annotations qui se chevauchent, en priorisant les fuites et FN."""
    if not annotations:
        return []
    
    # Priorité: LEAK > FN > TP > FP
    priority = {'LEAK': 0, 'FN': 1, 'TP': 2, 'FP': 3}
    
    result = []
    i = 0
    while i < len(annotations):
        current = annotations[i]
        
        # Chercher les chevauchements
        j = i + 1
        while j < len(annotations) and annotations[j]['start'] < current['end']:
            # Il y a chevauchement, garder celui avec la meilleure priorité
            if priority.get(annotations[j]['type'], 99) < priority.get(current['type'], 99):
                current = annotations[j]
            j += 1
        
        result.append(current)
        i = j if j > i + 1 else i + 1
    
    return result


def calculate_metrics_summary(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcule les métriques globales."""
    if not data:
        return {}
    
    total_docs = len(data)
    avg_precision = sum(d["precision"] for d in data) / total_docs
    avg_recall = sum(d["recall"] for d in data) / total_docs
    avg_f2 = sum(d["f2"] for d in data) / total_docs
    leaky_docs = sum(1 for d in data if d["leaks_count"] > 0)
    
    # Documents avec erreurs
    docs_with_fn = sum(1 for d in data if d["recall"] < 1.0)
    docs_with_fp = sum(1 for d in data if d["precision"] < 1.0)
    
    return {
        "total_docs": total_docs,
        "avg_precision": avg_precision,
        "avg_recall": avg_recall,
        "avg_f2": avg_f2,
        "leaky_docs": leaky_docs,
        "docs_with_fn": docs_with_fn,
        "docs_with_fp": docs_with_fp
    }


def main():
    """Point d'entrée principal de l'application."""
    st.title("🔍 Analyse des Erreurs d'Anonymisation")
    st.markdown("---")
    
    # Sidebar pour la sélection du rapport
    with st.sidebar:
        st.header("Configuration")
        
        available_reports = load_available_reports()
        
        if not available_reports:
            st.error("Aucun rapport trouvé. Exécutez d'abord `evaluate_pipeline.py`.")
            st.stop()
        
        selected_report = st.selectbox(
            "Choisir un rapport:",
            available_reports,
            index=0
        )
        
        st.markdown("---")
        st.header("Filtres")
        
        show_only_errors = st.checkbox("Afficher uniquement les documents avec erreurs", value=False)
        show_only_leaks = st.checkbox("Afficher uniquement les documents avec fuites", value=False)
        
        min_recall = st.slider("Rappel minimum", 0.0, 1.0, 0.0, 0.1)
        min_precision = st.slider("Précision minimum", 0.0, 1.0, 0.0, 0.1)
        
        st.markdown("---")
        st.header("Légende")
        st.markdown("""
        <div style='padding: 10px; background-color: #f8f9fa; border-radius: 5px;'>
            <p><mark style='background-color: #28a745; color: white; padding: 2px 4px;'>✓ TP</mark> True Positive (bien détecté)</p>
            <p><mark style='background-color: #dc3545; color: white; padding: 2px 4px;'>✗ FN</mark> False Negative (manqué)</p>
            <p><mark style='background-color: #fd7e14; color: white; padding: 2px 4px;'>⚠ FP</mark> False Positive (sur-détection)</p>
            <p><mark style='background-color: #e83e8c; color: white; padding: 2px 4px;'>⚠ LEAK</mark> Fuite détectée</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Charger les données
    data = load_report_data(selected_report)
    
    if not data:
        st.error("Impossible de charger les données du rapport.")
        st.stop()
    
    # Appliquer les filtres
    filtered_data = data
    
    if show_only_errors:
        filtered_data = [d for d in filtered_data if d["recall"] < 1.0 or d["precision"] < 1.0]
    
    if show_only_leaks:
        filtered_data = [d for d in filtered_data if d["leaks_count"] > 0]
    
    filtered_data = [d for d in filtered_data if d["recall"] >= min_recall and d["precision"] >= min_precision]
    
    # Afficher les métriques globales
    metrics = calculate_metrics_summary(data)
    
    st.header(f"📊 Métriques Globales - {selected_report}")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Documents Total", metrics["total_docs"])
    
    with col2:
        st.metric("Précision Moyenne", f"{metrics['avg_precision']:.2%}")
    
    with col3:
        st.metric("Rappel Moyen", f"{metrics['avg_recall']:.2%}")
    
    with col4:
        st.metric("F2 Moyen", f"{metrics['avg_f2']:.2%}")
    
    with col5:
        st.metric("Documents avec Fuites", metrics["leaky_docs"])
    
    col6, col7 = st.columns(2)
    with col6:
        st.metric("Docs avec False Negatives", metrics["docs_with_fn"])
    with col7:
        st.metric("Docs avec False Positives", metrics["docs_with_fp"])
    
    st.markdown("---")
    
    # Afficher le nombre de documents filtrés
    st.subheader(f"📄 Documents ({len(filtered_data)} / {len(data)})")
    
    if not filtered_data:
        st.info("Aucun document ne correspond aux filtres sélectionnés.")
        st.stop()
    
    # Options de tri
    sort_by = st.selectbox(
        "Trier par:",
        ["doc_id", "recall", "precision", "f2", "leaks_count"],
        index=0
    )
    
    sort_order = st.radio("Ordre:", ["Croissant", "Décroissant"], horizontal=True)
    
    sorted_data = sorted(
        filtered_data,
        key=lambda x: x[sort_by],
        reverse=(sort_order == "Décroissant")
    )
    
    # Pagination
    docs_per_page = st.slider("Documents par page", 1, 20, 5)
    total_pages = (len(sorted_data) - 1) // docs_per_page + 1
    
    page = st.number_input("Page", min_value=1, max_value=total_pages, value=1, step=1)
    
    start_idx = (page - 1) * docs_per_page
    end_idx = min(start_idx + docs_per_page, len(sorted_data))
    
    # Afficher les documents
    for i, doc in enumerate(sorted_data[start_idx:end_idx]):
        with st.expander(
            f"📄 Document {doc['doc_id']} - "
            f"P: {doc['precision']:.2%} | R: {doc['recall']:.2%} | F2: {doc['f2']:.2%} | "
            f"Fuites: {doc['leaks_count']}",
            expanded=(i == 0)
        ):
            # Métriques du document
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Précision", f"{doc['precision']:.2%}")
            with col2:
                st.metric("Rappel", f"{doc['recall']:.2%}")
            with col3:
                st.metric("Entités Prédites", doc['pred_count'])
            with col4:
                st.metric("Entités Réelles", doc['truth_count'])
            
            # Afficher le texte avec mise en évidence
            st.subheader("Texte avec Annotations")
            
            # Récupérer le texte complet depuis text_snippet (limité à 200 caractères dans le rapport)
            # Pour une meilleure visualisation, on utilise ce qui est disponible
            text = doc.get('text_snippet', '')
            ground_truth = doc.get('ground_truth', [])
            predictions = doc.get('predictions', [])
            leaks = doc.get('leaks', [])
            
            # Générer et afficher le HTML
            highlighted_html = highlight_text_with_entities(text, ground_truth, predictions, leaks)
            
            st.markdown(
                f'<div style="padding: 15px; background-color: #f8f9fa; '
                f'border-radius: 5px; font-family: monospace; white-space: pre-wrap; '
                f'line-height: 1.8;">{highlighted_html}</div>',
                unsafe_allow_html=True
            )
            
            # Détails des erreurs
            if doc['recall'] < 1.0 or doc['precision'] < 1.0 or doc['leaks_count'] > 0:
                st.subheader("Détails des Erreurs")
                
                # False Negatives
                gt_set = set(tuple(gt) for gt in ground_truth)
                pred_set = set(tuple(pred) for pred in predictions)
                
                fn_list = []
                for gt in ground_truth:
                    match_found = False
                    for pred in predictions:
                        if _has_overlap((gt[0], gt[1]), (pred[0], pred[1])):
                            match_found = True
                            break
                    if not match_found:
                        fn_list.append(gt)
                
                if fn_list:
                    st.error(f"❌ **False Negatives ({len(fn_list)})** - Entités manquées:")
                    for entity in fn_list:
                        st.write(f"  - `{text[entity[0]:entity[1]]}` [{entity[2]}] (pos: {entity[0]}-{entity[1]})")
                
                # False Positives
                fp_list = []
                for pred in predictions:
                    match_found = False
                    for gt in ground_truth:
                        if _has_overlap((pred[0], pred[1]), (gt[0], gt[1])):
                            match_found = True
                            break
                    if not match_found:
                        fp_list.append(pred)
                
                if fp_list:
                    st.warning(f"⚠️ **False Positives ({len(fp_list)})** - Sur-détections:")
                    for entity in fp_list:
                        st.write(f"  - `{text[entity[0]:entity[1]]}` [{entity[2]}] (pos: {entity[0]}-{entity[1]})")
                
                # Leaks
                if leaks:
                    st.error(f"🚨 **Fuites ({len(leaks)})** - Informations sensibles non masquées:")
                    for leak in leaks:
                        if isinstance(leak, (list, tuple)):
                            st.write(f"  - `{leak[0]}` [{leak[1] if len(leak) > 1 else 'UNKNOWN'}]")
                        else:
                            st.write(f"  - `{leak}`")
            
            st.markdown("---")


if __name__ == "__main__":
    main()
