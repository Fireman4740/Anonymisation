from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import Any, Dict, List, Optional
from ..metrics import compute_dataset_metrics, compute_label_metrics

try:
    import plotly.express as px
except ImportError:
    px = None


def render_comparison(
    report_a: List[Dict[str, Any]],
    report_b: List[Dict[str, Any]],
    meta_a: Optional[Dict[str, Any]],
    meta_b: Optional[Dict[str, Any]],
) -> None:
    st.title("Comparaison A/B")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Modele A (Actuel)")
        if meta_a:
            st.caption(meta_a.get("subtitle", ""))
            st.caption(f"Source: {meta_a.get('source', 'Unknown')}")
    with col2:
        st.subheader("Modele B (Reference)")
        if meta_b:
            st.caption(meta_b.get("subtitle", ""))
            st.caption(f"Source: {meta_b.get('source', 'Unknown')}")

    # 1. Top-level Metrics with Delta
    metrics_a = compute_dataset_metrics(report_a)
    metrics_b = compute_dataset_metrics(report_b)

    st.markdown("### Métriques Globales")
    c1, c2, c3, c4 = st.columns(4)

    def _safe_get(m: Dict[str, Any], key: str) -> float:
        return float(m.get(key, 0.0))

    with c1:
        val_a = _safe_get(metrics_a, "avg_prec")
        val_b = _safe_get(metrics_b, "avg_prec")
        st.metric(
            "Précision",
            f"{val_a:.1%}",
            f"{val_a - val_b:.1%}",
            delta_color="normal",
        )
    with c2:
        val_a = _safe_get(metrics_a, "avg_rec")
        val_b = _safe_get(metrics_b, "avg_rec")
        st.metric(
            "Rappel",
            f"{val_a:.1%}",
            f"{val_a - val_b:.1%}",
            delta_color="normal",
        )
    with c3:
        val_a = _safe_get(metrics_a, "avg_f2")
        val_b = _safe_get(metrics_b, "avg_f2")
        st.metric(
            "F2-Score",
            f"{val_a:.1%}",
            f"{val_a - val_b:.1%}",
            delta_color="normal",
        )
    with c4:
        # Leaky docs: fewer is better, so inverse delta color logic if needed
        # But st.metric defaults: positive delta = green.
        # For leaks, if A > B (more leaks), that's bad (red).
        # We can use delta_color="inverse".
        val_a = int(metrics_a.get("leaky_docs", 0))
        val_b = int(metrics_b.get("leaky_docs", 0))
        st.metric(
            "Docs avec fuites",
            f"{val_a}",
            f"{val_a - val_b}",
            delta_color="inverse",
        )

    # 2. Per-Label Comparison
    st.markdown("### Performance par Label (F1-Score)")

    labels_a = compute_label_metrics(report_a)
    labels_b = compute_label_metrics(report_b)

    # Create a DataFrame for comparison
    data = []
    all_labels = sorted(list(set(labels_a.keys()) | set(labels_b.keys())))

    for label in all_labels:
        f1_a = labels_a.get(label, {}).get("f1", 0.0)
        f1_b = labels_b.get(label, {}).get("f1", 0.0)
        data.append(
            {
                "Label": label,
                "Modele A (F1)": f1_a,
                "Modele B (F1)": f1_b,
                "Difference": f1_a - f1_b,
            }
        )

    if data:
        df_cmp = pd.DataFrame(data)

        # Sort by biggest regression first
        df_cmp = df_cmp.sort_values("Difference", ascending=True)

        st.dataframe(
            df_cmp.style.format(
                {"Modele A (F1)": "{:.1%}", "Modele B (F1)": "{:.1%}", "Difference": "{:+.1%}"}
            ).background_gradient(subset=["Difference"], cmap="RdYlGn", vmin=-0.2, vmax=0.2)
        )

        # Bar chart for visual comparison
        if px is not None:
            df_melt = df_cmp.melt(
                id_vars=["Label", "Difference"],
                value_vars=["Modele A (F1)", "Modele B (F1)"],
                var_name="Model",
                value_name="F1 Score",
            )
            fig = px.bar(
                df_melt,
                x="Label",
                y="F1 Score",
                color="Model",
                barmode="group",
                title="Comparaison F1 par Label",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.caption("Plotly non installé : affichage d'un graphique simplifié.")
            chart_df = df_cmp.set_index("Label")[["Modele A (F1)", "Modele B (F1)"]]
            st.bar_chart(chart_df)

    # 3. Regressions List (Docs that got worse)
    st.markdown("### Régressions Détectées")

    # We define regression as:
    # 1. Leaky in A but NOT in B
    # 2. Or, more leaks in A than B

    # Map by doc_id
    map_a = {str(d["doc_id"]): d for d in report_a}
    map_b = {str(d["doc_id"]): d for d in report_b}

    regressions = []

    for doc_id, doc_a in map_a.items():
        doc_b = map_b.get(doc_id)
        if not doc_b:
            continue

        leaks_a = len(doc_a.get("leaks", []))
        leaks_b = len(doc_b.get("leaks", []))

        is_regression = False
        reason = ""

        if leaks_a > 0 and leaks_b == 0:
            is_regression = True
            reason = "Nouvelle fuite (etait propre)"
        elif leaks_a > leaks_b:
            is_regression = True
            reason = f"Plus de fuites ({leaks_b} -> {leaks_a})"

        if is_regression:
            regressions.append(
                {
                    "Doc ID": doc_id,
                    "Type": reason,
                    "Fuites A": leaks_a,
                    "Fuites B": leaks_b,
                    "Texte (Debut)": (doc_a.get("text", "")[:100] + "..."),
                }
            )

    if regressions:
        st.warning(f"{len(regressions)} documents présentent des régressions.")
        st.dataframe(pd.DataFrame(regressions))
    else:
        st.success("Aucune régression majeure détectée (pas de nouvelles fuites).")
