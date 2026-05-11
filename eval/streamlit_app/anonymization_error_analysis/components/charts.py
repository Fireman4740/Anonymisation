from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st


def _render_distribution(data: List[Dict[str, Any]], field: str, title: str) -> None:
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is None:
        st.caption("Graphiques indisponibles (pandas manquant).")
        return

    df = pd.DataFrame(data)
    if field not in df.columns:
        return

    st.subheader(title)
    st.bar_chart(df[field])


def _render_leaks_distribution(data: List[Dict[str, Any]]) -> None:
    try:
        import pandas as pd
    except Exception:
        pd = None

    if pd is None:
        return

    df = pd.DataFrame(data)
    if "leaks_count" not in df.columns:
        return

    st.subheader("Fuites par document")
    st.bar_chart(df["leaks_count"])


def _render_advanced_charts(
    data: List[Dict[str, Any]], label_metrics: Dict[str, Dict[str, float]]
) -> None:
    try:
        import altair as alt
        import pandas as pd
    except ImportError:
        st.warning("Altair or Pandas not installed. Advanced charts unavailable.")
        return

    # --- 1. Scatter Plot: Precision vs Recall ---
    df_docs = pd.DataFrame(data)
    # Ensure columns exist and handle None
    if "precision" in df_docs.columns and "recall" in df_docs.columns:
        df_docs["precision"] = df_docs["precision"].fillna(0.0)
        df_docs["recall"] = df_docs["recall"].fillna(0.0)
        df_docs["doc_id_str"] = df_docs["doc_id"].astype(str)
        # Add snippet for tooltip, truncate for display
        df_docs["snippet"] = (
            df_docs.get("text_snippet", df_docs.get("full_text", "")).astype(str).str.slice(0, 100)
        )

        # Drop columns that can cause Arrow conversion issues (lists/dicts)
        cols_to_keep = ["precision", "recall", "leaks_count", "doc_id_str", "snippet"]
        plot_df = df_docs[cols_to_keep].copy()

        scatter = (
            alt.Chart(plot_df)
            .mark_circle(size=60)
            .encode(
                x=alt.X("recall", title="Recall", scale=alt.Scale(domain=[-0.1, 1.1])),
                y=alt.Y("precision", title="Precision", scale=alt.Scale(domain=[-0.1, 1.1])),
                tooltip=["doc_id_str", "precision", "recall", "snippet"],
                color=alt.Color("leaks_count", title="Leaks Count", scale=alt.Scale(scheme="reds")),
            )
            .properties(title="Precision vs Recall per Document", height=400)
            .interactive()
        )

        st.altair_chart(scatter, use_container_width=True)

    # --- 2. Bar Chart: Metrics per Entity Type ---
    if label_metrics:
        # Flatten dict for Altair: list of {label: "PER", metric: "precision", value: 0.9}
        flattened_data = []
        for label, metrics in label_metrics.items():
            flattened_data.append(
                {"Label": label, "Metric": "Precision", "Value": metrics["precision"]}
            )
            flattened_data.append({"Label": label, "Metric": "Recall", "Value": metrics["recall"]})
            flattened_data.append({"Label": label, "Metric": "F1", "Value": metrics["f1"]})

        df_labels = pd.DataFrame(flattened_data)

        bar_chart = (
            alt.Chart(df_labels)
            .mark_bar()
            .encode(
                x=alt.X("Label", title="Entity Label", sort="-y"),
                y=alt.Y("Value", title="Score", scale=alt.Scale(domain=[0, 1])),
                color="Metric",
                xOffset="Metric",  # Grouped bars
                tooltip=["Label", "Metric", "Value"],
            )
            .properties(title="Metrics per Entity Type")
        )

        st.altair_chart(bar_chart, use_container_width=True)


def render_charts(
    report: List[Dict[str, Any]], label_metrics: Dict[str, Dict[str, float]] | None = None
) -> None:
    if not report:
        return

    # Existing simple histograms (kept for backward compatibility or overview)
    st.markdown("### Distributions")
    col1, col2 = st.columns(2)
    with col1:
        _render_distribution(report, "recall", "Distribution du recall")
    with col2:
        _render_distribution(report, "precision", "Distribution de la precision")
    _render_leaks_distribution(report)

    # New Advanced Charts
    if label_metrics:
        st.markdown("### Detailed Analysis")
        _render_advanced_charts(report, label_metrics)


def render_llm_export_view(
    report: List[Dict[str, Any]],
    label_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    if not report:
        st.info("Aucune donnée disponible pour l'export.")
        return

    total_docs = len(report)
    leaky_docs = sum(1 for d in report if d.get("leaks_count", 0) > 0)

    precisions = [d.get("precision", 0.0) for d in report if d.get("precision") is not None]
    recalls = [d.get("recall", 0.0) for d in report if d.get("recall") is not None]

    avg_precision = (sum(precisions) / len(precisions)) if precisions else 0.0
    avg_recall = (sum(recalls) / len(recalls)) if recalls else 0.0

    lines = []
    lines.append(
        "Voici les métriques d'évaluation d'un modèle d'anonymisation sur un jeu de données de test."
    )

    # --- Section Présentation du Dataset ---
    if meta:
        dataset = meta.get("dataset", {})
        dataset_name = dataset.get("name") or meta.get("subtitle") or "N/A"
        split = dataset.get("split") or "N/A"
        limit = dataset.get("limit")
        limit_str = str(limit) if limit is not None else "Complet"
        entity_types = ", ".join(sorted(label_metrics.keys())) if label_metrics else "N/A"

        lines.append("")
        lines.append("### Présentation du Dataset")
        lines.append(f"- **Nom** : {dataset_name}")
        lines.append(f"- **Split** : {split}")
        lines.append(f"- **Nombre de documents évalués** : {total_docs}")
        lines.append(f"- **Limite appliquée** : {limit_str}")
        lines.append(f"- **Types d'entités** : {entity_types}")

        # --- Section Configuration du Run ---
        cfg = meta.get("config", {})
        profile = meta.get("profile") or cfg.get("profile", "N/A")
        eval_mode = meta.get("eval_mode") or cfg.get("eval_mode", "N/A")
        masking_mode = meta.get("masking_mode") or cfg.get("masking_mode", "N/A")

        def _yn(val: Any) -> str:
            return "oui" if val else "non"

        lines.append("")
        lines.append("### Configuration du Run")
        lines.append(f"- **Profil** : {profile}")
        lines.append(f"- **Mode évaluation** : {eval_mode}")
        lines.append(f"- **Mode masquage** : {masking_mode}")
        lines.append(f"- **Détection déterministe** : {_yn(cfg.get('enable_deterministic'))}")
        lines.append(f"- **Détection AI NER** : {_yn(cfg.get('enable_ai'))}")
        lines.append(f"- **LLM Détection** : {_yn(cfg.get('llm_detection'))}")
        lines.append(f"- **LLM Audit** : {_yn(cfg.get('llm_audit'))}")
        rupta = cfg.get("rupta_enabled") or cfg.get("rupta", {}).get("enabled", False)
        rupta_iter = cfg.get("rupta_max_iterations") or cfg.get("rupta", {}).get("max_iterations", "")
        rupta_thr = cfg.get("rupta_p_threshold") or cfg.get("rupta", {}).get("p_threshold", "")
        rupta_detail = f" (max_iter={rupta_iter}, seuil={rupta_thr})" if rupta else ""
        lines.append(f"- **RUPTA** : {_yn(rupta)}{rupta_detail}")
        lines.append(f"- **Anonymisation** : {_yn(cfg.get('enable_anonymization'))}")
    lines.append("")
    lines.append("### Statistiques Globales")
    lines.append(f"- **Nombre total de documents** : {total_docs}")
    lines.append(f"- **Documents avec fuites (leaks)** : {leaky_docs}")
    lines.append(f"- **Précision Moyenne (Global)** : {avg_precision:.4f}")
    lines.append(f"- **Rappel Moyen (Global)** : {avg_recall:.4f}")
    lines.append("")
    lines.append("### Performance par Label (Entité)")

    if label_metrics:
        # En-tête Markdown Table
        lines.append("| Label | Precision | Recall | F1-Score |")
        lines.append("|---|---|---|---|")
        for label, m in label_metrics.items():
            lines.append(
                f"| {label} | {m.get('precision', 0.0):.4f} | {m.get('recall', 0.0):.4f} | {m.get('f1', 0.0):.4f} |"
            )
    else:
        lines.append("_Aucune métrique par label disponible._")

    lines.append("")
    lines.append("### Demande d'Analyse")
    lines.append("En te basant sur ces métriques :")
    lines.append("1. Quels sont les points faibles du modèle ? (ex: types d'entités mal reconnus)")
    lines.append(
        "2. Observe-t-on un déséquilibre entre précision et rappel ? Qu'est-ce que cela implique pour l'anonymisation (risque de ré-identification vs sur-anonymisation) ?"
    )
    lines.append("3. Quelles recommandations proposerais-tu pour améliorer ce modèle ?")

    # --- Option: Inclure les détails des documents ---
    include_docs = st.checkbox("Inclure les détails des documents (Texte & Erreurs)", value=False)

    if include_docs:
        max_docs = st.number_input(
            "Nombre maximum de documents à inclure",
            min_value=1,
            max_value=len(report),
            value=min(20, len(report)),
        )
        lines.append("")
        lines.append("### Détails des Documents (Échantillon)")
        lines.append(f"Voici un échantillon de {max_docs} documents pour analyse contextuelle :")
        lines.append("")

        for i, doc in enumerate(report[:max_docs]):
            doc_id = doc.get("doc_id", "N/A")
            text = doc.get("full_text", doc.get("text", ""))
            leaks = doc.get("leaks", [])

            lines.append(f"#### Document ID: {doc_id}")
            lines.append(
                f"**Texte:**\n> {text[:500]}{'...' if len(text) > 500 else ''}"
            )  # Truncate to avoid exploding token limit

            if leaks:
                lines.append(f"**Fuites ({len(leaks)}):**")
                for leak in leaks:
                    lines.append(f"- {leak}")
            else:
                lines.append("**Fuites:** Aucune")

            lines.append("")
            lines.append("---")

    export_text = "\n".join(lines)

    st.markdown(
        "Copiez le texte ci-dessous pour l'utiliser dans un prompt LLM (ChatGPT, Claude, etc.) :"
    )
    st.text_area("Prompt LLM", value=export_text, height=400)
