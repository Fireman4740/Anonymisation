from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atlas_anno.dashboard.analysis import DashboardSummary, load_dashboard_data, summarize_dashboard_data


def run_dashboard_command(*, batch: str = "pilot_100", strategy: str = "masking") -> None:
    app_path = Path(__file__).resolve()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(app_path),
            "--",
            "--batch",
            batch,
            "--strategy",
            strategy,
        ],
        check=True,
    )


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Atlas_anno Streamlit dashboard")
    parser.add_argument("--batch", default="pilot_100")
    parser.add_argument("--strategy", default="masking")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args(argv)
    render_app(Path(args.data_dir), batch=args.batch, strategy=args.strategy)
    return 0


def render_app(data_dir: Path, *, batch: str, strategy: str) -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover - only visible at runtime.
        raise RuntimeError("streamlit is required for the Atlas dashboard") from exc

    st.set_page_config(page_title="Atlas_anno audit", layout="wide")

    data = load_dashboard_data(data_dir, batch=batch, strategy=strategy)
    summary = summarize_dashboard_data(data)

    st.title("Atlas_anno audit")
    st.caption(f"Batch `{data.batch}` | stratégie `{data.strategy}` | source `{data.data_dir}`")

    filters = _render_sidebar(st, summary, batch=batch, strategy=strategy)
    filtered_docs = _filter_documents(summary.document_rows, filters)

    _render_kpis(st, summary, filtered_docs)
    if summary.missing_files:
        st.warning("Artefacts manquants: " + ", ".join(summary.missing_files))

    tabs = st.tabs(
        [
            "Qualité dataset",
            "Diversité linguistique",
            "Anonymisation",
            "Ré-identification",
            "Documents",
            "Logs LLM",
        ]
    )
    with tabs[0]:
        _render_quality_tab(st, summary)
    with tabs[1]:
        _render_linguistic_tab(st, summary)
    with tabs[2]:
        _render_anonymization_tab(st, summary, filtered_docs)
    with tabs[3]:
        _render_reid_tab(st, summary, filters)
    with tabs[4]:
        _render_documents_tab(st, summary, filtered_docs)
    with tabs[5]:
        _render_llm_tab(st, summary)


def _render_sidebar(st: Any, summary: DashboardSummary, *, batch: str, strategy: str) -> Dict[str, Any]:
    st.sidebar.header("Filtres")
    st.sidebar.text_input("Batch", value=batch, disabled=True)
    st.sidebar.text_input("Stratégie", value=strategy, disabled=True)
    docs = summary.document_rows
    filters = {
        "split": st.sidebar.multiselect("Split", _options(docs, "split"), default=_options(docs, "split")),
        "domain": st.sidebar.multiselect("Domaine", _options(docs, "domain"), default=_options(docs, "domain")),
        "difficulty": st.sidebar.multiselect(
            "Difficulté", _options(docs, "difficulty"), default=_options(docs, "difficulty")
        ),
        "register": st.sidebar.multiselect(
            "Registre", _options(docs, "register"), default=_options(docs, "register")
        ),
        "address_form": st.sidebar.multiselect(
            "Tu/vous", _options(docs, "address_form"), default=_options(docs, "address_form")
        ),
        "aux_level": st.sidebar.multiselect(
            "Niveau attaque",
            _options(summary.reid_tables.get("attacks", []), "aux_level"),
            default=_options(summary.reid_tables.get("attacks", []), "aux_level"),
        ),
    }
    return filters


def _render_kpis(st: Any, summary: DashboardSummary, filtered_docs: List[Dict[str, Any]]) -> None:
    metrics = summary.report_metrics
    human_review_count = sum(1 for row in summary.document_rows if row.get("span_count", 0) > 0)
    cols = st.columns(7)
    cols[0].metric("Documents", len(filtered_docs), f"{summary.coverage.get('raw_docs', 0)} total")
    cols[1].metric("Privacy", _pct(metrics.get("privacy_score")))
    cols[2].metric("Utility", _pct(metrics.get("utility_score")))
    cols[3].metric("ReID top1", _pct(metrics.get("reid_top1")))
    cols[4].metric("Span F1", _pct(metrics.get("span_f1")))
    cols[5].metric("Self-BLEU", _num(metrics.get("self_bleu")))
    cols[6].metric("Review", _pct(human_review_count / max(1, summary.coverage.get("annotations", 0))))


def _render_quality_tab(st: Any, summary: DashboardSummary) -> None:
    st.subheader("Couverture des artefacts")
    _chart_or_table(st, [{"artifact": key, "count": value} for key, value in summary.coverage.items()], "artifact", "count")
    st.subheader("Duplicats au grain attendu")
    st.table([{"artifact": key, "duplicates": value} for key, value in summary.duplicates.items()])
    st.subheader("Champs requis manquants")
    _table(st, summary.quality_tables.get("missing_required", []), empty="Aucun champ requis manquant détecté.")
    st.subheader("Distributions principales")
    cols = st.columns(3)
    with cols[0]:
        _chart_or_table(st, summary.quality_tables.get("factor_domain", []), "domain", "count")
    with cols[1]:
        _chart_or_table(st, summary.quality_tables.get("factor_difficulty", []), "difficulty", "count")
    with cols[2]:
        _chart_or_table(st, summary.quality_tables.get("factor_split", []), "split", "count")


def _render_linguistic_tab(st: Any, summary: DashboardSummary) -> None:
    metrics = summary.linguistic_metrics
    st.subheader("Batterie de diversité linguistique")
    cols = st.columns(6)
    cols[0].metric("distinct-1", _num(metrics.get("distinct_1")))
    cols[1].metric("distinct-2", _num(metrics.get("distinct_2")))
    cols[2].metric("distinct-3", _num(metrics.get("distinct_3")))
    cols[3].metric("self-BLEU", _num(metrics.get("self_bleu")))
    cols[4].metric("cell coverage", _num(metrics.get("cell_coverage")))
    cols[5].metric("cell entropy", _num(metrics.get("cell_entropy")))

    flags = summary.linguistic_flags
    flagged = [name for name, enabled in flags.items() if enabled]
    if flagged:
        st.error("Signaux de diversity collapse: " + ", ".join(flagged))
    else:
        st.success("Aucun seuil de diversity collapse franchi.")

    st.caption(
        "Méthode: génération native FR, pas traduction; diversité conçue par facteurs explicites "
        "et évaluée par une batterie de métriques plutôt qu'un score unique."
    )
    st.subheader("Longueur des textes")
    st.table(
        [
            {
                "min": metrics.get("text_length_min"),
                "median": metrics.get("text_length_median"),
                "mean": metrics.get("text_length_mean"),
                "max": metrics.get("text_length_max"),
                "tokens_mean": metrics.get("token_count_mean"),
            }
        ]
    )
    st.subheader("Couverture factorielle Atlas")
    cols = st.columns(3)
    factors = [
        ("factor_register", "register", "Registre"),
        ("factor_address_form", "address_form", "Tu/vous"),
        ("factor_francophone_variety", "francophone_variety", "Variété francophone"),
        ("factor_expertise_level", "expertise_level", "Expertise"),
        ("factor_typo_propensity", "typo_propensity", "Typos"),
        ("factor_document_goal", "document_goal", "Objectif"),
    ]
    for index, (table_key, field, title) in enumerate(factors):
        with cols[index % 3]:
            st.markdown(f"**{title}**")
            _chart_or_table(st, summary.quality_tables.get(table_key, []), field, "count")


def _render_anonymization_tab(st: Any, summary: DashboardSummary, filtered_docs: List[Dict[str, Any]]) -> None:
    st.subheader("Actions d'anonymisation")
    _chart_or_table(st, summary.anonymization_tables.get("actions", [])[:30], "action", "count")
    st.subheader("Documents avec perte d'utilité estimée élevée")
    rows = summary.anonymization_tables.get("documents", [])
    visible_ids = {row["doc_id"] for row in filtered_docs}
    _table(st, [row for row in rows if row.get("doc_id") in visible_ids][:50])


def _render_reid_tab(st: Any, summary: DashboardSummary, filters: Dict[str, Any]) -> None:
    st.subheader("Risque par difficulté et niveau de connaissance")
    rows = [
        row
        for row in summary.reid_tables.get("by_segment", [])
        if not filters.get("aux_level") or row.get("aux_level") in filters["aux_level"]
    ]
    _table(st, rows)
    st.subheader("Attaques top1 réussies les plus confiantes")
    _table(st, summary.reid_tables.get("risky", [])[:50])


def _render_documents_tab(st: Any, summary: DashboardSummary, filtered_docs: List[Dict[str, Any]]) -> None:
    st.subheader("Documents")
    table_rows = [
        {
            key: row[key]
            for key in (
                "doc_id",
                "domain",
                "split",
                "difficulty",
                "register",
                "address_form",
                "span_count",
                "actions_count",
                "estimated_utility_loss",
                "attack_top1_success",
                "attack_count",
            )
        }
        for row in filtered_docs
    ]
    _table(st, table_rows[:200])
    if not filtered_docs:
        return
    selected_id = st.selectbox("Inspecter un document", [row["doc_id"] for row in filtered_docs])
    selected = next(row for row in filtered_docs if row["doc_id"] == selected_id)
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Texte original**")
        st.text_area("original", selected.get("text", ""), height=260, label_visibility="collapsed")
    with col_b:
        st.markdown("**Texte anonymisé**")
        st.text_area("anonymized", selected.get("anonymized_text", ""), height=260, label_visibility="collapsed")
    st.table(
        [
            {
                "span_count": selected.get("span_count"),
                "actions_count": selected.get("actions_count"),
                "candidate_pool_size": selected.get("candidate_pool_size"),
                "attack_top1_success": selected.get("attack_top1_success"),
                "attack_count": selected.get("attack_count"),
            }
        ]
    )


def _render_llm_tab(st: Any, summary: DashboardSummary) -> None:
    st.subheader("Étapes LLM")
    _chart_or_table(st, summary.llm_tables.get("steps", []), "step", "count")
    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Fallbacks**")
        _table(st, summary.llm_tables.get("fallback", []))
    with cols[1]:
        st.markdown("**Erreurs**")
        _table(st, summary.llm_tables.get("errors", []))


def _filter_documents(rows: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    filtered = rows
    for key in ("split", "domain", "difficulty", "register", "address_form"):
        values = filters.get(key)
        if values:
            filtered = [row for row in filtered if row.get(key) in values]
    return filtered


def _options(rows: Iterable[Dict[str, Any]], key: str) -> List[str]:
    return sorted({str(row.get(key)) for row in rows if row.get(key) not in (None, "")})


def _chart_or_table(st: Any, rows: List[Dict[str, Any]], x_key: str, y_key: str) -> None:
    if not rows:
        st.caption("Aucune donnée.")
        return
    try:
        import pandas as pd

        frame = pd.DataFrame(rows)
        if x_key in frame.columns and y_key in frame.columns:
            st.bar_chart(frame.set_index(x_key)[y_key])
        else:
            st.table(rows)
    except Exception:
        st.table(rows)


def _table(st: Any, rows: List[Dict[str, Any]], *, empty: str = "Aucune donnée.") -> None:
    if not rows:
        st.caption(empty)
        return
    st.table(rows)


def _pct(value: Any) -> str:
    try:
        return f"{float(value):.1%}"
    except (TypeError, ValueError):
        return "n/a"


def _num(value: Any) -> str:
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
