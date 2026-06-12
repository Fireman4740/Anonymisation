from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atlas_anno.dashboard.analysis import DashboardSummary, load_dashboard_data, summarize_dashboard_data
from atlas_anno.dashboard.indicators import INDICATORS, Indicator, format_value, guide_rows


# ---------------------------------------------------------------------------
# Palette de couleurs
# ---------------------------------------------------------------------------

_C = {
    "navy":       "#1e3a5f",
    "blue":       "#2d6a9f",
    "blue_light": "#e3f2fd",
    "green":      "#2e7d32",
    "green_mid":  "#43a047",
    "green_bg":   "#e8f5e9",
    "amber":      "#e65100",
    "amber_bg":   "#fff3e0",
    "red":        "#b71c1c",
    "red_mid":    "#ef5350",
    "red_bg":     "#ffebee",
    "gray":       "#546e7a",
    "gray_bg":    "#f5f7fa",
    "white":      "#ffffff",
}


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

def run_dashboard_command(*, batch: str = "pilot_100", strategy: str = "masking") -> None:
    app_path = Path(__file__).resolve()
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path), "--",
         "--batch", batch, "--strategy", strategy],
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


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_app(data_dir: Path, *, batch: str, strategy: str) -> None:
    try:
        import streamlit as st
    except ModuleNotFoundError as exc:  # pragma: no cover
        raise RuntimeError("streamlit is required for the Atlas dashboard") from exc

    st.set_page_config(page_title="Atlas_anno — audit", layout="wide", page_icon="🔒")
    _inject_css(st)

    data = load_dashboard_data(data_dir, batch=batch, strategy=strategy)
    summary = summarize_dashboard_data(data)

    _render_header(st, data.batch, data.strategy, str(data.data_dir))

    if summary.missing_files:
        st.warning(
            "⚠️ Artefacts manquants — certains indicateurs seront vides : "
            + " · ".join(summary.missing_files),
            icon="⚠️",
        )

    filters = _render_sidebar(st, summary, batch=batch, strategy=strategy)
    filtered_docs = _filter_documents(summary.document_rows, filters)

    _render_kpis(st, summary, filtered_docs)
    _render_health_verdict(st, summary)

    _render_reading_guide(st)

    tabs = st.tabs([
        "📊 Qualité dataset",
        "🔤 Diversité linguistique",
        "🔒 Anonymisation",
        "🎯 Ré-identification",
        "📄 Documents",
        "🤖 Logs LLM",
    ])
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


# ---------------------------------------------------------------------------
# CSS injection
# ---------------------------------------------------------------------------

def _inject_css(st: Any) -> None:
    st.markdown(f"""
    <style>
    /* ── Sidebar ─────────────────────────────────────────────────── */
    [data-testid="stSidebar"] {{
        background: {_C["navy"]};
    }}
    [data-testid="stSidebar"] * {{
        color: {_C["white"]} !important;
    }}
    [data-testid="stSidebar"] .stMultiSelect span {{
        background: {_C["blue"]} !important;
    }}

    /* ── KPI cards ───────────────────────────────────────────────── */
    .kpi-card {{
        border-radius: 14px;
        padding: 16px 18px;
        margin: 2px 0;
        border-left: 6px solid;
        font-family: inherit;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    }}
    .kpi-good  {{ border-color:{_C["green"]};  background:linear-gradient(135deg,{_C["green_bg"]},{_C["white"]}); }}
    .kpi-warn  {{ border-color:{_C["amber"]};  background:linear-gradient(135deg,{_C["amber_bg"]},{_C["white"]}); }}
    .kpi-bad   {{ border-color:{_C["red"]};    background:linear-gradient(135deg,{_C["red_bg"]},{_C["white"]}); }}
    .kpi-info  {{ border-color:{_C["blue"]};   background:linear-gradient(135deg,{_C["blue_light"]},{_C["white"]}); }}
    .kpi-label {{
        font-size:.70rem; font-weight:700; text-transform:uppercase;
        letter-spacing:.08em; color:{_C["gray"]}; margin-bottom:4px;
    }}
    .kpi-value {{
        font-size:1.85rem; font-weight:800; color:{_C["navy"]};
        line-height:1.1; letter-spacing:-.02em;
    }}
    .kpi-sub {{
        font-size:.68rem; color:{_C["gray"]}; margin-top:3px;
    }}

    /* ── Status pills ────────────────────────────────────────────── */
    .pill {{
        display:inline-block; padding:2px 10px; border-radius:12px;
        font-size:.72rem; font-weight:700; margin:1px;
    }}
    .pill-good {{ background:{_C["green_bg"]}; color:{_C["green"]}; }}
    .pill-warn {{ background:{_C["amber_bg"]}; color:{_C["amber"]}; }}
    .pill-bad  {{ background:{_C["red_bg"]};   color:{_C["red"]}; }}

    /* ── Section title bar ───────────────────────────────────────── */
    .section-bar {{
        border-left:5px solid {_C["blue"]};
        padding:6px 14px;
        margin:18px 0 8px;
        background:{_C["blue_light"]};
        border-radius:0 8px 8px 0;
    }}
    .section-bar span {{
        font-weight:700; color:{_C["navy"]}; font-size:1rem;
    }}
    .section-hint {{
        font-size:.78rem; color:{_C["gray"]}; margin-top:2px;
    }}

    /* ── Alert banners ───────────────────────────────────────────── */
    .banner-bad {{
        background:linear-gradient(135deg,{_C["red"]},{_C["red_mid"]});
        color:{_C["white"]};
        padding:14px 20px; border-radius:12px; margin:10px 0;
        font-weight:600;
    }}
    .banner-good {{
        background:linear-gradient(135deg,{_C["green"]},{_C["green_mid"]});
        color:{_C["white"]};
        padding:14px 20px; border-radius:12px; margin:10px 0;
        font-weight:600;
    }}
    .banner-warn {{
        background:linear-gradient(135deg,{_C["amber"]},#ff8c42);
        color:{_C["white"]};
        padding:14px 20px; border-radius:12px; margin:10px 0;
        font-weight:600;
    }}

    /* ── Misc ────────────────────────────────────────────────────── */
    .context-box {{
        background:{_C["gray_bg"]}; border:1px solid #dde3ea;
        border-radius:10px; padding:14px 18px; margin-bottom:14px;
        font-size:.83rem; color:{_C["gray"]};
        line-height:1.55;
    }}
    hr.divider {{ border:none; border-top:1px solid #dde3ea; margin:18px 0; }}
    </style>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------

def _render_header(st: Any, batch: str, strategy: str, data_dir: str) -> None:
    st.markdown(f"""
    <div style="
        background:linear-gradient(135deg,{_C["navy"]} 0%,{_C["blue"]} 100%);
        color:white; padding:24px 32px; border-radius:16px;
        margin-bottom:20px; box-shadow:0 4px 20px rgba(30,58,95,.30);
    ">
        <div style="font-size:1.6rem;font-weight:800;letter-spacing:-.02em;">
            🔒 Atlas_anno — audit du dataset
        </div>
        <div style="margin-top:8px;font-size:.88rem;opacity:.85;display:flex;gap:18px;flex-wrap:wrap;">
            <span>📦 Batch : <b>{batch}</b></span>
            <span>🛡 Stratégie : <b>{strategy}</b></span>
            <span>📁 Source : <code style="background:rgba(255,255,255,.15);padding:1px 6px;border-radius:4px;">{data_dir}</code></span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Health verdict banner
# ---------------------------------------------------------------------------

def _render_health_verdict(st: Any, summary: DashboardSummary) -> None:
    m = summary.report_metrics
    flags = summary.linguistic_flags

    problems: List[str] = []
    warnings_list: List[str] = []

    pv = _to_float_safe(m.get("privacy_score"))
    if pv is not None and pv < 0.60:
        problems.append(f"Confidentialité critique ({pv:.0%})")
    elif pv is not None and pv < 0.80:
        warnings_list.append(f"Confidentialité modérée ({pv:.0%})")

    rv = _to_float_safe(m.get("reid_top1"))
    if rv is not None and rv > 0.40:
        problems.append(f"ReID top-1 élevé ({rv:.0%} — risque de fuite)")
    elif rv is not None and rv > 0.20:
        warnings_list.append(f"ReID top-1 modéré ({rv:.0%})")

    if flags.get("self_bleu_collapse"):
        problems.append("self-BLEU > 0.90 (diversity collapse)")
    if flags.get("distinct_2_low"):
        warnings_list.append("distinct-2 < 0.15 (vocabulaire pauvre)")
    if flags.get("cell_coverage_low"):
        warnings_list.append("cell_coverage < 0.60 (couverture factorielle faible)")

    if problems:
        st.markdown(
            f'<div class="banner-bad">🚨 {len(problems)} problème(s) critique(s) : '
            + " — ".join(problems) + "</div>",
            unsafe_allow_html=True,
        )
    elif warnings_list:
        st.markdown(
            f'<div class="banner-warn">⚠️ {len(warnings_list)} point(s) d\'attention : '
            + " — ".join(warnings_list) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-good">✅ Tous les indicateurs clés sont dans les seuils cibles.</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar(st: Any, summary: DashboardSummary, *, batch: str, strategy: str) -> Dict[str, Any]:
    st.sidebar.markdown(f"""
    <div style="font-size:.95rem;font-weight:700;letter-spacing:.04em;
                border-bottom:1px solid rgba(255,255,255,.25);padding-bottom:10px;margin-bottom:12px;">
        🎛 Filtres
    </div>
    <div style="font-size:.72rem;opacity:.7;margin-bottom:14px;">
        S'appliquent aux onglets Anonymisation, Ré-identification et Documents.
    </div>
    """, unsafe_allow_html=True)

    st.sidebar.text_input("Batch", value=batch, disabled=True)
    st.sidebar.text_input("Stratégie", value=strategy, disabled=True)
    st.sidebar.markdown("<hr style='border:none;border-top:1px solid rgba(255,255,255,.2);margin:10px 0;'>",
                        unsafe_allow_html=True)

    docs = summary.document_rows
    filters = {
        "split": st.sidebar.multiselect(
            "Split", _options(docs, "split"), default=_options(docs, "split")
        ),
        "domain": st.sidebar.multiselect(
            "Domaine", _options(docs, "domain"), default=_options(docs, "domain")
        ),
        "difficulty": st.sidebar.multiselect(
            "Difficulté", _options(docs, "difficulty"), default=_options(docs, "difficulty")
        ),
        "register": st.sidebar.multiselect(
            "Registre", _options(docs, "register"), default=_options(docs, "register")
        ),
        "address_form": st.sidebar.multiselect(
            "Tu / Vous", _options(docs, "address_form"), default=_options(docs, "address_form")
        ),
        "aux_level": st.sidebar.multiselect(
            "Connaissance attaquant",
            _options(summary.reid_tables.get("attacks", []), "aux_level"),
            default=_options(summary.reid_tables.get("attacks", []), "aux_level"),
        ),
    }
    return filters


# ---------------------------------------------------------------------------
# KPI row
# ---------------------------------------------------------------------------

def _kpi_card(col: Any, label: str, value: str, status: str, sub: str = "") -> None:
    icons = {"good": "🟢", "warn": "🟡", "bad": "🔴", "info": "🔵"}
    icon = icons.get(status, "")
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    col.markdown(
        f'<div class="kpi-card kpi-{status}">'
        f'<div class="kpi-label">{icon} {label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f"{sub_html}"
        f"</div>",
        unsafe_allow_html=True,
    )


def _render_kpis(st: Any, summary: DashboardSummary, filtered_docs: List[Dict[str, Any]]) -> None:
    m = summary.report_metrics
    total = summary.coverage.get("raw_docs", 0)
    n = len(filtered_docs)

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    cols = st.columns(6)

    _kpi_card(cols[0], "Documents", str(n),
              "info" if n == total else "warn",
              f"{total} au total" if n != total else "tous inclus")

    pv = m.get("privacy_score")
    _kpi_card(cols[1], "Confidentialité ↑", _pct(pv),
              _health(pv, good=0.80, warn=0.60, higher_is_better=True),
              "objectif : élevée")

    uv = m.get("utility_score")
    _kpi_card(cols[2], "Utilité ↑", _pct(uv),
              _health(uv, good=0.70, warn=0.50, higher_is_better=True),
              "objectif : élevée")

    rv = m.get("reid_top1")
    _kpi_card(cols[3], "ReID top-1 ↓ ⚠", _pct(rv),
              _health(rv, good=0.20, warn=0.40, higher_is_better=False),
              "risque — bas = mieux")

    fv = m.get("span_f1")
    _kpi_card(cols[4], "Span F1 ↑", _pct(fv),
              _health(fv, good=0.70, warn=0.40, higher_is_better=True),
              "détection PII")

    bv = m.get("self_bleu")
    _kpi_card(cols[5], "self-BLEU ↓", _num(bv),
              _health(bv, good=0.80, warn=0.90, higher_is_better=False),
              "alerte > 0.90")

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Reading guide
# ---------------------------------------------------------------------------

def _render_reading_guide(st: Any) -> None:
    with st.expander("📖 Guide de lecture — qu'est-ce que chaque indicateur mesure ?", expanded=False):
        st.markdown(
            f"""<div class="context-box">
            Ce tableau de bord <b>audite un lot déjà généré</b> : il lit les artefacts de <code>data/</code>
            en lecture seule.<br><br>
            <b>🎯 Objectif d'une bonne anonymisation :</b> maximiser <b>Confidentialité ET Utilité</b>
            tout en gardant le <b>ReID top-1 bas</b>. Ces objectifs s'opposent — ce dashboard sert à
            visualiser l'arbitrage.<br><br>
            <b>Légende couleurs :</b>
            <span class="pill pill-good">🟢 OK</span>
            <span class="pill pill-warn">🟡 Attention</span>
            <span class="pill pill-bad">🔴 Problème</span>
            </div>""",
            unsafe_allow_html=True,
        )
        try:
            import pandas as pd
            df = pd.DataFrame(guide_rows())
            st.dataframe(df, use_container_width=True, hide_index=True)
        except Exception:
            st.table(guide_rows())
        st.caption("↑ = la valeur doit être élevée · ↓ = la valeur doit être basse (risque ou défaut)")


# ---------------------------------------------------------------------------
# Tab helpers
# ---------------------------------------------------------------------------

def _section(st: Any, title: str, hint: str = "") -> None:
    hint_html = f'<div class="section-hint">{hint}</div>' if hint else ""
    st.markdown(
        f'<div class="section-bar"><span>{title}</span>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def _context(st: Any, text: str) -> None:
    st.markdown(f'<div class="context-box">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tab 0 — Qualité dataset
# ---------------------------------------------------------------------------

def _render_quality_tab(st: Any, summary: DashboardSummary) -> None:
    _context(st,
        "Le dataset est-il <b>complet</b> (tous les artefacts présents), <b>propre</b> "
        "(zéro doublon, aucun champ requis absent) et <b>équilibré</b> entre domaines, "
        "difficultés et splits ?"
    )

    _section(st, "Couverture des artefacts",
             "Nombre d'enregistrements par étape de pipeline — un creux signale une étape non lancée.")
    _plotly_or_bar(st,
        [{"artifact": k, "count": v} for k, v in summary.coverage.items()],
        "artifact", "count", color="#2d6a9f",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        _section(st, "Duplicats",
                 "Doublons sur la clé naturelle. Tout doit valoir 0.")
        dup_rows = [{"artifact": k, "doublons": v} for k, v in summary.duplicates.items()]
        _dataframe(st, dup_rows, use_container_width=True)

    with col_b:
        _section(st, "Champs requis manquants",
                 "Champs obligatoires absents/vides — idéalement vide.")
        missing = summary.quality_tables.get("missing_required", [])
        if missing:
            st.markdown(
                f'<div class="banner-warn">⚠️ {len(missing)} champ(s) incomplets détectés</div>',
                unsafe_allow_html=True,
            )
            _dataframe(st, missing, use_container_width=True)
        else:
            st.markdown(
                '<div class="banner-good">✅ Aucun champ requis manquant</div>',
                unsafe_allow_html=True,
            )

    _section(st, "Distributions principales",
             "Des barres très déséquilibrées révèlent un biais de génération.")
    cols = st.columns(3)
    with cols[0]:
        st.caption("**Domaine**")
        _plotly_or_bar(st, summary.quality_tables.get("factor_domain", []), "domain", "count", color="#5c6bc0")
    with cols[1]:
        st.caption("**Difficulté**")
        _plotly_or_bar(st, summary.quality_tables.get("factor_difficulty", []), "difficulty", "count", color="#26a69a")
    with cols[2]:
        st.caption("**Split**")
        _plotly_or_bar(st, summary.quality_tables.get("factor_split", []), "split", "count", color="#ef5350")


# ---------------------------------------------------------------------------
# Tab 1 — Diversité linguistique
# ---------------------------------------------------------------------------

def _render_linguistic_tab(st: Any, summary: DashboardSummary) -> None:
    lm = summary.linguistic_metrics
    _context(st,
        "Les textes générés sont-ils <b>réellement variés</b> ou tournent-ils en rond ? "
        "La diversité est mesurée par une batterie de métriques complémentaires — un seul score "
        "n'est pas suffisant pour détecter le <em>diversity collapse</em>."
    )

    _section(st, "Batterie de diversité linguistique")
    cols = st.columns(6)
    _kpi_card(cols[0], "distinct-2 ↑", _num(lm.get("distinct_2")),
              _health(lm.get("distinct_2"), good=0.20, warn=0.15, higher_is_better=True),
              "bigrammes uniques")
    _kpi_card(cols[1], "distinct-1 ↑", _num(lm.get("distinct_1")),
              _health(lm.get("distinct_1"), good=0.50, warn=0.30, higher_is_better=True),
              "unigrammes uniques")
    _kpi_card(cols[2], "distinct-3 ↑", _num(lm.get("distinct_3")),
              _health(lm.get("distinct_3"), good=0.50, warn=0.30, higher_is_better=True),
              "trigrammes uniques")
    _kpi_card(cols[3], "self-BLEU ↓", _num(lm.get("self_bleu")),
              _health(lm.get("self_bleu"), good=0.80, warn=0.90, higher_is_better=False),
              "alerte > 0.90")
    _kpi_card(cols[4], "cell_coverage ↑", _num(lm.get("cell_coverage")),
              _health(lm.get("cell_coverage"), good=0.70, warn=0.60, higher_is_better=True),
              "cellules couvertes")
    _kpi_card(cols[5], "cell_entropy ↑", _num(lm.get("cell_entropy")),
              _health(lm.get("cell_entropy"), good=0.70, warn=0.40, higher_is_better=True),
              "équilibre répartition")

    st.markdown("<br>", unsafe_allow_html=True)
    flags = summary.linguistic_flags
    flag_labels = {
        "self_bleu_collapse": "self-BLEU > 0.90 — textes trop similaires (diversity collapse)",
        "distinct_2_low": "distinct-2 < 0.15 — vocabulaire trop pauvre",
        "cell_coverage_low": "cell_coverage < 0.60 — combinaisons factorielles peu couvertes",
    }
    flagged = [flag_labels.get(k, k) for k, v in flags.items() if v]
    if flagged:
        st.markdown(
            '<div class="banner-bad">🚨 Seuils de <em>diversity collapse</em> franchis :<br>'
            + "<br>".join(f"• {f}" for f in flagged) + "</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="banner-good">✅ Aucun seuil de <em>diversity collapse</em> franchi.</div>',
            unsafe_allow_html=True,
        )

    _section(st, "Longueur des textes",
             "Distribution en caractères et en tokens — repère les textes tronqués ou hors-format.")
    try:
        import pandas as pd
        df = pd.DataFrame([{
            "min (car.)": lm.get("text_length_min"),
            "médian (car.)": lm.get("text_length_median"),
            "moyen (car.)": lm.get("text_length_mean"),
            "max (car.)": lm.get("text_length_max"),
            "tokens (moy.)": lm.get("token_count_mean"),
        }])
        st.dataframe(df, use_container_width=True, hide_index=True)
    except Exception:
        pass

    _section(st, "Couverture factorielle Atlas",
             "Répartition sur chaque facteur de style — cases vides = combinaisons jamais générées.")
    cols = st.columns(3)
    factors = [
        ("factor_register",           "register",           "Registre",             "#5c6bc0"),
        ("factor_address_form",       "address_form",       "Tu / vous",            "#26a69a"),
        ("factor_francophone_variety","francophone_variety", "Variété francophone",  "#ec407a"),
        ("factor_expertise_level",    "expertise_level",    "Expertise",            "#ff7043"),
        ("factor_typo_propensity",    "typo_propensity",    "Typos",                "#8d6e63"),
        ("factor_document_goal",      "document_goal",      "Objectif document",    "#42a5f5"),
    ]
    for idx, (table_key, field, title, color) in enumerate(factors):
        with cols[idx % 3]:
            st.caption(f"**{title}**")
            _plotly_or_bar(st, summary.quality_tables.get(table_key, []), field, "count", color=color)


# ---------------------------------------------------------------------------
# Tab 2 — Anonymisation
# ---------------------------------------------------------------------------

def _render_anonymization_tab(st: Any, summary: DashboardSummary, filtered_docs: List[Dict[str, Any]]) -> None:
    _context(st,
        "<b>Comment</b> la stratégie transforme-t-elle les textes et <b>où</b> coûte-t-elle le plus "
        "en utilité ? <code>estimated_utility_loss</code> est un coût : plus élevé = document plus dégradé."
    )

    _section(st, "Actions d'anonymisation",
             "Fréquence de chaque type d'action — donne le profil opérationnel de la stratégie.")
    _plotly_or_bar(st, summary.anonymization_tables.get("actions", [])[:30],
                   "action", "count", color="#1e3a5f", horizontal=True)

    _section(st, "Documents les plus dégradés",
             "Triés par perte d'utilité décroissante — candidats à inspecter dans l'onglet Documents.")
    rows = summary.anonymization_tables.get("documents", [])
    visible_ids = {row["doc_id"] for row in filtered_docs}
    visible = [r for r in rows if r.get("doc_id") in visible_ids][:50]
    _dataframe_with_progress(st, visible,
        progress_cols={"estimated_utility_loss": (0.0, 1.0, "red"),
                       "estimated_privacy_gain": (0.0, 1.0, "green")})


# ---------------------------------------------------------------------------
# Tab 3 — Ré-identification
# ---------------------------------------------------------------------------

def _render_reid_tab(st: Any, summary: DashboardSummary, filters: Dict[str, Any]) -> None:
    _context(st,
        "Un attaquant peut-il retrouver <b>la personne</b> derrière un document anonymisé ? "
        "<code>top1</code> = ré-identifié au 1er rang, <code>top3</code> = présent dans le top 3. "
        "<b>Ce sont des risques : plus bas = mieux.</b>"
    )

    _section(st, "Risque par difficulté et niveau de connaissance de l'attaquant",
             "top1 élevé sur un segment = fuite de confidentialité ciblée.")
    rows = [
        r for r in summary.reid_tables.get("by_segment", [])
        if not filters.get("aux_level") or r.get("aux_level") in filters["aux_level"]
    ]
    _dataframe_with_progress(st, rows,
        progress_cols={"top1": (0.0, 1.0, "red"), "top3": (0.0, 1.0, "orange")})

    _section(st, "Cas les plus risqués (attaques réussies avec la plus forte confiance)",
             "Fuites les plus nettes — priorité de correction.")
    risky = summary.reid_tables.get("risky", [])
    _dataframe_with_progress(st, risky,
        progress_cols={"confidence": (0.0, 1.0, "red")})


# ---------------------------------------------------------------------------
# Tab 4 — Documents
# ---------------------------------------------------------------------------

def _render_documents_tab(st: Any, summary: DashboardSummary, filtered_docs: List[Dict[str, Any]]) -> None:
    _context(st,
        "Inspecte <b>un document précis</b> — compare l'original et la version anonymisée, "
        "voit le nombre de spans PII, les actions appliquées et si l'attaque a réussi."
    )

    _section(st, "Liste des documents")
    cols_to_show = (
        "doc_id", "domain", "split", "difficulty", "register", "address_form",
        "span_count", "actions_count", "estimated_utility_loss",
        "attack_top1_success", "attack_count",
    )
    table_rows = [{k: row[k] for k in cols_to_show} for row in filtered_docs]
    _dataframe_with_progress(st, table_rows[:200],
        progress_cols={"estimated_utility_loss": (0.0, 1.0, "red")})

    if not filtered_docs:
        return

    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    _section(st, "Inspection document")
    selected_id = st.selectbox(
        "Sélectionner un document",
        [row["doc_id"] for row in filtered_docs],
        label_visibility="collapsed",
    )
    selected = next(row for row in filtered_docs if row["doc_id"] == selected_id)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            f'<div style="font-weight:700;color:{_C["navy"]};margin-bottom:6px;">📄 Texte original</div>',
            unsafe_allow_html=True,
        )
        st.text_area("orig", selected.get("text", ""), height=280, label_visibility="collapsed")
    with col_b:
        st.markdown(
            f'<div style="font-weight:700;color:{_C["green"]};margin-bottom:6px;">🔒 Texte anonymisé</div>',
            unsafe_allow_html=True,
        )
        st.text_area("anon", selected.get("anonymized_text", ""), height=280, label_visibility="collapsed")

    st.caption(
        "`span_count` entités PII · `actions_count` transformations · "
        "`candidate_pool_size` taille vivier (plus grand = re-id plus dure) · "
        "`attack_top1_success` attaques ayant re-identifié ce document"
    )
    _dataframe(st, [{
        "span_count": selected.get("span_count"),
        "actions_count": selected.get("actions_count"),
        "candidate_pool_size": selected.get("candidate_pool_size"),
        "attack_top1_success": selected.get("attack_top1_success"),
        "attack_count": selected.get("attack_count"),
    }], use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 5 — Logs LLM
# ---------------------------------------------------------------------------

def _render_llm_tab(st: Any, summary: DashboardSummary) -> None:
    _context(st,
        "La génération s'est-elle bien passée côté LLM ? "
        "<code>fallback_used=True</code> = réponse LLM invalide rattrapée par le déterministe. "
        "Beaucoup de fallbacks ou d'erreurs = qualité potentiellement dégradée."
    )
    _section(st, "Appels par étape de pipeline")
    _plotly_or_bar(st, summary.llm_tables.get("steps", []), "step", "count",
                   color="#1e3a5f", horizontal=True)

    cols = st.columns(2)
    with cols[0]:
        _section(st, "Fallbacks")
        st.caption("True = LLM invalide, rattrapé par logique déterministe")
        _dataframe(st, summary.llm_tables.get("fallback", []), use_container_width=True)
    with cols[1]:
        _section(st, "Erreurs LLM")
        st.caption("True = appel LLM en échec complet")
        _dataframe(st, summary.llm_tables.get("errors", []), use_container_width=True)


# ---------------------------------------------------------------------------
# Chart helpers
# ---------------------------------------------------------------------------

def _plotly_or_bar(
    st: Any,
    rows: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    color: str = "#2d6a9f",
    horizontal: bool = False,
) -> None:
    if not rows:
        st.caption("Aucune donnée.")
        return
    try:
        import pandas as pd
        import plotly.express as px

        df = pd.DataFrame(rows)
        if x_key not in df.columns or y_key not in df.columns:
            _chart_or_table(st, rows, x_key, y_key)
            return

        if horizontal:
            fig = px.bar(
                df.sort_values(y_key, ascending=True),
                x=y_key, y=x_key, orientation="h",
                color_discrete_sequence=[color],
            )
        else:
            fig = px.bar(df, x=x_key, y=y_key, color_discrete_sequence=[color])

        fig.update_layout(
            height=max(220, len(df) * (26 if horizontal else 30)),
            margin=dict(l=0, r=0, t=8, b=8),
            xaxis_title="", yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=11),
        )
        fig.update_traces(marker_line_width=0)
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        _chart_or_table(st, rows, x_key, y_key)
    except Exception:
        _chart_or_table(st, rows, x_key, y_key)


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


def _dataframe(st: Any, rows: List[Dict[str, Any]], **kwargs: Any) -> None:
    if not rows:
        st.caption("Aucune donnée.")
        return
    try:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows), hide_index=True, **kwargs)
    except Exception:
        st.table(rows)


def _dataframe_with_progress(
    st: Any,
    rows: List[Dict[str, Any]],
    progress_cols: Dict[str, Tuple[float, float, str]] | None = None,
) -> None:
    if not rows:
        st.caption("Aucune donnée.")
        return
    try:
        import pandas as pd
        import streamlit as _st

        df = pd.DataFrame(rows)
        col_cfg: Dict[str, Any] = {}
        if progress_cols:
            for col, (lo, hi, fmt) in progress_cols.items():
                if col in df.columns:
                    col_cfg[col] = _st.column_config.ProgressColumn(
                        col, min_value=lo, max_value=hi, format="%.3f",
                    )
        st.dataframe(df, hide_index=True, use_container_width=True, column_config=col_cfg or None)
    except Exception:
        st.table(rows)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _filter_documents(rows: List[Dict[str, Any]], filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    filtered = rows
    for key in ("split", "domain", "difficulty", "register", "address_form"):
        values = filters.get(key)
        if values:
            filtered = [row for row in filtered if row.get(key) in values]
    return filtered


def _options(rows: Iterable[Dict[str, Any]], key: str) -> List[str]:
    return sorted({str(row.get(key)) for row in rows if row.get(key) not in (None, "")})


def _health(value: Any, *, good: float, warn: float, higher_is_better: bool) -> str:
    v = _to_float_safe(value)
    if v is None:
        return "info"
    if higher_is_better:
        return "good" if v >= good else ("warn" if v >= warn else "bad")
    return "good" if v <= good else ("warn" if v <= warn else "bad")


def _to_float_safe(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct(value: Any) -> str:
    v = _to_float_safe(value)
    return f"{v:.1%}" if v is not None else "n/a"


def _num(value: Any) -> str:
    v = _to_float_safe(value)
    return f"{v:.4f}" if v is not None else "n/a"


if __name__ == "__main__":
    raise SystemExit(main())
