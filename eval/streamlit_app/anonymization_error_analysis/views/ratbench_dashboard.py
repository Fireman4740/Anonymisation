"""Tableau de bord Streamlit dédié aux résultats RAT-Bench."""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import streamlit as st

from ..components.doc_inspector import render_doc_inspector
from ..components.doc_table import render_doc_table
from ..components.filters import render_doc_filters


def render_ratbench_dashboard(
    result: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Affiche l'intégralité des résultats d'un run RAT-Bench."""
    summary = result.get("summary", {})
    by_difficulty = result.get("by_difficulty", {})
    by_scenario = result.get("by_scenario", {})
    direct_id_rates = result.get("direct_id_detection_rates", {})
    report: List[Dict[str, Any]] = result.get("details", [])
    reid_risk: Optional[Dict[str, Any]] = result.get("reid_risk")

    # --- En-tête ---
    if meta:
        st.title(meta.get("title", "RAT-Bench — Résultats"))
        subtitle = meta.get("subtitle")
        if subtitle:
            st.caption(subtitle)
    else:
        lang = summary.get("language", "")
        lvl = summary.get("level")
        lvl_str = f"Level {lvl}" if lvl else "Tous niveaux"
        st.title("RAT-Bench — Résultats d'évaluation")
        st.caption(
            f"{lang.capitalize()} | {lvl_str} | {summary.get('n_documents', 0)} documents"
        )

    # --- Badge LLM actif ---
    cfg_run = summary.get("config", {})
    llm_active = any([
        cfg_run.get("llm_detection", False),
        cfg_run.get("llm_audit", False),
        cfg_run.get("llm_paraphrase", False),
    ])
    if llm_active:
        rupta_on = cfg_run.get("rupta_enabled", False)
        parts = []
        if cfg_run.get("llm_detection"):
            parts.append("🔎 Détection")
        if cfg_run.get("llm_audit"):
            parts.append("⚖️ Audit")
        if cfg_run.get("llm_paraphrase"):
            parts.append("✍️ Paraphrase")
        rupta_label = f" | ♻️ RUPTA max={cfg_run.get('rupta_max_iterations',3)} thr={cfg_run.get('rupta_p_threshold',15)}" if rupta_on else ""
        st.info(f"🤖 LLM actif : {' · '.join(parts)}{rupta_label}", icon="🤖")
    else:
        st.warning("⚠️ LLM désactivé pour cette évaluation", icon="⚠️")

    st.caption(
        f"Profil: {cfg_run.get('eval_profile') or cfg_run.get('profile') or 'n/a'} | "
        f"Mode évaluation: {cfg_run.get('eval_mode', 'n/a')} | "
        f"Mode masquage: {cfg_run.get('masking_mode', 'n/a')}"
    )

    # --- Métriques globales ---
    _render_summary_metrics(summary)
    _render_dual_metrics(report)

    st.markdown("---")

    # --- Onglets ---
    tab_labels = ["📊 Par difficulté", "🏥 Par scénario", "🔍 Identifiants directs", "🤖 RUPTA / LLM", "🛡️ Risque ré-identification", "📄 Documents"]
    tab_diff, tab_scen, tab_ids, tab_llm, tab_risk, tab_docs = st.tabs(tab_labels)

    with tab_diff:
        _render_by_difficulty(by_difficulty)

    with tab_scen:
        _render_by_scenario(by_scenario)

    with tab_ids:
        _render_direct_id_rates(direct_id_rates)

    with tab_llm:
        _render_rupta_llm(summary, report)

    with tab_risk:
        _render_reid_risk(reid_risk)

    with tab_docs:
        _render_documents(report)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _render_summary_metrics(summary: Dict[str, Any]) -> None:
    st.subheader("Métriques globales")

    st.caption(
        "Les métriques ci-dessous évaluent la **détection de spans PII** (Precision/Recall) "
        "par rapport aux annotations GT de RAT-Bench. "
        "Un bon pipeline doit maximiser le Rappel (ne rien laisser passer) "
        "tout en maintenant une Précision acceptable."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(
        "Précision macro",
        f"{summary.get('macro_precision', 0):.2%}",
        help=(
            "Parmi les spans détectés, quelle fraction correspond à un vrai PII du GT. "
            "Une précision faible (~10%) est normale : le pipeline sur-détecte pour ne rien rater. "
            "Objectif : > 20% est très bien sur RAT-Bench."
        ),
    )
    c2.metric(
        "Rappel macro",
        f"{summary.get('macro_recall', 0):.2%}",
        help=(
            "Parmi tous les PII du GT, quelle fraction a été couverte par une prédiction. "
            "C'est la métrique principale de sécurité. "
            "Objectif : > 90% est acceptable, > 95% est bon."
        ),
    )
    c3.metric(
        "F2 macro",
        f"{summary.get('macro_f2', 0):.2%}",
        help=(
            "Score F-bêta avec β=2 : pénalise 4× plus les faux négatifs (fuites) que les faux positifs. "
            "Conçu pour les systèmes de confidentialité où rater un PII est pire que sur-détecter. "
            "Objectif : > 60% est bon."
        ),
    )
    c4.metric(
        "Fuites totales",
        str(summary.get("total_leaks", 0)),
        help="Nombre de spans GT non couverts par aucune prédiction (faux négatifs cumulés sur tous les docs).",
    )
    c5.metric("Documents", str(summary.get("n_documents", 0)))

    # Ligne text-leak (optionnelle)
    if "avg_leak_rate" in summary:
        st.markdown(
            "**Analyse text-leak** — taux de fuite de PII dans le texte après anonymisation"
        )
        l1, l2, l3 = st.columns(3)
        l1.metric(
            "Fuite globale",
            f"{summary.get('avg_leak_rate', 0):.1%}",
            help=(
                "Part des valeurs PII du profil encore présentes verbatim dans le texte anonymisé. "
                "Objectif : < 5%. Une valeur élevée indique que l'anonymisation est insuffisante."
            ),
        )
        l2.metric(
            "Identifiants directs",
            f"{summary.get('avg_direct_leak_rate', 0):.1%}",
            help="Taux de fuite pour les identifiants directs (nom, prénom, email…). Doit être proche de 0%.",
        )
        l3.metric(
            "Identifiants indirects",
            f"{summary.get('avg_indirect_leak_rate', 0):.1%}",
            help=(
                "Taux de fuite pour les quasi-identifiants (âge, ville, profession…). "
                "Plus tolérant, mais des valeurs > 20% indiquent un risque de ré-identification."
            ),
        )

    # Ligne LLM / RUPTA (optionnelle)
    avg_ps = summary.get("avg_privacy_score")
    docs_rupta = summary.get("docs_with_rupta")
    if avg_ps is not None or docs_rupta is not None:
        st.markdown("**Boucle RUPTA / LLM Audit**")
        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Score vie privée moyen",
            f"{avg_ps:.0f}/100" if avg_ps is not None else "—",
            help="0=anonyme, 100=identifiable (après boucle RUPTA)",
        )
        r2.metric(
            "Docs avec RUPTA activé",
            str(docs_rupta) if docs_rupta is not None else "—",
            help="Docs où la boucle paraphrase a itéré au moins une fois",
        )
        r3.metric(
            "Entités LLM ajoutées",
            str(summary.get("llm_entities_total", 0)),
            help="Quasi-identifiants détectés uniquement par le LLM Detection",
        )


def _avg_nested_metric(report: List[Dict[str, Any]], key: str, metric: str) -> float:
    vals = [float(doc.get(key, {}).get(metric, 0.0)) for doc in report if isinstance(doc.get(key), dict)]
    return sum(vals) / len(vals) if vals else 0.0


def _render_dual_metrics(report: List[Dict[str, Any]]) -> None:
    if not report or not any("canonical_metrics" in doc or "benchmark_metrics" in doc for doc in report):
        return
    st.markdown("**Métriques canonique vs benchmark**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Canonique P", f"{_avg_nested_metric(report, 'canonical_metrics', 'precision'):.2%}")
    c2.metric("Canonique R", f"{_avg_nested_metric(report, 'canonical_metrics', 'recall'):.2%}")
    c3.metric("Canonique F2", f"{_avg_nested_metric(report, 'canonical_metrics', 'f2'):.2%}")
    c4.metric("Benchmark P", f"{_avg_nested_metric(report, 'benchmark_metrics', 'precision'):.2%}")
    c5.metric("Benchmark R", f"{_avg_nested_metric(report, 'benchmark_metrics', 'recall'):.2%}")
    c6.metric("Benchmark F2", f"{_avg_nested_metric(report, 'benchmark_metrics', 'f2'):.2%}")


def _metric_table(rows: List[Dict[str, Any]], pct_cols: List[str]) -> None:
    """Affiche un DataFrame formaté avec pandas (ou fallback texte)."""
    try:
        import pandas as pd

        df = pd.DataFrame(rows)
        if df.empty:
            st.info("Aucune donnée.")
            return

        first_col = df.columns[0]
        df = df.set_index(first_col)

        fmt = {c: "{:.2%}" for c in pct_cols if c in df.columns}
        st.dataframe(df.style.format(fmt), use_container_width=True)

        chart_cols = [c for c in pct_cols if c in df.columns]
        if chart_cols:
            st.bar_chart(df[chart_cols])
    except ImportError:
        for row in rows:
            st.markdown("  ".join(f"**{k}**: {v}" for k, v in row.items()))


def _render_by_difficulty(by_difficulty: Dict[str, Any]) -> None:
    st.subheader("Performances par niveau de difficulté")
    st.caption(
        "RAT-Bench définit 3 niveaux d'obfuscation des PII dans le texte source. "
        "Un pipeline robuste maintiendra un rappel élevé sur tous les niveaux."
    )
    if not by_difficulty:
        st.info("Aucune donnée par difficulté.")
        return

    rows = []
    for lvl, metrics in sorted(by_difficulty.items(), key=lambda x: int(x[0])):
        rows.append(
            {
                "Niveau": f"Level {lvl}",
                "Précision": round(metrics.get("macro_precision", 0), 4),
                "Rappel": round(metrics.get("macro_recall", 0), 4),
                "F2": round(metrics.get("macro_f2", 0), 4),
                "Docs": int(metrics.get("n_documents", 0)),
                "Fuites": int(metrics.get("total_leaks", 0)),
            }
        )
    _metric_table(rows, pct_cols=["Précision", "Rappel", "F2"])

    with st.expander("💡 À propos des niveaux"):
        st.markdown(
            """
- **Level 1** : PII directement présents dans le texte → bonne couverture attendue
- **Level 2** : PII partiellement obfusqués → détection plus difficile
- **Level 3** : PII fortement camuflés → épreuve de résistance maximale
"""
        )


def _render_by_scenario(by_scenario: Dict[str, Any]) -> None:
    st.subheader("Performances par scénario")
    st.caption(
        "RAT-Bench regroupe les profils par contexte (médical, légal, réseaux sociaux…). "
        "Des disparités importantes entre scénarios révèlent des angles morts du pipeline."
    )
    if not by_scenario:
        st.info("Aucune donnée par scénario.")
        return

    rows = []
    for scenario, metrics in sorted(by_scenario.items()):
        rows.append(
            {
                "Scénario": scenario,
                "Précision": round(metrics.get("macro_precision", 0), 4),
                "Rappel": round(metrics.get("macro_recall", 0), 4),
                "F2": round(metrics.get("macro_f2", 0), 4),
                "Docs": int(metrics.get("n_documents", 0)),
                "Fuites": int(metrics.get("total_leaks", 0)),
            }
        )
    _metric_table(rows, pct_cols=["Précision", "Rappel", "F2"])


def _render_direct_id_rates(direct_id_rates: Dict[str, Any]) -> None:
    st.subheader("Taux de détection — identifiants directs")
    st.caption(
        "Taux de couverture par type de PII direct (NOM, PRÉNOM, EMAIL, TÉLÉPHONE…). "
        "Disponible uniquement sur le Level 1 où les PII apparaissent explicitement. "
        "Un taux < 80% sur un type signale un manque dans les patterns regex ou le NER."
    )
    if not direct_id_rates:
        st.info("Aucune donnée disponible (les niveaux 2 et 3 n'exposent pas de spans GT directs).")
        return

    rows = []
    for id_type, stats in sorted(direct_id_rates.items()):
        rows.append(
            {
                "Type": id_type,
                "Détecté": int(stats.get("detected", 0)),
                "Total": int(stats.get("total", 0)),
                "Taux": round(stats.get("detection_rate", 0), 4),
            }
        )
    _metric_table(rows, pct_cols=["Taux"])


def _render_rupta_llm(summary: Dict[str, Any], report: List[Dict[str, Any]]) -> None:
    """Onglet RUPTA / LLM : distribution des scores de vie privée et itérations."""
    st.subheader("🤖 Analyse RUPTA / LLM")

    avg_ps = summary.get("avg_privacy_score")
    docs_rupta = summary.get("docs_with_rupta", 0)
    avg_iter = summary.get("avg_rupta_iterations", 0)
    llm_ents = summary.get("llm_entities_total", 0)

    if avg_ps is None:
        st.info(
            "⚠️ Les métriques LLM ne sont pas disponibles pour ce run — évaluez avec LLM activé."
        )
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Score vie privée moyen",
        f"{avg_ps:.0f}/100",
        help="0=anonyme, 100=identifiable. Calculé par le LLM Audit après anonymisation.",
    )
    c2.metric(
        "Docs avec RUPTA",
        str(docs_rupta),
        help="Documents où la paraphrase adversariale a itéré au moins une fois.",
    )
    c3.metric(
        "Itérations RUPTA moy.",
        f"{avg_iter:.2f}",
    )
    c4.metric(
        "Entités LLM ajoutées",
        str(llm_ents),
        help="Quasi-identifiants supplémentaires trouvés par LLM Détection.",
    )

    # Distribution des privacy_scores
    ps_values = [
        d.get("privacy_score") for d in report if d.get("privacy_score") is not None
    ]
    if ps_values:
        st.markdown("**Distribution du privacy_score (0=anonyme, 100=identifiable)**")
        try:
            import pandas as pd

            df_ps = pd.DataFrame({"privacy_score": ps_values})
            bins = list(range(0, 105, 10))
            df_ps["tranche"] = pd.cut(
                df_ps["privacy_score"], bins=bins, right=False,
                labels=[f"{i}-{i+9}" for i in range(0, 100, 10)]
            )
            hist = df_ps["tranche"].value_counts().sort_index()
            st.bar_chart(hist)
        except ImportError:
            st.write(ps_values)

    # Distribution des itérations RUPTA
    iter_values = [d.get("rupta_iterations", 0) for d in report]
    if any(i > 0 for i in iter_values):
        st.markdown("**Distribution des itérations RUPTA par document**")
        try:
            import pandas as pd

            iter_counts = pd.Series(iter_values).value_counts().sort_index()
            iter_counts.index = [f"{i} itér." for i in iter_counts.index]
            st.bar_chart(iter_counts)
        except ImportError:
            pass

    # Tableau détaillé des documents où LLM a ajouté des entités
    llm_docs = [
        {
            "Doc ID": d.get("doc_id", ""),
            "Privacy score": d.get("privacy_score", "—"),
            "RUPTA iter.": d.get("rupta_iterations", 0),
            "Entités LLM": d.get("llm_entities", 0),
            "Assessment": (d.get("llm_feedback") or {}).get("assessment", "")[:120],
        }
        for d in report
        if d.get("llm_entities", 0) > 0 or d.get("rupta_iterations", 0) > 0
    ]
    if llm_docs:
        st.markdown(f"**Documents avec activité LLM ({len(llm_docs)})**")
        try:
            import pandas as pd

            st.dataframe(pd.DataFrame(llm_docs), use_container_width=True)
        except ImportError:
            for row in llm_docs:
                st.json(row)


def _render_reid_risk(reid_risk: Optional[Dict[str, Any]]) -> None:
    """Onglet ré-identification risk (RAT-Bench Alg. 1-2 + LLM attacker)."""
    st.subheader("🛡️ Risque de ré-identification")

    if not reid_risk:
        st.info(
            "⚠️ L'évaluation du risque de ré-identification n'a pas été lancée.\n\n"
            "Activez l'option **« Évaluation du risque (LLM attacker) »** dans la barre latérale "
            "et relancez le benchmark. Nécessite une clé `OPENROUTER_API_KEY` dans le fichier `.env`."
        )
        return

    st.caption(
        "Évaluation basée sur les algorithmes RAT-Bench (Alg. 1 & 2) : un LLM attaquant tente de "
        "ré-identifier chaque profil à partir du texte anonymisé. Le risque R(x) = 1/k est dérivé "
        "du modèle Rocher — plus k est grand (peu d'individus identiques dans la population), plus le risque est élevé."
    )

    avg_risk = reid_risk.get("avg_risk", 0)
    frac_high = reid_risk.get("frac_high_risk_geq_0_09", 0)
    frac_perfect = reid_risk.get("frac_perfect_reid", 0)
    avg_attrs = reid_risk.get("avg_correct_attrs", 0)

    # Bandeaux d'interprétation
    if avg_risk < 0.05:
        st.success(f"✅ Risque moyen **{avg_risk:.2%}** — anonymisation efficace (< 5%)")
    elif avg_risk < 0.15:
        st.warning(f"⚠️ Risque moyen **{avg_risk:.2%}** — risque modéré (5–15% : à surveiller)")
    else:
        st.error(f"🚨 Risque moyen **{avg_risk:.2%}** — risque élevé (> 15% : anonymisation insuffisante)")

    # Métriques clés
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Risque moyen",
        f"{avg_risk:.2%}",
        help=(
            "Moyenne R(x) = 1/k (Rocher-style) sur tous les profils. "
            "Interprétation : < 5% = bien anonymisé · 5–15% = risque modéré · > 15% = risque élevé (seuil RGPD)."
        ),
    )
    c2.metric(
        "% Haut risque (≥9%)",
        f"{frac_high:.1%}",
        help=(
            "Fraction de profils avec risque ≥ 0.09. "
            "Le seuil 1/11 ≈ 9% correspond à k ≤ 11 individus identiques — seuil typique RGPD/CNIL. "
            "Objectif : < 10% de profils haut risque."
        ),
    )
    c3.metric(
        "% Ré-identification parfaite",
        f"{frac_perfect:.1%}",
        help=(
            "Fraction de profils où k=1 : l'individu est unique dans la population simulée. "
            "Même un seul attribut résiduel peut suffire. Objectif : 0%."
        ),
    )
    c4.metric(
        "Attributs corrects (moy.)",
        f"{avg_attrs:.1f} / 9",
        help=(
            "Nombre moyen d'attributs quasi-identifiants (âge, sexe, ville, profession…) "
            "correctement inférés par le LLM attaquant à partir du texte anonymisé. "
            "< 3/9 = bonne protection · ≥ 6/9 = risque critique."
        ),
    )

    st.markdown("---")

    # Par difficulté
    by_diff = reid_risk.get("by_difficulty", {})
    if by_diff:
        st.markdown("**Risque moyen par niveau de difficulté**")
        try:
            import pandas as pd

            rows = []
            for lvl, stats in sorted(by_diff.items(), key=lambda x: str(x[0])):
                mean_val = stats.get("mean", 0) if isinstance(stats, dict) else 0
                count_val = stats.get("count", 0) if isinstance(stats, dict) else 0
                rows.append({
                    "Niveau": f"Level {lvl}",
                    "Risque moyen": round(mean_val, 4),
                    "Profils": int(count_val),
                })
            df = pd.DataFrame(rows).set_index("Niveau")
            st.dataframe(df.style.format({"Risque moyen": "{:.2%}"}), use_container_width=True)
            st.bar_chart(df["Risque moyen"])
        except ImportError:
            st.json(by_diff)

    # Par scénario
    by_scen = reid_risk.get("by_scenario", {})
    if by_scen:
        st.markdown("**Risque moyen par scénario**")
        try:
            import pandas as pd

            rows = []
            for scenario, stats in sorted(by_scen.items()):
                mean_val = stats.get("mean", 0) if isinstance(stats, dict) else 0
                count_val = stats.get("count", 0) if isinstance(stats, dict) else 0
                rows.append({
                    "Scénario": scenario,
                    "Risque moyen": round(mean_val, 4),
                    "Profils": int(count_val),
                })
            df = pd.DataFrame(rows).set_index("Scénario")
            st.dataframe(df.style.format({"Risque moyen": "{:.2%}"}), use_container_width=True)
            st.bar_chart(df["Risque moyen"])
        except ImportError:
            st.json(by_scen)

    with st.expander(" Méthodologie", expanded=False):
        st.markdown(
            """
**Algorithme RAT-Bench (Alg. 1-2)** :

1. Le texte du profil est anonymisé par le pipeline PipeGraph
2. Un **LLM attacker** (OpenRouter / Gemini Flash) tente d'inférer les attributs indirects
   (`citoyenneté`, `date de naissance`, `état de résidence`, `genre`, `race`, `statut marital`,
    `éducation`, `emploi`, `occupation`)
3. Les attributs correctement inférés sont utilisés pour filtrer une **population de référence**
4. Le risque est $R(x) = 1/k$ où $k$ = taille de la classe d'équivalence restante

**Interprétation** :
- $R(x) = 0$ → aucun attribut correctement inféré → anonyme
- $R(x) \\geq 0.09$ → risque élevé (seuil RGPD)
- $R(x) = 1.0$ → profil unique → ré-identification certaine
"""
        )


def _render_documents(report: List[Dict[str, Any]]) -> None:
    st.subheader("Analyse par document")
    if not report:
        st.info("Aucun document.")
        return

    filters = render_doc_filters()
    selection = render_doc_table(report, filters)

    if selection.selected_doc_id:
        doc = next(
            (d for d in report if str(d.get("doc_id")) == str(selection.selected_doc_id)),
            None,
        )
        if doc:
            st.subheader("Document inspector")

            # Métadonnées RAT-Bench enrichies
            ratbench_meta = doc.get("ratbench_metadata")
            if ratbench_meta:
                with st.expander("🏷️ Métadonnées RAT-Bench", expanded=False):
                    col_a, col_b = st.columns(2)
                    col_a.write(f"**Scénario** : {ratbench_meta.get('scenario', '—')}")
                    col_a.write(f"**Difficulté** : Level {ratbench_meta.get('difficulty', '—')}")
                    col_b.write(
                        "**Types directs** : "
                        + ", ".join(ratbench_meta.get("direct_id_types", []))
                    )
                    col_b.write(
                        "**Types indirects** : "
                        + ", ".join(ratbench_meta.get("indirect_id_types", []))
                    )

            # Text-leak analysis pour ce document
            tla = doc.get("text_leak_analysis")
            if tla and isinstance(tla, dict):
                with st.expander("🛡️ Text-leak analysis", expanded=False):
                    t1, t2, t3 = st.columns(3)
                    t1.metric("Fuite globale", f"{tla.get('leak_rate', 0):.1%}")
                    t2.metric("Directs", f"{tla.get('direct_leak_rate', 0):.1%}")
                    t3.metric("Indirects", f"{tla.get('indirect_leak_rate', 0):.1%}")

            # LLM / RUPTA metrics pour ce document
            ps = doc.get("privacy_score")
            rupta_iter = doc.get("rupta_iterations", 0)
            llm_ents = doc.get("llm_entities", 0)
            llm_fb = doc.get("llm_feedback") or {}
            if ps is not None or rupta_iter > 0 or llm_ents > 0:
                with st.expander("🤖 LLM / RUPTA", expanded=rupta_iter > 0):
                    la1, la2, la3 = st.columns(3)
                    la1.metric(
                        "Privacy score",
                        f"{ps}/100" if ps is not None else "—",
                        help="0=anonyme, 100=identifiable",
                    )
                    la2.metric("Itérations RUPTA", str(rupta_iter))
                    la3.metric("Entités LLM", str(llm_ents))

                    if llm_fb.get("assessment"):
                        st.info(f"💬 Audit : {llm_fb['assessment']}")

                    leaked = llm_fb.get("leaked_attributes", [])
                    if leaked:
                        st.markdown("**Attributs résiduels identifiés par l'audit :**")
                        try:
                            import pandas as pd

                            rows_llm = [
                                {
                                    "Attribut": a.get("attribute", "?"),
                                    "Evidence": a.get("evidence", "")[:80],
                                    "Confiance": f"{float(a.get('confidence', 0)):.0%}",
                                    "Suggestion": a.get("suggestion", "")[:80],
                                }
                                for a in leaked
                            ]
                            st.dataframe(pd.DataFrame(rows_llm), use_container_width=True)
                        except ImportError:
                            for a in leaked:
                                st.write(a)

            # Risque de ré-identification pour ce document
            risk_assessment = doc.get("reid_risk_assessment")
            if risk_assessment:
                with st.expander("🛡️ Analyse du risque de ré-identification (Attaquant LLM)", expanded=True):
                    risk_val = risk_assessment.get("risk", 0)
                    k_val = risk_assessment.get("k", 0)
                    correct_count = risk_assessment.get("correct_attrs", 0)
                    inferred = risk_assessment.get("inferred_attrs", {})
                    
                    rk1, rk2, rk3 = st.columns(3)
                    
                    # Couleur métrique selon risque
                    if risk_val >= 0.09:
                        rk1.metric("Risque R(x)", f"{risk_val:.2%}", delta="ÉLEVÉ", delta_color="inverse")
                    else:
                        rk1.metric("Risque R(x)", f"{risk_val:.2%}", delta="BAS", delta_color="normal")
                        
                    rk2.metric("Indice K", f"{k_val}", help="Nombre d'individus avec ces attributs dans la population")
                    rk3.metric("Attributs trouvés", f"{correct_count} / 9")

                    correct_names = risk_assessment.get("correct_attr_names", [])
                    if correct_names:
                        st.markdown(
                            "**✅ Attributs correctement ré-identifiés :** "
                            + ", ".join(str(name) for name in correct_names)
                        )
                    
                    if inferred:
                        st.markdown("**🔍 Attributs inférés par l'attaquant :**")
                        # Tableau des attributs inférés vs réels
                        try:
                            import pandas as pd
                            rmeta = doc.get("ratbench_metadata", {})
                            profile = rmeta.get("profile", {})
                            
                            rows = []
                            for attr, val in inferred.items():
                                true_val = profile.get(attr, "—")
                                # Comparaison best-effort
                                is_correct = str(val).lower().strip() == str(true_val).lower().strip()
                                rows.append({
                                    "Attribut": attr,
                                    "Valeur inférée": val,
                                    "Valeur réelle (GT)": true_val,
                                    "Correct ?": "✅" if is_correct else "❌"
                                })
                            st.table(pd.DataFrame(rows))
                        except Exception:
                            st.json(inferred)
                    
                    st.caption("Le risque R(x) = 1/k. Un risque ≥ 9% (k ≤ 11) est considéré comme critique par la CNIL.")

            st.markdown("**Détails des détections**")
            render_doc_inspector(doc)
