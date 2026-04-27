from __future__ import annotations

import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

from ..core.models import AblationConfig


def _load_ablation_file(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError("Format de rapport invalide")
    return data


def _ablation_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for item in payload.get("results", []) or []:
        rows.append(
            {
                "config": item.get("name", ""),
                "description": item.get("description", ""),
                "precision": float(item.get("precision", 0.0) or 0.0),
                "recall": float(item.get("recall", 0.0) or 0.0),
                "f2": float(item.get("f2", 0.0) or 0.0),
                "leaks": int(item.get("leaks", 0) or 0),
                "elapsed_s": float(item.get("elapsed_s", 0.0) or 0.0),
            }
        )
    return rows


def _run_ablation_from_ui(cfg: AblationConfig) -> int:
    from eval import run_ablation

    args: List[str] = [
        "--suite",
        cfg.suite,
        "--dataset",
        cfg.dataset_kind,
        "--limit",
        str(cfg.limit),
    ]

    if cfg.dataset_kind == "ratbench":
        args.extend(["--language", "english", "--level", "1"])

    if cfg.save_run:
        args.append("--save-runs")

    return int(run_ablation.main(args))


def render_ablation_dashboard(*, cfg: AblationConfig, reports_dir: str) -> None:
    st.title("Études d'Ablation")
    st.caption("Analyse des contributions de chaque module du pipeline et comparaison des configurations.")

    if cfg.run_new:
        with st.expander("Lancer une nouvelle étude d'ablation", expanded=True):
            st.write(
                f"Suite: `{cfg.suite}` · Dataset: `{cfg.dataset_kind}` · Limit: `{cfg.limit}` · Save runs: `{cfg.save_run}`"
            )
            if st.button("▶ Lancer l'ablation", type="primary", key="ablation_launch"):
                with st.spinner("Exécution de l'ablation en cours..."):
                    code = _run_ablation_from_ui(cfg)
                if code == 0:
                    st.success("Ablation terminée.")
                else:
                    st.error(f"Ablation échouée (code {code}).")

    files = sorted(glob.glob(os.path.join(reports_dir, "ablation_*.json")))
    if not files:
        st.info("Aucun rapport d'ablation trouvé dans eval/evaluation/reports.")
        return

    selected = st.selectbox(
        "Rapport d'ablation",
        files,
        index=max(0, len(files) - 1),
        format_func=lambda p: os.path.basename(p),
    )

    try:
        payload = _load_ablation_file(selected)
    except Exception as exc:
        st.error(f"Impossible de charger le rapport: {exc}")
        return

    suite = payload.get("suite", "?")
    dataset = payload.get("dataset", "?")
    n_configs = payload.get("n_configs", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Suite", str(suite))
    c2.metric("Dataset", str(dataset))
    c3.metric("Configs", int(n_configs) if isinstance(n_configs, int) else 0)

    rows = _ablation_rows(payload)
    if not rows:
        st.warning("Le rapport ne contient pas de résultats exploitables.")
        return

    df = pd.DataFrame(rows)
    df_sorted_f2 = df.sort_values("f2", ascending=False)

    best = df_sorted_f2.iloc[0]
    st.success(
        f"Meilleure config F2: {best['config']} · F2={best['f2']:.3f} · Recall={best['recall']:.3f} · Precision={best['precision']:.3f}"
    )

    st.subheader("Comparatif des configurations")
    st.dataframe(df_sorted_f2, use_container_width=True, hide_index=True)

    st.subheader("Graphiques")
    col_l, col_r = st.columns(2)
    with col_l:
        st.caption("F2 par configuration")
        st.bar_chart(df_sorted_f2.set_index("config")["f2"])
    with col_r:
        st.caption("Fuites par configuration (plus bas = mieux)")
        st.bar_chart(df.sort_values("leaks").set_index("config")["leaks"])

    st.subheader("Détail des erreurs")
    regressions = df.sort_values(["leaks", "f2"], ascending=[False, True]).head(10)
    st.write("Configurations avec le plus de fuites :")
    st.dataframe(regressions[["config", "description", "leaks", "precision", "recall", "f2"]], use_container_width=True, hide_index=True)
