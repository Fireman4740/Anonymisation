from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from ..core.constants import (
    DATASET_KINDS,
    DEFAULT_SOURCE,
    DETECTION_MODES,
    SOURCE_LABELS,
    SOURCE_ORDER,
)
from ..core.models import LocalEvalConfig, ReportSelection, RunSummary, RunsFilter


def render_sidebar_ui(
    reports: List[str],
    runs: List[RunSummary],
    eval_dir: str,
    run_store: Any,
    *,
    reports_dir: str,
    runs_dir: str,
) -> Tuple[str, Any, Any]:
    """
    Renders the unified sidebar with tabs-like navigation.
    Returns:
        (active_mode, config_object_or_none, extra_data)

    modes: 'local_eval', 'history', 'comparison'
    """
    st.sidebar.title("Configuration")

    # Use radio buttons to simulate the "Tabs" selection because we need the state.
    # Standard st.tabs does not return the active tab, making it unsuitable for
    # controlling the main view mode (Dashboard vs Comparison) effectively
    # without complex workarounds.
    mode_map = {"Nouveau Run": "local_eval", "Historique": "history", "Comparaison": "comparison"}

    # Determine default index
    default_index = 1  # Default to History

    # Check if we have a stored preference or existing state
    # We prioritize the explicit session_state["source"] or "comparison_mode"
    if st.session_state.get("comparison_mode"):
        default_index = 2
    elif st.session_state.get("source") == "local_eval":
        default_index = 0
    elif st.session_state.get("source") == "history":
        default_index = 1

    # In case the current source is legacy/invalid
    if default_index == 1 and st.session_state.get("source") not in [
        "local_eval",
        "history",
        "comparison",
    ]:
        # fallback is history
        pass

    selected_tab = st.sidebar.radio(
        "Mode", list(mode_map.keys()), index=default_index, label_visibility="collapsed"
    )

    st.sidebar.markdown("---")

    active_mode = mode_map[selected_tab]

    # Update global state
    if active_mode == "comparison":
        st.session_state["comparison_mode"] = True
    else:
        st.session_state["comparison_mode"] = False
        st.session_state["source"] = active_mode

    config_obj = None
    extra_data = None

    if active_mode == "local_eval":
        config_obj = render_local_eval_controls(eval_dir=eval_dir)

    elif active_mode == "history":
        sub_mode, selection = render_history_controls(reports, runs)
        config_obj = selection
        extra_data = sub_mode

    elif active_mode == "comparison":
        render_comparison_controls_adapted(
            reports,
            runs,
            run_store,
            reports_dir=reports_dir,
            runs_dir=runs_dir,
        )

    return active_mode, config_obj, extra_data


def render_comparison_controls_adapted(
    reports: List[str],
    runs: List[RunSummary],
    run_store,
    *,
    reports_dir: str,
    runs_dir: str,
) -> None:
    st.sidebar.subheader("Configuration Comparaison")
    st.sidebar.info("Sélectionnez le rapport de référence (Modèle B).")

    comp_source = st.sidebar.radio(
        "Source (Ref B)", ["Rapport Existant", "Run Sauvegardé"], key="comp_source_radio"
    )

    from ..core.state import update_comparison_report
    from ..services.report_loader import load_report_from_file, load_report_from_run

    if comp_source == "Rapport Existant":
        if not reports:
            st.sidebar.warning("Aucun rapport disponible.")
        else:
            report_names = [os.path.basename(p) for p in reports]
            selected_comp = st.sidebar.selectbox("Fichier", report_names, key="comp_file_select")
            if selected_comp:
                path = reports[report_names.index(selected_comp)]
                if st.sidebar.button("Charger B (Fichier)", key="btn_load_b_file"):
                    try:
                        rep = load_report_from_file(path, base_dir=reports_dir)
                        meta = {"title": "Reference B", "subtitle": selected_comp, "source": "file"}
                        update_comparison_report(rep, meta)
                        st.sidebar.success("Chargé !")
                    except Exception as e:
                        st.sidebar.error(f"Erreur: {e}")

    elif comp_source == "Run Sauvegardé":
        if not runs:
            st.sidebar.warning("Aucun run disponible.")
        else:
            run_opts = [f"{r.created_dt} | {os.path.basename(str(r.path))}" for r in runs]
            selected_run_idx = st.sidebar.selectbox(
                "Run", range(len(runs)), format_func=lambda i: run_opts[i], key="comp_run_select"
            )

            if st.sidebar.button("Charger B (Run)", key="btn_load_b_run"):
                run_obj = runs[selected_run_idx]
                try:
                    meta_run, rep = load_report_from_run(run_obj.path, run_store, base_dir=runs_dir)
                    meta = {
                        "title": "Reference B",
                        "subtitle": f"{meta_run.get('created_at')} | {meta_run.get('dataset', {}).get('name')}",
                        "source": "run",
                    }
                    update_comparison_report(rep, meta)
                    st.sidebar.success("Chargé !")
                except Exception as e:
                    st.sidebar.error(f"Erreur: {e}")


def render_history_controls(reports: List[str], runs: List[RunSummary]) -> Tuple[str, Any]:
    st.sidebar.subheader("Runs sauvegardes")
    return "saved_runs", render_saved_runs_controls(runs)


def _list_json_datasets(data_dir: str) -> List[Tuple[str, str]]:
    files = sorted([p for p in os.listdir(data_dir) if p.endswith(".json")])
    return [(name, os.path.join(data_dir, name)) for name in files]


def _list_tab_splits(tab_dir: str) -> List[Tuple[str, str]]:
    files = sorted([p for p in os.listdir(tab_dir) if p.endswith(".jsonl")])
    out = [(name.replace(".jsonl", ""), os.path.join(tab_dir, name)) for name in files]
    preferred = ["test", "dev", "train"]
    out.sort(key=lambda x: (preferred.index(x[0]) if x[0] in preferred else 999, x[0]))
    return out


def render_local_eval_controls(*, eval_dir: str) -> Optional[LocalEvalConfig]:
    st.sidebar.subheader("PipeGraph (local)")

    dataset_kind = st.sidebar.selectbox("Dataset", DATASET_KINDS, index=0)
    run_full_dataset = st.sidebar.checkbox("Run sur le dataset entier", value=False)

    split: Optional[str] = None
    dataset_label = ""
    in_path = ""

    if dataset_kind == "TAB":
        tab_dir = os.path.join(eval_dir, "datasets", "TAB")
        tab_files = _list_tab_splits(tab_dir) if os.path.isdir(tab_dir) else []
        if not tab_files:
            st.sidebar.error(f"Aucun dataset TAB trouve dans {tab_dir}")
            return None
        split, in_path = tab_files[0]
        selected = st.sidebar.selectbox("Split TAB", tab_files, index=0, format_func=lambda x: x[0])
        split, in_path = selected
        dataset_label = f"TAB/{split}"
    elif dataset_kind == "DB-bio":
        db_dir = os.path.join(eval_dir, "datasets", "DB-bio")
        db_files = (
            sorted([os.path.join(db_dir, p) for p in os.listdir(db_dir) if p.endswith(".jsonl")])
            if os.path.isdir(db_dir)
            else []
        )
        if not db_files:
            st.sidebar.error(f"Aucun dataset DB-bio trouve dans {db_dir}")
            return None
        in_path = st.sidebar.selectbox(
            "Fichier DB-bio", db_files, index=0, format_func=os.path.basename
        )
        split = os.path.basename(in_path).replace(".jsonl", "")
        dataset_label = f"DB-bio/{split}"
    else:
        data_dir = os.path.join(eval_dir, "datasets", "data")
        json_datasets = _list_json_datasets(data_dir) if os.path.isdir(data_dir) else []
        if not json_datasets:
            st.sidebar.error(f"Aucun dataset JSON trouve dans {data_dir}")
            return None
        selected = st.sidebar.selectbox(
            "Dataset JSON", json_datasets, index=0, format_func=lambda x: x[0]
        )
        dataset_file, in_path = selected
        dataset_label = f"data/{dataset_file}"

    limit_default = 200
    limit = int(
        st.sidebar.number_input(
            "Limit",
            min_value=1,
            max_value=5000,
            value=limit_default,
            step=10,
            disabled=run_full_dataset,
        )
    )
    if run_full_dataset:
        st.sidebar.caption("Limit ignore (dataset entier).")

    enable_detection = st.sidebar.checkbox("enable_detection", value=True)
    enable_deterministic = st.sidebar.checkbox("enable_deterministic", value=True)
    enable_ai = st.sidebar.checkbox("enable_ai", value=True)
    enable_anonymization = st.sidebar.checkbox("enable_anonymization", value=True)
    detection_mode = st.sidebar.selectbox("detection_mode", DETECTION_MODES, index=0)

    st.sidebar.markdown("---")
    save_run = st.sidebar.checkbox("Sauvegarder un run", value=False)
    run_name = st.sidebar.text_input("run_name (optionnel)", value="") if save_run else ""

    return LocalEvalConfig(
        dataset_kind=dataset_kind,
        dataset_label=dataset_label,
        dataset_path=in_path,
        split=split,
        run_full_dataset=run_full_dataset,
        limit=limit,
        enable_detection=enable_detection,
        enable_deterministic=enable_deterministic,
        enable_ai=enable_ai,
        enable_anonymization=enable_anonymization,
        detection_mode=detection_mode,
        save_run=save_run,
        run_name=run_name,
    )


def render_saved_runs_controls(runs: List[RunSummary]) -> RunsFilter:
    st.sidebar.subheader("Runs sauvegardes")
    if not runs:
        st.sidebar.info("Aucun run sauvegarde.")
        return RunsFilter(
            run_paths=[],
            selected_path=None,
            start_date=None,
            end_date=None,
            dataset_filter="",
            config_contains="",
            config_key="",
            config_value="",
        )

    min_dt = min((r.created_dt for r in runs if r.created_dt is not None), default=None)
    max_dt = max((r.created_dt for r in runs if r.created_dt is not None), default=None)

    if min_dt and max_dt:
        start_date = st.sidebar.date_input("Du", value=min_dt.date())
        end_date = st.sidebar.date_input("Au", value=max_dt.date())
    else:
        start_date = st.sidebar.date_input("Du")
        end_date = st.sidebar.date_input("Au")

    if start_date and end_date and start_date > end_date:
        start_date, end_date = end_date, start_date

    dataset_filter = st.sidebar.text_input("Dataset contient", value="")
    config_contains = st.sidebar.text_input("Config contient", value="")
    config_key = st.sidebar.text_input("Config key", value="")
    config_value = st.sidebar.text_input("Config value (texte)", value="")

    filtered = []
    for r in runs:
        dt = r.created_dt
        if dt is not None and start_date and end_date:
            d = dt.date()
            if d < start_date or d > end_date:
                continue
        if dataset_filter and dataset_filter.lower() not in str(r.dataset or "").lower():
            continue
        if config_contains:
            cfg = r.config
            cfg_s = str(cfg)
            if config_contains.lower() not in cfg_s.lower():
                continue
        if config_key:
            if not isinstance(r.config, dict):
                continue
            if config_key not in r.config:
                continue
            if config_value and str(r.config.get(config_key)) != str(config_value):
                continue
        filtered.append(r)

    run_paths = [r.path for r in filtered]
    if not run_paths:
        st.sidebar.warning("Aucun run ne correspond aux filtres.")
        return RunsFilter(
            run_paths=[],
            selected_path=None,
            start_date=start_date,
            end_date=end_date,
            dataset_filter=dataset_filter,
            config_contains=config_contains,
            config_key=config_key,
            config_value=config_value,
        )

    selected_path = st.sidebar.selectbox(
        "Selectionner un run", run_paths, format_func=lambda p: os.path.basename(str(p))
    )
    return RunsFilter(
        run_paths=run_paths,
        selected_path=selected_path,
        start_date=start_date,
        end_date=end_date,
        dataset_filter=dataset_filter,
        config_contains=config_contains,
        config_key=config_key,
        config_value=config_value,
    )


def render_existing_report_controls(reports: List[str]) -> ReportSelection:
    st.sidebar.subheader("Rapports")
    if not reports:
        st.sidebar.info("Aucun report disponible.")
        return ReportSelection(report_paths=[], selected_path=None)
    report_names = [os.path.basename(p) for p in reports]
    selected = st.sidebar.selectbox("Selectionner un report", report_names)
    selected_path = reports[report_names.index(selected)]
    return ReportSelection(report_paths=reports, selected_path=selected_path)


def render_comparison_controls(
    reports: List[str],
    runs: List[RunSummary],
    run_store,
    *,
    reports_dir: str,
    runs_dir: str,
) -> None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comparaison")

    # Toggle Comparison Mode
    is_compare_mode = st.session_state.get("comparison_mode", False)
    new_mode = st.sidebar.checkbox("Mode Comparaison", value=is_compare_mode)

    if new_mode != is_compare_mode:
        st.session_state["comparison_mode"] = new_mode
        st.rerun()

    if not new_mode:
        return

    st.sidebar.info("Sélectionnez le rapport de référence (Modèle B).")

    # Source selector for Comparison Report
    # We allow "Saved Runs" or "Existing Report" (Local Files)
    # We do NOT implement Local Eval for comparison slot for simplicity yet.

    comp_source = st.sidebar.radio(
        "Source (Ref B)", ["Rapport Existant", "Run Sauvegardé"], key="comp_source_radio"
    )

    from ..core.state import update_comparison_report
    from ..services.report_loader import load_report_from_file, load_report_from_run

    if comp_source == "Rapport Existant":
        if not reports:
            st.sidebar.warning("Aucun rapport disponible.")
        else:
            report_names = [os.path.basename(p) for p in reports]
            selected_comp = st.sidebar.selectbox("Fichier", report_names, key="comp_file_select")
            if selected_comp:
                path = reports[report_names.index(selected_comp)]
                if st.sidebar.button("Charger B (Fichier)", key="btn_load_b_file"):
                    try:
                        rep = load_report_from_file(path, base_dir=reports_dir)
                        meta = {"title": "Reference B", "subtitle": selected_comp, "source": "file"}
                        update_comparison_report(rep, meta)
                        st.sidebar.success("Chargé !")
                    except Exception as e:
                        st.sidebar.error(f"Erreur: {e}")

    elif comp_source == "Run Sauvegardé":
        if not runs:
            st.sidebar.warning("Aucun run disponible.")
        else:
            # Simple list for now
            run_opts = [f"{r.created_dt} | {os.path.basename(str(r.path))}" for r in runs]
            selected_run_idx = st.sidebar.selectbox(
                "Run", range(len(runs)), format_func=lambda i: run_opts[i], key="comp_run_select"
            )

            if st.sidebar.button("Charger B (Run)", key="btn_load_b_run"):
                run_obj = runs[selected_run_idx]
                try:
                    meta_run, rep = load_report_from_run(run_obj.path, run_store, base_dir=runs_dir)
                    meta = {
                        "title": "Reference B",
                        "subtitle": f"{meta_run.get('created_at')} | {meta_run.get('dataset', {}).get('name')}",
                        "source": "run",
                    }
                    update_comparison_report(rep, meta)
                    st.sidebar.success("Chargé !")
                except Exception as e:
                    st.sidebar.error(f"Erreur: {e}")
