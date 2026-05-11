from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st

from eval.core.profiles import EVAL_MODE_CHOICES, MASKING_MODE_CHOICES, PROFILE_CHOICES

from ..core.constants import (
    DATASET_KINDS,
    DETECTION_MODES,
    RATBENCH_LANGUAGES,
    RATBENCH_LEVELS,
    SOURCE_LABELS,
    SOURCE_ORDER,
)
from ..core.models import BenchmarkEvalConfig, LocalEvalConfig, RATBenchEvalConfig, RunSummary, RunsFilter, AblationConfig

def render_sidebar_ui(
    reports: List[str],
    runs: List[RunSummary],
    eval_dir: str,
    run_store: Any,
    *,
    reports_dir: str,
    runs_dir: str,
) -> Tuple[str, Any, Any]:
    st.sidebar.title("Navigation")
    
    mode_map = {
        "📊 Lancer Benchmark": "benchmark",
        "🗂️ Historique & Comparaison": "history",
        "🧩 Études d'Ablation": "ablation",
    }

    labels = list(mode_map.keys())
    source_to_label = {
        "benchmark": labels[0],
        "history": labels[1],
        "ablation": labels[2],
    }
    nav_key = "nav_mode_choice"

    default_label = source_to_label.get(str(st.session_state.get("source", "benchmark")), labels[0])

    raw_value = st.session_state.get(nav_key)
    if raw_value is not None and raw_value not in labels:
        del st.session_state[nav_key]

    try:
        selected_tab = st.sidebar.selectbox(
            "Mode",
            options=labels,
            index=labels.index(default_label),
            label_visibility="collapsed",
            key=nav_key,
        )
    except TypeError:
        st.session_state.pop(nav_key, None)
        selected_tab = st.sidebar.selectbox(
            "Mode",
            options=labels,
            index=labels.index(default_label),
            label_visibility="collapsed",
            key=nav_key,
        )
    
    st.sidebar.markdown("---")
    active_mode = mode_map[selected_tab]
    
    # State update
    if active_mode == "history":
        st.session_state["comparison_mode"] = True
    else:
        st.session_state["comparison_mode"] = False
        
    st.session_state["source"] = active_mode
    
    config_obj = None
    extra_data = None
    
    if active_mode == "benchmark":
        config_obj = render_benchmark_controls(eval_dir=eval_dir)
    elif active_mode == "history":
        config_obj = render_history_controls(runs)
    elif active_mode == "ablation":
        config_obj = render_ablation_controls()
        
    return active_mode, config_obj, extra_data

def _list_json_datasets(data_dir: str) -> List[Tuple[str, str]]:
    files = sorted([p for p in os.listdir(data_dir) if p.endswith(".json")])
    return [(name, os.path.join(data_dir, name)) for name in files]

def _list_tab_splits(tab_dir: str) -> List[Tuple[str, str]]:
    files = sorted([p for p in os.listdir(tab_dir) if p.endswith(".jsonl")])
    out = [(name.replace(".jsonl", ""), os.path.join(tab_dir, name)) for name in files]
    preferred = ["test", "dev", "train"]
    out.sort(key=lambda x: (preferred.index(x[0]) if x[0] in preferred else 999, x[0]))
    return out


def _count_jsonl_rows(path: str) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


def _count_json_examples(path: str) -> int:
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    examples = payload.get("examples", []) if isinstance(payload, dict) else []
    return len(examples) if isinstance(examples, list) else 0


def _count_conll_docs(path: str) -> Optional[int]:
    if not os.path.exists(path):
        return None

    docs = 0
    in_doc = False
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if line and not line.startswith("-DOCSTART-"):
                in_doc = True
                continue
            if in_doc:
                docs += 1
                in_doc = False

    if in_doc:
        docs += 1
    return docs


def _count_ratbench_profiles(eval_dir: str, language: str, level: Optional[int]) -> Optional[int]:
    cache_path = os.path.join(eval_dir, "datasets", "RAT-Bench", "cache", f"ratbench_{language}.json")
    if not os.path.exists(cache_path):
        return None
    with open(cache_path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list):
        return None
    if level is None:
        return len(records)
    return sum(1 for rec in records if isinstance(rec, dict) and rec.get("difficulty") == level)

def render_benchmark_controls(*, eval_dir: str) -> Optional[BenchmarkEvalConfig]:
    st.sidebar.subheader("Configuration du Run")
    
    dataset_type = st.sidebar.selectbox(
        "Type de Benchmark/Dataset", ["RAT-Bench", *DATASET_KINDS], index=0
    )
    
    # Shared controls setup
    run_full_dataset = False
    limit = 200
    ratbench_conf = None
    local_conf = None
    split = None
    dataset_label = ""
    in_path = ""
    language = "english"
    level = None
    dataset_instances: Optional[int] = None
    dataset_profiles: Optional[int] = None
    full_run_key = f"full_run_{dataset_type.lower().replace('-', '_')}"
    
    if dataset_type == "RAT-Bench":
        language = st.sidebar.selectbox("Langue", RATBENCH_LANGUAGES, index=0)
        level_str = st.sidebar.selectbox("Niveau", RATBENCH_LEVELS, index=0)
        level = None if level_str == "Tous" else int(level_str)
        if st.sidebar.button("✅ Sélectionner tout le benchmark", key=f"btn_{full_run_key}"):
            st.session_state[full_run_key] = True
        run_full_dataset = st.sidebar.checkbox(
            "Tous les profils du niveau",
            value=bool(st.session_state.get(full_run_key, False)),
            key=full_run_key,
        )
        limit = int(st.sidebar.number_input("Limit (profils)", min_value=1, max_value=5000, value=50, step=10, disabled=run_full_dataset))
        dataset_profiles = _count_ratbench_profiles(eval_dir=eval_dir, language=language, level=level)
        dataset_instances = dataset_profiles
    else:
        if st.sidebar.button("✅ Sélectionner tout le dataset", key=f"btn_{full_run_key}"):
            st.session_state[full_run_key] = True
        run_full_dataset = st.sidebar.checkbox(
            "Run sur le dataset entier",
            value=bool(st.session_state.get(full_run_key, False)),
            key=full_run_key,
        )
        limit = int(st.sidebar.number_input("Limit", min_value=1, max_value=5000, value=200, step=10, disabled=run_full_dataset))
        
        if dataset_type == "TAB":
            tab_dir = os.path.join(eval_dir, "datasets", "TAB")
            tab_files = _list_tab_splits(tab_dir) if os.path.isdir(tab_dir) else []
            if not tab_files:
                st.sidebar.error(f"Aucun dataset TAB trouvé dans {tab_dir}")
                return None
            selected = st.sidebar.selectbox("Split TAB", tab_files, index=0, format_func=lambda x: x[0])
            split, in_path = selected
            dataset_label = f"TAB/{split}"
            dataset_instances = _count_jsonl_rows(in_path)
            dataset_profiles = dataset_instances
        elif dataset_type == "DB-bio":
            db_dir = os.path.join(eval_dir, "datasets", "DB-bio")
            db_files = sorted([os.path.join(db_dir, p) for p in os.listdir(db_dir) if p.endswith(".jsonl")]) if os.path.isdir(db_dir) else []
            if not db_files:
                st.sidebar.error(f"Aucun dataset DB-bio trouvé dans {db_dir}")
                return None
            in_path = st.sidebar.selectbox("Fichier DB-bio", db_files, index=0, format_func=os.path.basename)
            split = os.path.basename(in_path).replace(".jsonl", "")
            dataset_label = f"DB-bio/{split}"
            dataset_instances = _count_jsonl_rows(in_path)
            dataset_profiles = dataset_instances
        elif dataset_type == "cleanconll2003":
            split = st.sidebar.selectbox("Split CleanCoNLL", ["test", "dev", "train"], index=0)
            in_path = os.path.join(
                eval_dir,
                "datasets",
                "cleanconll_cache",
                "cleanconll",
                f"cleanconll.{split}",
            )
            dataset_label = f"cleanconll2003/{split}"
            dataset_instances = _count_conll_docs(in_path)
            dataset_profiles = dataset_instances
        else:
            data_dir = os.path.join(eval_dir, "datasets", "data")
            json_datasets = _list_json_datasets(data_dir) if os.path.isdir(data_dir) else []
            if not json_datasets:
                st.sidebar.error(f"Aucun dataset JSON trouvé dans {data_dir}")
                return None
            selected = st.sidebar.selectbox("Dataset JSON", json_datasets, index=0, format_func=lambda x: x[0])
            dataset_file, in_path = selected
            dataset_label = f"data/{dataset_file}"
            dataset_instances = _count_json_examples(in_path)
            dataset_profiles = dataset_instances

    st.sidebar.caption("Taille du dataset sélectionné")
    c_inst, c_prof = st.sidebar.columns(2)
    c_inst.metric("Instances", str(dataset_instances) if dataset_instances is not None else "?")
    c_prof.metric("Profils", str(dataset_profiles) if dataset_profiles is not None else "?")

    st.sidebar.markdown("---")
    st.sidebar.caption("Modules & Pipeline")
    profile = st.sidebar.selectbox("Profil dataset", PROFILE_CHOICES, index=0)
    eval_mode = st.sidebar.selectbox(
        "Mode évaluation",
        EVAL_MODE_CHOICES,
        index=EVAL_MODE_CHOICES.index("both"),
    )
    masking_mode = st.sidebar.selectbox(
        "Mode masquage",
        MASKING_MODE_CHOICES,
        index=MASKING_MODE_CHOICES.index("benchmark"),
    )
    enable_detection = st.sidebar.checkbox("Détection Regex/GLiNER", value=True)
    enable_deterministic = st.sidebar.checkbox("Déterministe (Regex)", value=True)
    enable_ai = st.sidebar.checkbox("AI (GLiNER)", value=True)
    enable_anonymization = st.sidebar.checkbox("Anonymisation", value=True)
    detection_mode = st.sidebar.selectbox("Mode de détection", DETECTION_MODES, index=0)
    
    st.sidebar.caption("🤖 Modules LLM")
    llm_detection_enabled = st.sidebar.checkbox("LLM Détection", value=True)
    llm_audit_enabled = st.sidebar.checkbox("LLM Audit", value=True)
    llm_paraphrase_enabled = st.sidebar.checkbox("LLM Paraphrase (RUPTA)", value=True)
    
    rupta_enabled = st.sidebar.checkbox("Activer RUPTA", value=True, disabled=not (llm_audit_enabled and llm_paraphrase_enabled))
    rupta_max_iterations = int(st.sidebar.number_input("Max itérations RUPTA", min_value=1, max_value=5, value=3, disabled=not rupta_enabled))
    rupta_p_threshold = int(st.sidebar.slider("Seuil privacy (p_threshold)", min_value=0, max_value=100, value=15, step=5, disabled=not rupta_enabled))

    # Risk re-identification (RAT-Bench only, requires OpenRouter)
    enable_risk_eval = False
    if dataset_type == "RAT-Bench":
        st.sidebar.caption("🛡️ Risque de ré-identification")
        enable_risk_eval = st.sidebar.checkbox(
            "Évaluation du risque (LLM attacker)",
            value=False,
            help="Utilise un LLM OpenRouter pour tenter de ré-identifier les profils anonymisés. Nécessite OPENROUTER_API_KEY dans .env.",
        )

    st.sidebar.markdown("---")
    save_run = st.sidebar.checkbox("Sauvegarder le run", value=False)
    run_name = st.sidebar.text_input("Nom du run (optionnel)", value="") if save_run else ""
    
    if dataset_type == "RAT-Bench":
        ratbench_conf = RATBenchEvalConfig(
            language=str(language),
            level=level,
            limit=limit,
            run_full_dataset=run_full_dataset,
            enable_detection=enable_detection,
            enable_deterministic=enable_deterministic,
            enable_ai=enable_ai,
            enable_anonymization=enable_anonymization,
            detection_mode=str(detection_mode),
            profile=str(profile),
            eval_mode=str(eval_mode),
            masking_mode=str(masking_mode),
            llm_detection_enabled=llm_detection_enabled,
            llm_audit_enabled=llm_audit_enabled,
            llm_paraphrase_enabled=llm_paraphrase_enabled,
            rupta_enabled=rupta_enabled and llm_audit_enabled and llm_paraphrase_enabled,
            rupta_max_iterations=rupta_max_iterations,
            rupta_p_threshold=rupta_p_threshold,
            enable_risk_eval=enable_risk_eval,
            save_run=save_run,
            run_name=run_name,
        )
    else:
        local_conf = LocalEvalConfig(
            dataset_kind=dataset_type,
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
            profile=str(profile),
            eval_mode=str(eval_mode),
            masking_mode=str(masking_mode),
            llm_detection_enabled=llm_detection_enabled,
            llm_audit_enabled=llm_audit_enabled,
            llm_paraphrase_enabled=llm_paraphrase_enabled,
            rupta_enabled=rupta_enabled and llm_audit_enabled and llm_paraphrase_enabled,
            rupta_max_iterations=rupta_max_iterations,
            rupta_p_threshold=rupta_p_threshold,
            save_run=save_run,
            run_name=run_name,
        )
        
    return BenchmarkEvalConfig(type=dataset_type, ratbench_config=ratbench_conf, local_config=local_conf)

def render_history_controls(runs: List[RunSummary]) -> RunsFilter:
    st.sidebar.subheader("Filtres d'historique")
    
    if not runs:
        return RunsFilter([], None, None, None, "", "", "", "")

    min_dt = min((r.created_dt for r in runs if r.created_dt is not None), default=None)
    max_dt = max((r.created_dt for r in runs if r.created_dt is not None), default=None)

    col1, col2 = st.sidebar.columns(2)
    if min_dt and max_dt:
        start_date = col1.date_input("Du", value=min_dt.date())
        end_date = col2.date_input("Au", value=max_dt.date())
    else:
        start_date = col1.date_input("Du")
        end_date = col2.date_input("Au")
    
    dataset_filter = st.sidebar.text_input("Dataset", value="", help="Ex: tab, ratbench, etc.")
    
    filtered_runs = runs
    if start_date and end_date:
        if start_date > end_date:
            start_date, end_date = end_date, start_date
        filtered_runs = [r for r in runs if r.created_dt 
                        and start_date <= r.created_dt.date() <= end_date]
        
    if dataset_filter:
        filtered_runs = [r for r in filtered_runs if dataset_filter.lower() in str(r.dataset or "").lower()]

    return RunsFilter(
        run_paths=[r.path for r in filtered_runs],
        selected_path=None,
        start_date=start_date,
        end_date=end_date,
        dataset_filter=dataset_filter,
        config_contains="",
        config_key="",
        config_value=""
    )

def render_ablation_controls() -> AblationConfig:
    st.sidebar.subheader("Options d'Ablation")
    st.sidebar.info("Ceci permet de relancer / valider chaque module (GLiNER, LLM, RUPTA...)")
    
    run_new = st.sidebar.checkbox("Lancer une nouvelle étude", value=False)
    
    dataset_kind = "TAB"
    limit = 50
    suite = "nodes"
    save_run = False
    
    if run_new:
        dataset_kind = st.sidebar.selectbox("Dataset par défaut", ["TAB", "ratbench", "anonymization"])
        limit = int(st.sidebar.number_input("Limit (documents)", min_value=1, max_value=500, value=50, step=10))
        suite = st.sidebar.selectbox("Suite (nodes, ner_presets, ner_ensemble...)", ["nodes", "ner_presets", "ner_ensemble", "full"])
        save_run = st.sidebar.checkbox("Sauvegarder", value=True)
        
    return AblationConfig(
        run_new=run_new,
        dataset_kind=dataset_kind,
        limit=limit,
        suite=suite,
        save_run=save_run
    )
