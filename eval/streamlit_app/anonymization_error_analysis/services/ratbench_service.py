"""Service Streamlit pour l'évaluation RAT-Bench sur PipeGraph."""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from ..core.errors import AppError
from ..loaders.pipeline_cache import load_pipegraph_cached


def run_ratbench_eval(
    *,
    language: str,
    level: Optional[int],
    limit: Optional[int],
    config: Dict[str, Any],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    enable_risk_eval: bool = False,
) -> Dict[str, Any]:
    """Lance l'évaluation RAT-Bench sur PipeGraph.

    Returns le dict complet :
        summary, by_difficulty, by_scenario, direct_id_detection_rates, details
    """
    try:
        from eval.ratbench_loader import build_docs_from_ratbench, load_ratbench_profiles
        from eval.core.ratbench import (
            build_ratbench_report,
            build_ratbench_result,
        )
    except Exception as exc:
        raise AppError(
            "Impossible d'importer les modules RAT-Bench", details=str(exc)
        ) from exc

    try:
        create_pipeline_graph, create_initial_state = load_pipegraph_cached()
        pipeline = create_pipeline_graph()
    except Exception as exc:
        raise AppError("Impossible de charger PipeGraph", details=str(exc)) from exc

    try:
        profiles = load_ratbench_profiles(language=language, level=level, limit=limit)
        docs = build_docs_from_ratbench(language=language, level=level, limit=limit)
    except Exception as exc:
        raise AppError(
            f"Impossible de charger RAT-Bench (language={language}, level={level})",
            details=str(exc),
        ) from exc

    try:
        report = build_ratbench_report(
            docs,
            profiles,
            pipeline,
            create_initial_state,
            config=config,
            progress_cb=progress_cb,
        )
    except Exception as exc:
        raise AppError("Échec de l'évaluation RAT-Bench", details=str(exc)) from exc

    result = build_ratbench_result(
        report=report,
        language=language,
        level=level,
        config=config,
    )

    # --- Agrégats LLM / RUPTA ---
    privacy_scores = [
        d["privacy_score"] for d in report if d.get("privacy_score") is not None
    ]
    avg_privacy_score = (
        round(sum(privacy_scores) / len(privacy_scores), 1) if privacy_scores else None
    )
    rupta_iters = [d.get("rupta_iterations", 0) for d in report]
    avg_rupta_iterations = (
        round(sum(rupta_iters) / len(rupta_iters), 2) if rupta_iters else 0.0
    )
    docs_with_rupta = sum(1 for i in rupta_iters if i > 0)
    llm_entities_total = sum(d.get("llm_entities", 0) for d in report)

    # --- Risque de ré-identification (optionnel) ---
    risk_result: Optional[Dict[str, Any]] = None
    if enable_risk_eval:
        try:
            from eval.evaluate_ratbench_risk import evaluate_ratbench_risk_from_pipeline

            print("🛡️ Démarrage évaluation du risque de ré-identification...")

            risk_result = evaluate_ratbench_risk_from_pipeline(
                pipeline=pipeline,
                create_initial_state=create_initial_state,
                profiles=profiles,
                config=config,
                limit=limit,
                report=report,
            )
            print("🛡️ Évaluation du risque terminée.")
            if risk_result and risk_result.get("error"):
                import logging
                logging.getLogger("ratbench_service").warning(
                    f"Risk eval warning: {risk_result['error']}"
                )
        except Exception as exc:
            import logging
            logging.getLogger("ratbench_service").warning(
                f"Risk evaluation failed (non-blocking): {exc}"
            )

    result["summary"].update(
        {
            "avg_privacy_score": avg_privacy_score,
            "avg_rupta_iterations": avg_rupta_iterations,
            "docs_with_rupta": docs_with_rupta,
            "llm_entities_total": llm_entities_total,
        }
    )

    if risk_result and not risk_result.get("error"):
        result["reid_risk"] = risk_result.get("metrics", {})
        
        # Injecter le risque individuel dans chaque doc du report pour affichage Inspector
        detailed_df = risk_result.get("detailed_results")
        if detailed_df is not None and not detailed_df.empty:
            risk_map = {}
            risk_map_by_text = {}
            for _, row in detailed_df.iterrows():
                # On utilise doc_id comme clé (c'est l'ID RAT-Bench)
                doc_id = row.get("doc_id")
                if doc_id:
                    risk_map[str(doc_id)] = {
                        "risk": row.get("risk", 0),
                        "k": row.get("k", 0),
                        "correct_attrs": row.get("num_correct_attrs", 0),
                        "correct_attr_names": row.get("correct_attrs", []),
                        "inferred_attrs": row.get("inferred_attrs", {}),
                    }
                full_text = row.get("full_text")
                if full_text:
                    risk_map_by_text[str(full_text)] = {
                        "risk": row.get("risk", 0),
                        "k": row.get("k", 0),
                        "correct_attrs": row.get("num_correct_attrs", 0),
                        "correct_attr_names": row.get("correct_attrs", []),
                        "inferred_attrs": row.get("inferred_attrs", {}),
                    }
            
            for doc in result["details"]:
                d_id = str(doc.get("doc_id"))
                if d_id in risk_map:
                    doc["reid_risk_assessment"] = risk_map[d_id]
                    continue
                d_text = str(doc.get("full_text", ""))
                if d_text and d_text in risk_map_by_text:
                    doc["reid_risk_assessment"] = risk_map_by_text[d_text]

    return result
