from __future__ import annotations

from collections import Counter
from typing import Any, Dict

from atlas_anno.config import load_config
from atlas_anno.console import log
from atlas_anno.generation.llm_generation import (
    aggregate_document_metadata,
    build_pilot_difficulty_schedule,
    build_pilot_domain_schedule,
    refine_characters,
    refine_document_texts,
    refine_scenarios,
    refine_worlds,
)
from atlas_anno.attacks.pairs import build_attack_pairs
from atlas_anno.evaluation.calibration import run_difficulty_calibration
from atlas_anno.evaluation.diversity import evaluate_diversity
from atlas_anno.evaluation.register_conformance import apply_register_conformance
from atlas_anno.evaluation.realism_judge import apply_human_realism_sample
from atlas_anno.generation.auditor import audit_document
from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.llm_text_generator import generate_llm_texts
from atlas_anno.generation.scenario_planner import build_candidate_pools, build_scenarios
from atlas_anno.generation.text_generator import build_documents
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.llm import OpenRouterClient
from atlas_anno.runtime import build_runtime_options
from atlas_anno.schemas import LLMRunMeta, SCHEMA_VERSION
from atlas_anno.settings import load_settings
from atlas_anno.storage import (
    load_characters,
    load_documents,
    load_scenarios,
    load_worlds,
    save_batch_manifest,
    save_characters,
    save_documents,
    save_report,
    save_scenarios,
    save_worlds,
    worlds_path,
    characters_path,
    scenarios_path,
    raw_docs_path,
    attack_pairs_path,
    save_attack_pairs,
)


def run_generate_worlds_command(count: int) -> None:
    log(f"Generating {count} worlds")
    worlds = build_worlds(count=count)
    save_worlds(worlds)
    log("Worlds saved")


def run_generate_characters_command(per_world: int) -> None:
    log(f"Generating characters with per-world={per_world}")
    worlds = load_worlds()
    characters = build_characters(worlds, per_world=per_world)
    save_characters(characters)
    log(f"Characters saved: {len(characters)}")


def run_generate_scenarios_command(documents: int) -> None:
    log(f"Generating scenarios for {documents} documents")
    characters = load_characters()
    scenarios = build_scenarios(characters, documents=documents)
    save_scenarios(scenarios)
    log(f"Scenarios saved: {len(scenarios)}")


def run_generate_texts_command() -> None:
    log("Generating raw texts")
    worlds = load_worlds()
    characters = load_characters()
    scenarios = load_scenarios()
    candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
    documents = build_documents(worlds, characters, scenarios, candidate_pools)
    save_documents(documents, annotated=False)
    log(f"Raw documents saved: {len(documents)}")


def run_validate_dataset_command() -> None:
    log("Validating dataset")
    characters = {character.person_id: character for character in load_characters()}
    documents = load_documents_for_validation()
    issues = []
    splits = Counter()
    for document in documents:
        splits[document.split] += 1
        issues.extend(f"{document.doc_id}:{issue}" for issue in audit_document(document, characters[document.author_id]))
    if issues:
        raise RuntimeError("Dataset validation failed:\n" + "\n".join(issues[:20]))
    log("Dataset validation passed")
    log(str(dict(splits)))


def load_documents_for_validation():
    return load_documents(annotated=False)


def _deterministic_text_runs(documents: list) -> tuple[Dict[str, LLMRunMeta], Dict[str, Any]]:
    runs = {
        document.doc_id: LLMRunMeta(
            step_name="deterministic_text_generation",
            model="deterministic",
            prompt_version="deterministic",
            llm_used=False,
            fallback_used=False,
            retry_count=0,
        )
        for document in documents
    }
    return runs, {
        "total_items": len(documents),
        "processed_items": len(documents),
        "resumed_items": 0,
        "cache_hits": 0,
        "fallback_items": 0,
        "llm_used_items": 0,
        "retry_total": 0,
        "attempt_total": 0,
        "avg_latency_ms": 0.0,
        "p95_latency_ms": 0,
        "elapsed_seconds": 0.0,
        "peak_concurrency": 0,
        "repair_retries": 0,
    }


def run_generate_dataset_command(
    documents: int,
    llm_mode: str,
    reasoning_workers: int | None = None,
    creative_workers: int | None = None,
    resume_enabled: bool | None = None,
    cache_enabled: bool | None = None,
) -> None:
    config = load_config()
    pilot = config.defaults["generation"]["pilot"]
    batch_name = str(config.defaults["generation"]["pilot_batch"])
    runtime_options = build_runtime_options(
        batch_name=batch_name,
        reasoning_workers=reasoning_workers,
        creative_workers=creative_workers,
        resume_enabled=resume_enabled,
        cache_enabled=cache_enabled,
    )
    client = OpenRouterClient(load_settings(), runtime_overrides=runtime_options)

    log(
        f"Starting batch {batch_name} with documents={documents} llm_mode={llm_mode} "
        f"reasoning_workers={runtime_options['reasoning_workers']} "
        f"creative_workers={runtime_options['creative_workers']} "
        f"resume={runtime_options['resume_enabled']} cache={runtime_options['cache_enabled']}"
    )
    log("Stage 1/5: worlds")
    worlds = build_worlds(count=int(pilot["worlds"]))
    worlds, world_runs, world_stats = refine_worlds(worlds, client, llm_mode, runtime_options)
    save_worlds(worlds)

    log("Stage 2/5: characters")
    sampling_stats: Dict[str, Any] = {}
    characters = build_characters(worlds, per_world=int(pilot["characters_per_world"]), stats=sampling_stats)
    characters = characters[: int(pilot["characters_total"])]
    characters, character_runs, character_stats = refine_characters(characters, client, llm_mode, runtime_options)
    save_characters(characters)

    log("Stage 3/5: scenarios")
    domain_schedule = build_pilot_domain_schedule(documents)
    difficulty_schedule = build_pilot_difficulty_schedule(documents)
    scenarios = build_scenarios(
        characters,
        documents=documents,
        domain_schedule=domain_schedule,
        difficulty_schedule=difficulty_schedule,
        mention_mode_mix=config.defaults["generation"].get("mention_mode_mix"),
    )
    characters_by_id = {character.person_id: character for character in characters}
    scenarios, scenario_runs, scenario_stats = refine_scenarios(scenarios, characters_by_id, client, llm_mode, runtime_options)
    save_scenarios(scenarios)

    log("Stage 4/6: texts")
    candidate_pools = {character.person_id: build_candidate_pools(character, characters) for character in characters}
    documents_rows = build_documents(worlds, characters, scenarios, candidate_pools)
    worlds_by_id = {world.world_id: world for world in worlds}
    configured_text_mode = str(config.defaults["generation"].get("text_mode", "llm-first"))
    text_mode = "deterministic" if llm_mode == "disabled" else configured_text_mode
    runtime_options["text_repair_retries"] = int(config.defaults["generation"].get("text_repair_retries", 2))
    if text_mode == "hybrid":
        documents_rows, text_runs, text_stats = refine_document_texts(documents_rows, characters_by_id, worlds_by_id, client, llm_mode, runtime_options)
    elif text_mode == "llm-first":
        documents_rows, text_runs, text_stats = generate_llm_texts(documents_rows, characters_by_id, worlds_by_id, client, llm_mode, runtime_options)
    else:
        text_runs, text_stats = _deterministic_text_runs(documents_rows)
    aggregate_document_metadata(documents_rows, world_runs, character_runs, scenario_runs, text_runs)
    for document in documents_rows:
        document.metadata["batch_name"] = batch_name
        document.metadata["schema_version"] = SCHEMA_VERSION
    save_documents(documents_rows, annotated=False)

    log("Stage 5/6: attack pairs")
    attack_pairs = build_attack_pairs(documents_rows, characters_by_id)
    save_attack_pairs(attack_pairs)

    log("Stage 6/6: audit, calibration and manifest")
    issues = []
    split_counts = Counter()
    difficulty_counts = Counter()
    domain_counts = Counter()
    mention_difficulty = Counter()
    register_counts = Counter()
    for document in documents_rows:
        split_counts[document.split] += 1
        difficulty_counts[document.metadata.get("difficulty", "unknown")] += 1
        domain_counts[document.domain] += 1
        register_counts[document.metadata.get("register") or "unknown"] += 1
        mention_difficulty.update(document.metadata.get("mention_difficulty", {}))
        issues.extend(audit_document(document, characters_by_id[document.author_id]))
    if issues:
        raise RuntimeError("Dataset generation audit failed:\n" + "\n".join(issues[:20]))
    calibration = run_difficulty_calibration(documents_rows, characters_by_id, attack_pairs)

    # Stage 6b : conformité de registre + échantillon humain réalisme
    realism_config = dict(config.defaults.get("realism", {}) or {})
    register_conformance_stats = apply_register_conformance(documents_rows)
    n_realism_sampled = apply_human_realism_sample(documents_rows, realism_config)
    log(
        f"Register conformance: checked={register_conformance_stats['checked']} "
        f"failed={register_conformance_stats['failed']} "
        f"rate={register_conformance_stats['rate']}"
    )
    log(f"Realism human sample: {n_realism_sampled} documents marqués pour revue")
    enforce_realism = bool(realism_config.get("enforce", False))
    if enforce_realism and register_conformance_stats["rate"] > float(realism_config.get("max_register_mismatch_rate", 0.10)):
        raise RuntimeError(
            f"Register mismatch rate {register_conformance_stats['rate']:.3f} "
            f"> {realism_config.get('max_register_mismatch_rate', 0.10)} (realism.enforce=true)"
        )

    # Stage 6c : batterie de diversité (report-only par défaut)
    diversity_config = dict(config.defaults.get("diversity", {}) or {})
    diversity_report = evaluate_diversity(documents_rows, diversity_config)
    save_report("raw", "diversity", diversity_report)
    log(
        "Diversity: distinct2={distinct_2} selfBLEU={self_bleu} dup={duplicate_rate} "
        "coverage={cell_coverage} passed={passed}".format(**diversity_report["summary"])
    )
    if diversity_config.get("enforce") and not diversity_report["summary"]["passed"]:
        raise RuntimeError(
            "Diversity gate failed: " + "; ".join(diversity_report["summary"]["failures"])
        )
    # Documents re-sauvegardés avec metadata enrichie (register_conformance, realism_sample)
    save_documents(documents_rows, annotated=False)

    save_batch_manifest(
        batch_name,
        {
            "batch_name": batch_name,
            "worlds_total": len(worlds),
            "characters_total": len(characters),
            "documents_total": len(documents_rows),
            "llm_mode": llm_mode,
            "schema_version": SCHEMA_VERSION,
            "artifacts": {
                "worlds": str(worlds_path()),
                "characters": str(characters_path()),
                "scenarios": str(scenarios_path()),
                "raw_docs": str(raw_docs_path()),
                "attack_pairs": str(attack_pairs_path()),
            },
            "stats": {
                "schema_version": SCHEMA_VERSION,
                "text_mode": text_mode,
                "sampling": sampling_stats,
                "splits": dict(split_counts),
                "difficulties": dict(difficulty_counts),
                "domains": dict(domain_counts),
                "registers": dict(register_counts),
                "mention_difficulty": dict(mention_difficulty),
                "attack_pairs_total": len(attack_pairs),
                "calibration": calibration,
                "register_conformance": register_conformance_stats,
                "realism_sampled": n_realism_sampled,
                "diversity": {
                    "distinct_2": diversity_report["summary"]["distinct_2"],
                    "self_bleu": diversity_report["summary"]["self_bleu"],
                    "duplicate_rate": diversity_report["summary"]["duplicate_rate"],
                    "cell_coverage": diversity_report["summary"]["cell_coverage"],
                    "passed": diversity_report["summary"]["passed"],
                },
                "llm_generated_documents": sum(1 for document in documents_rows if document.metadata.get("text_generation_mode") == "llm"),
                "repair_retries": sum(int(document.metadata.get("text_repair_retries", 0)) for document in documents_rows),
                "llm_used_documents": sum(1 for document in documents_rows if document.metadata.get("llm_audit", {}).get("llm_used")),
                "fallback_documents": sum(1 for document in documents_rows if document.metadata.get("llm_audit", {}).get("fallback_used")),
                "stage_stats": {
                    "world_generation": world_stats,
                    "character_generation": character_stats,
                    "scenario_generation": scenario_stats,
                    "text_generation": text_stats,
                },
                "runtime": {
                    "reasoning_workers": runtime_options["reasoning_workers"],
                    "creative_workers": runtime_options["creative_workers"],
                    "resume_enabled": runtime_options["resume_enabled"],
                    "cache_enabled": runtime_options["cache_enabled"],
                    "checkpoint_every": runtime_options["checkpoint_every"],
                },
            },
        },
    )
    log(f"Batch ready: {batch_name}")
    log(f"Stats domains={dict(domain_counts)} difficulties={dict(difficulty_counts)} splits={dict(split_counts)}")
