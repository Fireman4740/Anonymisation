from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Tuple

from atlas_anno.constants import (
    PROMPT_CHARACTER,
    PROMPT_SCENARIO,
    PROMPT_TEXT,
    PROMPT_WORLD,
)
from atlas_anno.io import serialize
from atlas_anno.llm import OpenRouterClient
from atlas_anno.prompts import load_prompt_spec
from atlas_anno.records import (
    character_draft_from_dict,
    character_from_dict,
    document_from_dict,
    generated_text_draft_from_dict,
    scenario_draft_from_dict,
    scenario_from_dict,
    world_draft_from_dict,
    world_from_dict,
)
from atlas_anno.runtime import run_parallel_stage
from atlas_anno.schemas import CharacterDraft, CharacterProfile, DocumentRecord, GeneratedTextDraft, LLMRunMeta, ScenarioDraft, ScenarioSpec, World, WorldDraft


def _aggregate_llm_runs(runs: Dict[str, LLMRunMeta]) -> Dict[str, object]:
    total_latency = sum(run.latency_ms for run in runs.values())
    total_cost = sum(run.estimated_cost for run in runs.values())
    validation_errors: List[str] = []
    for run in runs.values():
        validation_errors.extend(run.validation_errors)
    return {
        "model": ",".join(sorted({run.model for run in runs.values()})),
        "prompt_version": {name: run.prompt_version for name, run in runs.items()},
        "llm_used": any(run.llm_used for run in runs.values()),
        "fallback_used": any(run.fallback_used for run in runs.values()),
        "retry_count": sum(run.retry_count for run in runs.values()),
        "attempt_count": sum(run.attempt_count for run in runs.values()),
        "queue_wait_ms": sum(run.queue_wait_ms for run in runs.values()),
        "cache_hit": any(run.cache_hit for run in runs.values()),
        "validation_errors": validation_errors,
        "latency_ms": total_latency,
        "estimated_cost": round(total_cost, 6),
    }


def _world_draft_from_world(world: World) -> WorldDraft:
    return WorldDraft(
        organization_name=world.organization_name,
        departments=list(world.departments),
        teams=list(world.teams),
        projects=list(world.projects),
        products=list(world.products),
        incidents=list(world.incidents),
        calendar_events=list(world.calendar_events),
    )


def _character_draft_from_character(character: CharacterProfile) -> CharacterDraft:
    return CharacterDraft(
        full_name=character.full_name,
        country=character.country,
        location=character.location,
        age_range=character.age_range,
        nationality=character.nationality,
        department=character.department,
        team=character.team,
        role=character.role,
        seniority=character.seniority,
        tenure_years=character.tenure_years,
        degrees=list(character.degrees),
        skills=list(character.skills),
        certifications=list(character.certifications),
        rare_traits=list(character.rare_traits),
        events=list(character.events),
        sensitive_attributes=list(character.sensitive_attributes),
        style_profile=character.style_profile,
    )


def _scenario_draft_from_scenario(scenario: ScenarioSpec) -> ScenarioDraft:
    return ScenarioDraft(
        unit_type=scenario.unit_type,
        recipient_role=scenario.recipient_role,
        document_goal=scenario.document_goal,
        difficulty=scenario.difficulty,
        required_signals=list(scenario.required_signals),
        implicit_signals=list(scenario.implicit_signals),
        include_signature=scenario.include_signature,
        include_direct_identifiers=scenario.include_direct_identifiers,
        include_sensitive=scenario.include_sensitive,
        urgency=scenario.urgency,
        noise_level=scenario.noise_level,
        split=scenario.split,
    )


def _default_meta(step_name: str, model: str, prompt_version: str, reason: str) -> LLMRunMeta:
    return LLMRunMeta(
        step_name=step_name,
        model=model,
        prompt_version=prompt_version,
        llm_used=False,
        fallback_used=True,
        retry_count=0,
        attempt_count=0,
        queue_wait_ms=0,
        cache_hit=False,
        validation_errors=[reason],
        latency_ms=0,
        estimated_cost=0.0,
    )


def _runtime_value(runtime_options: Dict[str, Any], key: str, default: Any) -> Any:
    if key in runtime_options:
        return runtime_options[key]
    return default


def refine_worlds(
    worlds: List[World],
    client: OpenRouterClient,
    llm_mode: str,
    runtime_options: Dict[str, Any] | None = None,
) -> tuple[List[World], Dict[str, LLMRunMeta], Dict[str, Any]]:
    runtime_options = runtime_options or {}
    prompt_spec = load_prompt_spec(PROMPT_WORLD)
    if llm_mode != "primary-fallback":
        return (
            worlds,
            {world.world_id: _default_meta("world_generation", client.settings.atlas_model_reasoning, prompt_spec.version, "llm mode disabled") for world in worlds},
            {"total_items": len(worlds), "processed_items": 0, "resumed_items": 0, "cache_hits": 0, "fallback_items": len(worlds), "llm_used_items": 0, "retry_total": 0, "attempt_total": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0, "elapsed_seconds": 0.0, "peak_concurrency": 0},
        )

    def _worker(world: World) -> tuple[World, LLMRunMeta]:
        fallback = _world_draft_from_world(world)
        user_prompt = (
            "Enrichis ce monde organisationnel synthétique pour un benchmark d'anonymisation.\n"
            "Retourne strictement un JSON avec les champs: organization_name, departments, teams, projects, products, incidents, calendar_events.\n"
            f"Monde seed:\n{json.dumps(serialize(fallback), ensure_ascii=False, indent=2)}"
        )
        value, meta = client.complete_json(
            step_name="world_generation",
            prompt_spec=prompt_spec,
            user_prompt=user_prompt,
            model=client.settings.atlas_model_reasoning,
            validator=world_draft_from_dict,
            fallback_value=serialize(fallback),
            temperature=0.0,
        )
        draft = value if isinstance(value, WorldDraft) else world_draft_from_dict(value)
        return (
            World(
                world_id=world.world_id,
                language=world.language,
                organization_id=world.organization_id,
                organization_name=draft.organization_name,
                departments=draft.departments,
                teams=draft.teams,
                projects=draft.projects,
                products=draft.products,
                incidents=draft.incidents,
                calendar_events=draft.calendar_events,
            ),
            meta,
        )

    return run_parallel_stage(
        items=worlds,
        stage_name="world_generation",
        label="worlds",
        batch_name=str(_runtime_value(runtime_options, "batch_name", "pilot_100")),
        prompt_version=prompt_spec.version,
        model=client.settings.atlas_model_reasoning,
        max_workers=int(_runtime_value(runtime_options, "reasoning_workers", 12)),
        resume_enabled=bool(_runtime_value(runtime_options, "resume_enabled", True)),
        checkpoint_every=int(_runtime_value(runtime_options, "checkpoint_every", 1)),
        item_id_fn=lambda world: world.world_id,
        worker_fn=_worker,
        result_from_dict=world_from_dict,
    )


def refine_characters(
    characters: List[CharacterProfile],
    client: OpenRouterClient,
    llm_mode: str,
    runtime_options: Dict[str, Any] | None = None,
) -> tuple[List[CharacterProfile], Dict[str, LLMRunMeta], Dict[str, Any]]:
    runtime_options = runtime_options or {}
    prompt_spec = load_prompt_spec(PROMPT_CHARACTER)
    if llm_mode != "primary-fallback":
        return (
            characters,
            {character.person_id: _default_meta("character_generation", client.settings.atlas_model_reasoning, prompt_spec.version, "llm mode disabled") for character in characters},
            {"total_items": len(characters), "processed_items": 0, "resumed_items": 0, "cache_hits": 0, "fallback_items": len(characters), "llm_used_items": 0, "retry_total": 0, "attempt_total": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0, "elapsed_seconds": 0.0, "peak_concurrency": 0},
        )

    def _worker(character: CharacterProfile) -> tuple[CharacterProfile, LLMRunMeta]:
        fallback = _character_draft_from_character(character)
        user_prompt = (
            "Améliore ce profil synthétique sans changer sa cohérence métier.\n"
            "Retourne strictement un JSON avec les champs: full_name, country, location, age_range, nationality, department, team, role, seniority, tenure_years, degrees, skills, certifications, rare_traits, events, sensitive_attributes, style_profile.\n"
            f"Profil seed:\n{json.dumps(serialize(fallback), ensure_ascii=False, indent=2)}"
        )
        value, meta = client.complete_json(
            step_name="character_generation",
            prompt_spec=prompt_spec,
            user_prompt=user_prompt,
            model=client.settings.atlas_model_reasoning,
            validator=character_draft_from_dict,
            fallback_value=serialize(fallback),
            temperature=0.0,
        )
        draft = value if isinstance(value, CharacterDraft) else character_draft_from_dict(value)
        return (
            CharacterProfile(
                person_id=character.person_id,
                full_name=draft.full_name,
                email=character.email,
                phone=character.phone,
                username=character.username,
                account_id=character.account_id,
                language=character.language,
                country=draft.country,
                location=draft.location,
                age_range=draft.age_range,
                gender=character.gender,
                nationality=draft.nationality,
                organization_id=character.organization_id,
                department=draft.department,
                team=draft.team,
                role=draft.role,
                seniority=draft.seniority,
                tenure_years=draft.tenure_years,
                degrees=draft.degrees,
                skills=draft.skills,
                certifications=draft.certifications,
                rare_traits=draft.rare_traits,
                events=draft.events,
                sensitive_attributes=draft.sensitive_attributes,
                style_profile=draft.style_profile,
            ),
            meta,
        )

    return run_parallel_stage(
        items=characters,
        stage_name="character_generation",
        label="characters",
        batch_name=str(_runtime_value(runtime_options, "batch_name", "pilot_100")),
        prompt_version=prompt_spec.version,
        model=client.settings.atlas_model_reasoning,
        max_workers=int(_runtime_value(runtime_options, "reasoning_workers", 12)),
        resume_enabled=bool(_runtime_value(runtime_options, "resume_enabled", True)),
        checkpoint_every=int(_runtime_value(runtime_options, "checkpoint_every", 1)),
        item_id_fn=lambda character: character.person_id,
        worker_fn=_worker,
        result_from_dict=character_from_dict,
    )


def refine_scenarios(
    scenarios: List[ScenarioSpec],
    characters_by_id: Dict[str, CharacterProfile],
    client: OpenRouterClient,
    llm_mode: str,
    runtime_options: Dict[str, Any] | None = None,
) -> tuple[List[ScenarioSpec], Dict[str, LLMRunMeta], Dict[str, Any]]:
    runtime_options = runtime_options or {}
    prompt_spec = load_prompt_spec(PROMPT_SCENARIO)
    if llm_mode != "primary-fallback":
        return (
            scenarios,
            {scenario.scenario_id: _default_meta("scenario_generation", client.settings.atlas_model_reasoning, prompt_spec.version, "llm mode disabled") for scenario in scenarios},
            {"total_items": len(scenarios), "processed_items": 0, "resumed_items": 0, "cache_hits": 0, "fallback_items": len(scenarios), "llm_used_items": 0, "retry_total": 0, "attempt_total": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0, "elapsed_seconds": 0.0, "peak_concurrency": 0},
        )

    def _worker(scenario: ScenarioSpec) -> tuple[ScenarioSpec, LLMRunMeta]:
        author = characters_by_id[scenario.author_id]
        fallback = _scenario_draft_from_scenario(scenario)
        user_prompt = (
            "Améliore ce brief documentaire pour un benchmark d'anonymisation.\n"
            "Retourne strictement un JSON avec les champs: unit_type, recipient_role, document_goal, difficulty, required_signals, implicit_signals, include_signature, include_direct_identifiers, include_sensitive, urgency, noise_level, split.\n"
            f"Scenario seed:\n{json.dumps(serialize(fallback), ensure_ascii=False, indent=2)}\n\n"
            f"Auteur seed:\n{json.dumps({'role': author.role, 'team': author.team, 'rare_traits': author.rare_traits, 'sensitive_attributes': author.sensitive_attributes}, ensure_ascii=False, indent=2)}"
        )
        value, meta = client.complete_json(
            step_name="scenario_generation",
            prompt_spec=prompt_spec,
            user_prompt=user_prompt,
            model=client.settings.atlas_model_reasoning,
            validator=scenario_draft_from_dict,
            fallback_value=serialize(fallback),
            temperature=0.0,
        )
        draft = value if isinstance(value, ScenarioDraft) else scenario_draft_from_dict(value)
        return (
            ScenarioSpec(
                scenario_id=scenario.scenario_id,
                domain=scenario.domain,
                unit_type=draft.unit_type,
                language=scenario.language,
                author_id=scenario.author_id,
                recipient_role=draft.recipient_role,
                document_goal=draft.document_goal,
                difficulty=draft.difficulty,
                required_signals=draft.required_signals,
                implicit_signals=draft.implicit_signals,
                include_signature=draft.include_signature,
                include_direct_identifiers=draft.include_direct_identifiers,
                include_sensitive=draft.include_sensitive,
                urgency=draft.urgency,
                noise_level=draft.noise_level,
                split=draft.split,
            ),
            meta,
        )

    return run_parallel_stage(
        items=scenarios,
        stage_name="scenario_generation",
        label="scenarios",
        batch_name=str(_runtime_value(runtime_options, "batch_name", "pilot_100")),
        prompt_version=prompt_spec.version,
        model=client.settings.atlas_model_reasoning,
        max_workers=int(_runtime_value(runtime_options, "reasoning_workers", 12)),
        resume_enabled=bool(_runtime_value(runtime_options, "resume_enabled", True)),
        checkpoint_every=int(_runtime_value(runtime_options, "checkpoint_every", 1)),
        item_id_fn=lambda scenario: scenario.scenario_id,
        worker_fn=_worker,
        result_from_dict=scenario_from_dict,
    )


def refine_document_texts(
    documents: List[DocumentRecord],
    characters_by_id: Dict[str, CharacterProfile],
    worlds_by_id: Dict[str, World],
    client: OpenRouterClient,
    llm_mode: str,
    runtime_options: Dict[str, Any] | None = None,
) -> tuple[List[DocumentRecord], Dict[str, LLMRunMeta], Dict[str, Any]]:
    runtime_options = runtime_options or {}
    prompt_spec = load_prompt_spec(PROMPT_TEXT)
    if llm_mode != "primary-fallback":
        return (
            documents,
            {document.doc_id: _default_meta("text_generation", client.settings.atlas_model_creative, prompt_spec.version, "llm mode disabled") for document in documents},
            {"total_items": len(documents), "processed_items": 0, "resumed_items": 0, "cache_hits": 0, "fallback_items": len(documents), "llm_used_items": 0, "retry_total": 0, "attempt_total": 0, "avg_latency_ms": 0.0, "p95_latency_ms": 0, "elapsed_seconds": 0.0, "peak_concurrency": 0},
        )

    def _worker(document: DocumentRecord) -> tuple[DocumentRecord, LLMRunMeta]:
        author = characters_by_id[document.author_id]
        world = worlds_by_id[document.world_id]
        fallback = GeneratedTextDraft(
            text=document.text,
            notes=list(document.metadata.get("text_notes", [])),
            grounding=list(generated_text_draft_from_dict({"text": document.text, "notes": [], "grounding": document.metadata.get("surface_grounding", [])}).grounding),
        )
        user_prompt = (
            "Redige un texte francais naturel pour un benchmark d'anonymisation.\n"
            "Retourne strictement un JSON avec les champs: text, notes, grounding.\n"
            "Chaque entree de grounding doit contenir: label, canonical_value, snippet, occurrence_hint.\n"
            "Le texte doit paraitre humain, ne pas exposer les codes machine bruts, et chaque snippet du grounding doit apparaitre dans text.\n\n"
            f"Document seed:\n{json.dumps({'domain': document.domain, 'goal': document.scenario.document_goal, 'difficulty': document.scenario.difficulty, 'required_signals': document.scenario.required_signals, 'implicit_signals': document.scenario.implicit_signals}, ensure_ascii=False, indent=2)}\n\n"
            f"Auteur seed:\n{json.dumps({'full_name': author.full_name, 'role': author.role, 'team': author.team, 'degrees': author.degrees, 'certifications': author.certifications, 'rare_traits': author.rare_traits, 'events': author.events, 'sensitive_attributes': author.sensitive_attributes}, ensure_ascii=False, indent=2)}\n\n"
            f"World seed:\n{json.dumps({'organization_name': world.organization_name, 'products': world.products, 'incidents': world.incidents}, ensure_ascii=False, indent=2)}\n\n"
            f"Signal values canoniques:\n{json.dumps(document.metadata.get('signal_values', {}), ensure_ascii=False, indent=2)}\n\n"
            f"Grounding fallback:\n{json.dumps(document.metadata.get('surface_grounding', []), ensure_ascii=False, indent=2)}\n\n"
            f"Texte de fallback:\n{document.text}"
        )
        value, meta = client.complete_json(
            step_name="text_generation",
            prompt_spec=prompt_spec,
            user_prompt=user_prompt,
            model=client.settings.atlas_model_creative,
            validator=generated_text_draft_from_dict,
            fallback_value=serialize(fallback),
            temperature=0.2,
        )
        draft = value if isinstance(value, GeneratedTextDraft) else generated_text_draft_from_dict(value)
        document.text = draft.text
        document.metadata["surface_grounding"] = [serialize(mention) for mention in draft.grounding]
        document.metadata["text_notes"] = list(draft.notes)
        return document, meta

    return run_parallel_stage(
        items=documents,
        stage_name="text_generation",
        label="texts",
        batch_name=str(_runtime_value(runtime_options, "batch_name", "pilot_100")),
        prompt_version=prompt_spec.version,
        model=client.settings.atlas_model_creative,
        max_workers=int(_runtime_value(runtime_options, "creative_workers", 8)),
        resume_enabled=bool(_runtime_value(runtime_options, "resume_enabled", True)),
        checkpoint_every=int(_runtime_value(runtime_options, "checkpoint_every", 1)),
        item_id_fn=lambda document: document.doc_id,
        worker_fn=_worker,
        result_from_dict=document_from_dict,
    )


def build_pilot_domain_schedule(documents: int) -> List[str]:
    if documents == 100:
        return ["support_ticket"] * 50 + ["email"] * 50
    schedule = ["support_ticket" if index % 2 == 0 else "email" for index in range(documents)]
    return schedule


def build_pilot_difficulty_schedule(documents: int) -> List[str]:
    if documents == 100:
        schedule = ["easy"] * 40 + ["medium"] * 35 + ["hard"] * 25
        rng = random.Random(91)
        rng.shuffle(schedule)
        return schedule
    return ["easy" if index % 5 < 2 else "medium" if index % 5 < 4 else "hard" for index in range(documents)]


def aggregate_document_metadata(
    documents: List[DocumentRecord],
    world_runs: Dict[str, LLMRunMeta],
    character_runs: Dict[str, LLMRunMeta],
    scenario_runs: Dict[str, LLMRunMeta],
    text_runs: Dict[str, LLMRunMeta],
) -> None:
    for document in documents:
        llm_runs = {
            "world": world_runs[document.world_id],
            "character": character_runs[document.author_id],
            "scenario": scenario_runs[document.scenario.scenario_id],
            "text": text_runs[document.doc_id],
        }
        document.metadata["llm_runs"] = {name: serialize(meta) for name, meta in llm_runs.items()}
        document.metadata["llm_audit"] = _aggregate_llm_runs(llm_runs)
