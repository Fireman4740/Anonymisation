from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PIPEGRAPH_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PIPEGRAPH_ROOT.parent
ATLAS_ROOT = WORKSPACE_ROOT / "Atlas_anno"
DEFAULT_ANNOTATED_INPUT = ATLAS_ROOT / "data" / "annotations" / "preannotations.jsonl"
DEFAULT_RAW_INPUT = ATLAS_ROOT / "data" / "raw_docs" / "raw_docs.jsonl"
DEFAULT_OUTPUT_DIR = ATLAS_ROOT / "data" / "anonymized"
PIPEGRAPH_CONFIG_PATH = PIPEGRAPH_ROOT / "config.json"
PURE_PROFILES = ("pseudo", "mask", "generalize", "redact")
SUPPORTED_PROFILES = (*PURE_PROFILES, "mixed", "all")


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_pipegraph_policy() -> Tuple[str, Dict[str, str]]:
    if not PIPEGRAPH_CONFIG_PATH.exists():
        return "pseudo", {}
    with PIPEGRAPH_CONFIG_PATH.open("r", encoding="utf-8") as handle:
        raw = json.load(handle) or {}
    anonymization = raw.get("pipeline", {}).get("nodes", {}).get("anonymization", {})
    global_strategy = str(anonymization.get("strategy", "pseudo"))
    policy: Dict[str, str] = {}
    for entity_type, config in (anonymization.get("policy") or {}).items():
        if isinstance(config, dict):
            policy[str(entity_type).upper()] = str(config.get("action", global_strategy))
        else:
            policy[str(entity_type).upper()] = str(config)
    return global_strategy, policy


def _parse_policy_json(raw: str | None) -> Dict[str, str]:
    if not raw:
        return {}
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("--anon-policy-json must decode to a JSON object")
    return {str(key).upper(): str(value) for key, value in payload.items()}


def _default_input_path() -> Path:
    if DEFAULT_ANNOTATED_INPUT.exists():
        return DEFAULT_ANNOTATED_INPUT
    return DEFAULT_RAW_INPUT


def _profile_names(profile: str) -> List[str]:
    if profile == "all":
        return list(PURE_PROFILES) + ["mixed"]
    return [profile]


def _strategy_name(prefix: str, profile: str, explicit_name: str | None) -> str:
    if explicit_name:
        return explicit_name
    return f"{prefix}_{profile}"


def _resolve_scope_id(document: Dict[str, Any], scope_mode: str, fixed_scope_id: str | None) -> str:
    if scope_mode == "author":
        return str(document.get("author_id") or document.get("doc_id") or "unknown_author")
    if scope_mode == "fixed":
        return str(fixed_scope_id or "pipegraph_scope")
    return str(document.get("doc_id") or "unknown_doc")


def _build_runtime_config(args: argparse.Namespace, profile: str, runtime_policy: Dict[str, str]) -> Dict[str, Any]:
    config: Dict[str, Any] = {
        "enable_detection": not args.disable_detection,
        "enable_deterministic": not args.disable_deterministic,
        "enable_ai": not args.disable_ai,
        "enable_anonymization": True,
        "detection_mode": args.detection_mode,
        "llm_detection": not args.disable_llm_review,
        "llm_verification": not args.disable_llm_verification,
        "llm_audit": not args.disable_llm_audit,
        "llm_paraphrase": not args.disable_llm_paraphrase,
        "rupta_enabled": not args.no_rupta,
    }
    if args.disable_llm:
        config["disable_llm"] = True
    if args.ner_provider:
        config["ner_provider"] = args.ner_provider
    if args.gliner_preset:
        config["gliner_preset"] = args.gliner_preset
    if args.gliner_threshold is not None:
        config["gliner_threshold"] = args.gliner_threshold
    if args.ner_min_vote is not None:
        config["ner_min_vote"] = args.ner_min_vote
    if args.ner_min_len is not None:
        config["ner_min_len"] = args.ner_min_len
    if args.llm_provider:
        config["llm_provider"] = args.llm_provider
    if args.rupta_max_iterations is not None:
        config["rupta_max_iterations"] = args.rupta_max_iterations
    if args.rupta_p_threshold is not None:
        config["rupta_p_threshold"] = args.rupta_p_threshold

    if profile in PURE_PROFILES:
        config["anon_strategy"] = profile
        config["anon_clear_yaml_policy"] = True
    config["anon_policy"] = dict(runtime_policy)
    return config


def _effective_action(
    entity_type: str,
    profile: str,
    yaml_global_strategy: str,
    yaml_policy: Dict[str, str],
    runtime_policy: Dict[str, str],
) -> str:
    normalized = str(entity_type).upper()
    if normalized in runtime_policy:
        return runtime_policy[normalized]
    if profile in PURE_PROFILES:
        return profile
    return yaml_policy.get(normalized, yaml_global_strategy)


def _estimate_metrics(original_text: str, anonymized_text: str, entities: List[Dict[str, Any]]) -> Tuple[float, float]:
    unique_entities = {
        (
            int(entity.get("start", -1)),
            int(entity.get("end", -1)),
            str(entity.get("type", "")),
            str(entity.get("value", "")),
        )
        for entity in entities
        if entity.get("value")
    }
    total = len(unique_entities)
    removed = sum(1 for _, _, _, value in unique_entities if value not in anonymized_text)
    privacy_gain = round((removed / total) if total else 0.0, 4)
    utility_loss = round(min(1.0, abs(len(original_text) - len(anonymized_text)) / max(1, len(original_text))), 4)
    return privacy_gain, utility_loss


def _run_profile(
    documents: List[Dict[str, Any]],
    args: argparse.Namespace,
    profile: str,
    yaml_global_strategy: str,
    yaml_policy: Dict[str, str],
    runtime_policy: Dict[str, str],
) -> Path:
    sys.path.insert(0, str(PIPEGRAPH_ROOT))
    from src.graph import GraphResources
    from src.state import create_initial_state

    strategy_name = _strategy_name(args.strategy_prefix, profile, args.strategy_name)
    runtime_config = _build_runtime_config(args, profile, runtime_policy)
    results: List[Dict[str, Any]] = []

    with GraphResources() as resources:
        pipeline = resources.graph
        total = len(documents)
        for index, document in enumerate(documents, start=1):
            original_text = str(document.get("text", ""))
            scope_id = _resolve_scope_id(document, args.scope_mode, args.scope_id)
            state = create_initial_state(original_text, dict(runtime_config))
            state["metadata"]["scope_id"] = scope_id

            try:
                final_state = pipeline.invoke(state)
            except Exception as exc:
                if args.fail_fast:
                    raise
                final_state = {
                    "text": original_text,
                    "entities": [],
                    "errors": [str(exc)],
                    "privacy_score": 0,
                    "llm_feedback": {},
                    "iteration": 0,
                }

            entities = list(final_state.get("entities", []))
            anonymized_text = str(final_state.get("text", original_text))
            privacy_gain, utility_loss = _estimate_metrics(original_text, anonymized_text, entities)
            actions_performed = sorted(
                {
                    f"pipegraph:{_effective_action(str(entity.get('type', 'UNKNOWN')), profile, yaml_global_strategy, yaml_policy, runtime_policy)}:{str(entity.get('type', 'UNKNOWN')).upper()}"
                    for entity in entities
                }
            )
            results.append(
                {
                    "doc_id": str(document.get("doc_id", f"doc_{index:06d}")),
                    "strategy": strategy_name,
                    "anonymized_text": anonymized_text,
                    "actions_performed": actions_performed,
                    "rationale": f"pipegraph profile={profile} entities={len(entities)}",
                    "estimated_privacy_gain": privacy_gain,
                    "estimated_utility_loss": utility_loss,
                    "metadata": {
                        "split": document.get("split"),
                        "difficulty": document.get("metadata", {}).get("difficulty"),
                        "profile": profile,
                        "scope_id": scope_id,
                        "pipegraph_runtime": runtime_config,
                        "entities_detected": len(entities),
                        "privacy_score": final_state.get("privacy_score"),
                        "llm_feedback": final_state.get("llm_feedback", {}),
                        "iteration": final_state.get("iteration", 0),
                        "errors": list(final_state.get("errors", [])),
                    },
                }
            )
            if index == 1 or index % 10 == 0 or index == total:
                print(f"[{strategy_name}] processed {index}/{total}")

    output_path = args.output_dir / f"{strategy_name}.jsonl"
    _write_jsonl(output_path, results)
    print(f"[{strategy_name}] wrote {len(results)} rows to {output_path}")
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run PipeGraph on Atlas jsonl documents and save Atlas-compatible anonymization outputs.")
    parser.add_argument("--input", type=Path, default=None, help="Atlas documents jsonl. Defaults to preannotations, then raw docs.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Output directory for Atlas-compatible anonymized jsonl files.")
    parser.add_argument("--profile", choices=SUPPORTED_PROFILES, default="all", help="Anonymization profile to run.")
    parser.add_argument("--strategy-prefix", default="pipegraph", help="Prefix used for output strategy names.")
    parser.add_argument("--strategy-name", default=None, help="Explicit strategy name. Only use with a single profile.")
    parser.add_argument("--scope-mode", choices=["doc", "author", "fixed"], default="doc", help="Scope used by pseudonymization.")
    parser.add_argument("--scope-id", default=None, help="Fixed scope id when --scope-mode fixed.")
    parser.add_argument("--anon-policy-json", default=None, help='Runtime policy override as JSON object, for example {"PER":"pseudo","LOC":"generalize"}.')
    parser.add_argument("--limit", type=int, default=None, help="Optional limit on the number of documents.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on the first document that fails.")

    parser.add_argument("--disable-detection", action="store_true", help="Disable the detection node.")
    parser.add_argument("--disable-deterministic", action="store_true", help="Disable regex detection.")
    parser.add_argument("--disable-ai", action="store_true", help="Disable AI NER detection.")
    parser.add_argument("--detection-mode", choices=["serial", "parallel"], default="parallel", help="Detection execution mode.")
    parser.add_argument("--ner-provider", choices=["gliner", "flair", "spacy"], default=None, help="Optional runtime NER provider override.")
    parser.add_argument("--gliner-preset", choices=["fast", "balanced", "accuracy", "best", "full", "pii", "multitask"], default=None, help="Optional GLiNER preset override.")
    parser.add_argument("--gliner-threshold", type=float, default=None, help="Optional GLiNER threshold override.")
    parser.add_argument("--ner-min-vote", type=float, default=None, help="Optional minimum aggregated vote for AI entities.")
    parser.add_argument("--ner-min-len", type=int, default=None, help="Optional minimum AI entity length.")

    parser.add_argument("--disable-llm", action="store_true", help="Disable all LLM nodes.")
    parser.add_argument("--disable-llm-review", action="store_true", help="Disable the additive LLM review node.")
    parser.add_argument("--disable-llm-verification", action="store_true", help="Disable the LLM verification node.")
    parser.add_argument("--disable-llm-audit", action="store_true", help="Disable the LLM audit node.")
    parser.add_argument("--disable-llm-paraphrase", action="store_true", help="Disable the LLM paraphrase node.")
    parser.add_argument("--llm-provider", default=None, help="Optional runtime LLM provider override.")
    parser.add_argument("--no-rupta", action="store_true", help="Disable the audit/paraphrase adversarial loop.")
    parser.add_argument("--rupta-max-iterations", type=int, default=None, help="Optional maximum number of RUPTA iterations.")
    parser.add_argument("--rupta-p-threshold", type=int, default=None, help="Optional privacy threshold used by RUPTA.")
    return parser


def main(argv: List[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.strategy_name and args.profile == "all":
        parser.error("--strategy-name cannot be used with --profile all")

    input_path = args.input or _default_input_path()
    if not input_path.exists():
        parser.error(f"input file not found: {input_path}")

    documents = _read_jsonl(input_path)
    if args.limit is not None:
        documents = documents[: args.limit]
    if not documents:
        parser.error(f"no documents found in {input_path}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    yaml_global_strategy, yaml_policy = _load_pipegraph_policy()
    runtime_policy = _parse_policy_json(args.anon_policy_json)

    written_paths = []
    for profile in _profile_names(args.profile):
        written_paths.append(
            _run_profile(
                documents=documents,
                args=args,
                profile=profile,
                yaml_global_strategy=yaml_global_strategy,
                yaml_policy=yaml_policy,
                runtime_policy=runtime_policy,
            )
        )

    print("Completed:")
    for path in written_paths:
        print(f" - {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
