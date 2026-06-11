from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Dict

from atlas_anno.io import write_json, write_text
from atlas_anno.storage import load_report, report_html_path, report_json_path, report_markdown_path


def _summary_block(name: str, payload: Dict[str, object]) -> str:
    if not payload:
        return f"## {name}\n\nNo data.\n"
    summary = payload.get("summary", {})
    lines = [f"## {name}", ""]
    for key, value in summary.items():
        lines.append(f"- `{key}`: {value}")
    lines.append("")
    return "\n".join(lines)


def _calibration_summary(payload: Dict[str, object]) -> Dict[str, object]:
    if not payload:
        return {}
    return {
        "passed": payload.get("passed"),
        "by_difficulty": payload.get("by_difficulty"),
        "by_mode": payload.get("by_mode"),
    }


def build_consolidated_report(strategy: str) -> Dict[str, object]:
    spans = load_report(strategy, "spans")
    privacy = load_report(strategy, "privacy")
    reid = load_report(strategy, "reid")
    utility = load_report(strategy, "utility")
    calibration = load_report("raw", "calibration")
    diversity = load_report("raw", "diversity")
    realism = load_report("raw", "realism")
    div_summary = diversity.get("summary", {}) if diversity else {}
    realism_summary = realism.get("summary", {}) if realism else {}
    summary = {
        "strategy": strategy,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "privacy_score": privacy.get("summary", {}).get("privacy_score"),
        "utility_score": utility.get("summary", {}).get("utility_score"),
        "reid_top1": reid.get("summary", {}).get("top1"),
        "span_f1": spans.get("summary", {}).get("f1"),
        "calibration_passed": calibration.get("passed") if calibration else None,
        "distinct_2": div_summary.get("distinct_2"),
        "self_bleu": div_summary.get("self_bleu"),
        "duplicate_rate": div_summary.get("duplicate_rate"),
        "diversity_passed": div_summary.get("passed"),
        "realism_avg_overall": realism_summary.get("avg_overall"),
        "realism_mode": realism_summary.get("mode"),
    }
    return {
        "meta": {"strategy": strategy, "generated_at": summary["created_at"]},
        "summary": summary,
        "sections": {
            "spans": spans,
            "privacy": privacy,
            "reid": reid,
            "utility": utility,
            "calibration": calibration,
            "diversity": diversity,
            "realism": realism,
        },
    }


def build_markdown(strategy: str, report: Dict[str, object]) -> str:
    sections = report.get("sections", {})
    blocks = [
        f"# Atlas_anno report: {strategy}",
        "",
        _summary_block("Spans", sections.get("spans", {})),
        _summary_block("Privacy", sections.get("privacy", {})),
        _summary_block("Re-identification", sections.get("reid", {})),
        _summary_block("Utility", sections.get("utility", {})),
        _summary_block("Calibration", {"summary": _calibration_summary(sections.get("calibration", {}))} if sections.get("calibration") else {}),
        _summary_block("Diversity", sections.get("diversity", {})),
        _summary_block("Realism", sections.get("realism", {})),
    ]
    return "\n".join(blocks)


def build_html(strategy: str, report: Dict[str, object]) -> str:
    rows = []
    for section_name, section_payload in report.get("sections", {}).items():
        summary = section_payload.get("summary", {})
        summary_rows = "".join(
            f"<tr><td>{html.escape(str(key))}</td><td>{html.escape(str(value))}</td></tr>"
            for key, value in summary.items()
        )
        rows.append(f"<h2>{html.escape(section_name.title())}</h2><table>{summary_rows}</table>")
    return (
        "<html><head><meta charset='utf-8'><title>Atlas_anno report</title>"
        "<style>body{font-family:Arial,sans-serif;margin:32px;}table{border-collapse:collapse;width:100%;}"
        "td{border:1px solid #ccc;padding:8px;}</style></head><body>"
        f"<h1>Atlas_anno report: {html.escape(strategy)}</h1>"
        + "".join(rows)
        + "</body></html>"
    )


def run_build_report_command(strategy: str) -> None:
    report = build_consolidated_report(strategy)
    write_json(report_json_path(strategy, "report"), report)
    write_text(report_markdown_path(strategy), build_markdown(strategy, report))
    write_text(report_html_path(strategy), build_html(strategy, report))

