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


def build_consolidated_report(strategy: str) -> Dict[str, object]:
    spans = load_report(strategy, "spans")
    privacy = load_report(strategy, "privacy")
    reid = load_report(strategy, "reid")
    utility = load_report(strategy, "utility")
    summary = {
        "strategy": strategy,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "privacy_score": privacy.get("summary", {}).get("privacy_score"),
        "utility_score": utility.get("summary", {}).get("utility_score"),
        "reid_top1": reid.get("summary", {}).get("top1"),
        "span_f1": spans.get("summary", {}).get("f1"),
    }
    return {
        "meta": {"strategy": strategy, "generated_at": summary["created_at"]},
        "summary": summary,
        "sections": {
            "spans": spans,
            "privacy": privacy,
            "reid": reid,
            "utility": utility,
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

