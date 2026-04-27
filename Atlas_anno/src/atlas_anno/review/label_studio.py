from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from atlas_anno.annotation.preannotator import build_annotation_bundle_from_reviewed_spans
from atlas_anno.config import load_config
from atlas_anno.constants import LABEL_STUDIO_FROM_NAME, LABEL_STUDIO_TO_NAME
from atlas_anno.io import read_json, serialize
from atlas_anno.schemas import AnnotationSpan, LabelStudioPrediction, LabelStudioTask
from atlas_anno.storage import (
    label_studio_export_path,
    load_batch_manifest,
    load_documents,
    reviewed_annotations_path,
    save_batch_manifest,
    save_label_config,
    save_label_studio_export,
    save_label_studio_tasks,
    save_reviewed_documents,
)


def _all_labels() -> List[str]:
    ontology = load_config().ontology
    labels: List[str] = []
    for group_name in ["direct_identifiers", "quasi_identifiers", "sensitive_attributes", "style_signals"]:
        labels.extend(ontology.get(group_name, []))
    return labels


def _label_config_xml(labels: List[str]) -> str:
    label_lines = "\n".join(f'    <Label value="{label}" />' for label in labels)
    return (
        "<View>\n"
        '  <Text name="text" value="$text" />\n'
        f'  <Labels name="{LABEL_STUDIO_FROM_NAME}" toName="{LABEL_STUDIO_TO_NAME}">\n'
        f"{label_lines}\n"
        "  </Labels>\n"
        "</View>\n"
    )


def _prediction_result(doc_id: str, spans: List[AnnotationSpan]) -> LabelStudioPrediction:
    result = []
    total_conf = 0.0
    for index, span in enumerate(spans):
        total_conf += span.confidence
        result.append(
            {
                "id": f"{doc_id}_{index}",
                "from_name": LABEL_STUDIO_FROM_NAME,
                "to_name": LABEL_STUDIO_TO_NAME,
                "type": "labels",
                "value": {
                    "start": span.start,
                    "end": span.end,
                    "text": span.text,
                    "labels": [span.label],
                },
                "score": span.confidence,
            }
        )
    avg_conf = round(total_conf / len(spans), 4) if spans else 0.0
    return LabelStudioPrediction(model_version="atlas_anno_hybrid", score=avg_conf, result=result)


def _selected_documents(selection: str) -> List[object]:
    documents = load_documents(annotated=True)
    if selection == "review-required":
        return [document for document in documents if bool(document.metadata.get("human_review_required"))]
    return documents


def export_label_studio_review_pack(batch: str, selection: str = "all") -> Dict[str, str]:
    documents = _selected_documents(selection)
    tasks: List[LabelStudioTask] = []
    for document in documents:
        review_bundle = document.metadata.get("review_target_annotations") or serialize(document.annotations)
        spans = [AnnotationSpan(**span) for span in review_bundle.get("spans", [])]
        task = LabelStudioTask(
            id=document.doc_id,
            data={
                "text": document.text,
                "doc_id": document.doc_id,
                "batch_name": batch,
                "domain": document.domain,
                "difficulty": document.metadata.get("difficulty", "medium"),
                "split": document.split,
                "human_review_required": bool(document.metadata.get("human_review_required", document.annotations.human_review_required)),
            },
            predictions=[_prediction_result(document.doc_id, spans)],
        )
        tasks.append(task)

    tasks_path = save_label_studio_tasks(batch, [serialize(task) for task in tasks])
    config_path = save_label_config(batch, _label_config_xml(_all_labels()))
    manifest = load_batch_manifest(batch)
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.update(
        {
            "label_studio_tasks": str(tasks_path),
            "label_studio_config": str(config_path),
        }
    )
    manifest["artifacts"] = artifacts
    manifest["review"] = {
        "target": "label-studio",
        "selection": selection,
        "tasks_total": len(tasks),
        "human_review_required_total": sum(1 for task in tasks if task.data["human_review_required"]),
        "exported_at": datetime.now(timezone.utc).isoformat(),
    }
    save_batch_manifest(batch, manifest)
    return {"tasks": str(tasks_path), "config": str(config_path)}


def _annotation_results(task: Dict[str, object]) -> List[Dict[str, object]]:
    annotations = task.get("annotations")
    if isinstance(annotations, list) and annotations:
        annotation = annotations[-1]
        if isinstance(annotation, dict):
            result = annotation.get("result", [])
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
    predictions = task.get("predictions")
    if isinstance(predictions, list) and predictions:
        prediction = predictions[-1]
        if isinstance(prediction, dict):
            result = prediction.get("result", [])
            if isinstance(result, list):
                return [item for item in result if isinstance(item, dict)]
    return []


def _task_doc_id(task: Dict[str, object]) -> str:
    data = task.get("data", {})
    if isinstance(data, dict):
        doc_id = data.get("doc_id")
        if doc_id:
            return str(doc_id)
    return str(task.get("id", ""))


def _reviewed_spans(task: Dict[str, object], text: str) -> List[AnnotationSpan]:
    spans: List[AnnotationSpan] = []
    for item in _annotation_results(task):
        if item.get("type") != "labels":
            continue
        value = item.get("value", {})
        if not isinstance(value, dict):
            continue
        labels = value.get("labels", [])
        if not isinstance(labels, list) or not labels:
            continue
        start = int(value.get("start", 0))
        end = int(value.get("end", start))
        if end < start:
            continue
        snippet = str(value.get("text") or text[start:end])
        spans.append(
            AnnotationSpan(
                start=start,
                end=end,
                label=str(labels[0]),
                text=snippet,
                confidence=1.0,
                source="review",
            )
        )
    return spans


def import_label_studio_review_pack(batch: str, input_path: str) -> Dict[str, object]:
    exported_tasks = read_json(Path(input_path))
    if not isinstance(exported_tasks, list):
        raise ValueError("Label Studio export must be a JSON list")

    documents = load_documents(annotated=True)
    documents_by_id = {document.doc_id: document for document in documents}
    review_map: Dict[str, Dict[str, object]] = {}
    for task in exported_tasks:
        if not isinstance(task, dict):
            continue
        doc_id = _task_doc_id(task)
        if doc_id:
            review_map[doc_id] = task

    reviewed_documents = []
    reviewed_total = 0
    imported_at = datetime.now(timezone.utc).isoformat()
    for document in documents:
        task = review_map.get(document.doc_id)
        if task is not None:
            bundle = build_annotation_bundle_from_reviewed_spans(document, _reviewed_spans(task, document.text))
            document.annotations = bundle
            document.metadata["reviewed_annotations"] = serialize(bundle)
            document.metadata["review_status"] = "reviewed"
            document.metadata["review_source"] = "label-studio"
            document.metadata["review_imported_at"] = imported_at
            document.metadata["human_review_required"] = bundle.human_review_required
            reviewed_total += 1
        else:
            document.metadata.setdefault("review_status", "machine-predicted")
        reviewed_documents.append(document)

    reviewed_path = save_reviewed_documents(reviewed_documents)
    raw_export_path = save_label_studio_export(batch, exported_tasks)
    manifest = load_batch_manifest(batch)
    artifacts = dict(manifest.get("artifacts", {}))
    artifacts.update(
        {
            "label_studio_export": str(raw_export_path),
            "reviewed_annotations": str(reviewed_path),
        }
    )
    manifest["artifacts"] = artifacts
    manifest["review_roundtrip"] = {
        "target": "label-studio",
        "input_path": str(Path(input_path)),
        "stored_export": str(label_studio_export_path(batch)),
        "reviewed_annotations": str(reviewed_annotations_path()),
        "tasks_total": len(exported_tasks),
        "reviewed_total": reviewed_total,
        "synced_at": imported_at,
    }
    save_batch_manifest(batch, manifest)
    return {
        "input_path": str(Path(input_path)),
        "stored_export": str(raw_export_path),
        "reviewed_annotations": str(reviewed_path),
        "reviewed_total": reviewed_total,
    }
