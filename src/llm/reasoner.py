import json
import re
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

try:
    from .openrouter_client import OpenRouterClient  # type: ignore
except Exception:  # fallback si exécuté sans package
    from openrouter_client import OpenRouterClient  # type: ignore


# Placeholders / tokens à geler pendant la paraphrase (regex)
DEFAULT_FROZEN_PATTERNS = [
    r"\[[A-Z]{2,}_[A-Z0-9]+\]",          # [PER_A], [ORG_B], [IP_A], ...
    r"\[DATE_[^\]]+\]",                  # [DATE_2024-06], [DATE_2023-Q3], ...
    r"\[UUID_[A-Z0-9]+\]",
    r"\[TICKET_[A-Z0-9]+\]",
    r"\bERR_[A-Z0-9_]+\b",
    r"\bHTTP/\d\.\d\b",
    r"\b(?:[45]\d{2})\b",                # HTTP 4xx/5xx
]

PLACEHOLDER_RE = re.compile(r"\[[A-Z]{2,}_[A-Z0-9]+\]")


@dataclass
class SeedSpan:
    type: str
    start: int
    end: int
    surface: str


@dataclass
class DetectionPlan:
    entities: List[Dict[str, Any]]
    relations: List[Dict[str, Any]]
    generalizations: List[Dict[str, Any]]
    edits: List[Dict[str, Any]]
    notes: List[str]


DETECTION_SYSTEM = (
    "You are a FR/EN privacy reasoner for de-identification with utility preservation. "
    "Return a STRICT JSON object only (no prose, no markdown, no code fences). "
    "If you cannot comply, return {}. Do not leak original text. Use conservative actions if unsure."
)

DETECTION_USER = """\
POLICY:
{policy_json}

SEEDS (pre-detections with offsets):
{seeds_json}

TASK:
1) Read ORIGINAL_TEXT and the SEEDS. Find missing direct identifiers and quasi-identifiers, infer SAME_AS clusters (co-reference) and useful categories (APP/API/TEAM/PROJECT/SERVICE/ARTEFACT/HOST/USERNAME/TICKET).
2) For each mention, propose an action in {{REPLACE, KEEP, GENERALIZE, REDACT}} with:
   - type (PER, ORG, LOC, HOST, USERNAME, APP, API, PROJECT, SERVICE, ARTEFACT, TICKET, IP, URL, PATH, UUID, DATE, AMOUNT, SECRET, CARD, ...),
   - cluster_id (stable ID per entity), canonical form (never reveal sensitive raw; can be a safe label),
   - placeholder_type (e.g., PER/ORG/API/...), justification,
   - risk vector (rarity/linkage/sensitivity/confidence ∈ [0,1]).
3) Suggest targeted generalizations (e.g., DATE→month/quarter, IP→/24/public-private, AMOUNT→bin) with offsets and exact replacement strings preserving technical sense.
4) Produce a final, NON-OVERLAPPING 'edits' list: {{ "span":[start,end], "with":"replacement" }}. Edits must not modify any existing placeholder like [TYPE_X].

STRICT SCHEMA:
{{
  "entities": [
    {{
      "id": "e1",
      "type": "PER",
      "surface": "Jean D.",
      "start": 120,
      "end": 127,
      "cluster_id": "C12",
      "canonical": "PERSON_1",
      "action": "REPLACE",
      "placeholder_type": "PER",
      "risk": {{"rarity": 0.6, "linkage": 0.7, "sensitivity": 0.9, "confidence": 0.8}},
      "justification": "co-reference with 'Jean Dupont'"
    }}
  ],
  "relations": [{{"type":"SAME_AS","from":"e1","to":"e3","confidence":0.9}}],
  "generalizations": [
    {{
      "type":"DATE_MONTH",
      "start": 402, "end": 412,
      "surface":"2024-06-12",
      "replacement":"[DATE_2024-06]",
      "justification":"policy level L2"
    }}
  ],
  "edits": [
    {{"span":[120,127], "with":"[PER_A]"}},
    {{"span":[402,412], "with":"[DATE_2024-06]"}}
  ],
  "notes": []
}}

ORIGINAL_TEXT:
<<<
{text}
>>>
"""

PARAPHRASE_SYSTEM = (
    "Normalize writing style to reduce stylometric signals while strictly preserving the given frozen tokens. "
    "Do NOT modify placeholders like [TYPE_ID], error codes, or log structures. "
    "Return JSON only (no markdown, no code fences): {\"text\": \"...\"}. "
    "If unable to comply, return the original text in the same JSON shape."
)
PARAPHRASE_USER = """{{
  "text": {text_json},
  "constraints": {{
    "frozen_tokens": {frozen_json},
    "no_code_modification": true
  }}
}}"""

AUDIT_SYSTEM = (
    "You are a red-team auditor for re-identification risk. "
    "Analyze the anonymized text for linkage vectors, implicit clues, and stylometry residue. "
    "Return a STRICT JSON object only (no prose, no markdown, no code fences). "
    "If you cannot comply, return {\"risk_score\": 100, \"findings\": [], \"recommendations\": []}."
)
AUDIT_USER = """{{
  "text": {text_json}
}}"""


def _json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _safe_list(val: Any) -> List[Any]:
    return val if isinstance(val, list) else []


class LLMReasoner:
    def __init__(
        self,
        client: OpenRouterClient,
        model_detect: str,
        model_paraphrase: Optional[str] = None,
        model_audit: Optional[str] = None,
        frozen_patterns: Optional[List[str]] = None,
    ):
        self.client = client
        self.model_detect = model_detect
        self.model_paraphrase = model_paraphrase or model_detect
        self.model_audit = model_audit or model_detect
        self.frozen_patterns = frozen_patterns or DEFAULT_FROZEN_PATTERNS

    def _coerce_detect_response(self, resp: Any) -> Dict[str, Any]:
        """
        Coerce les réponses non-objet en une forme de plan de détection attendue.
        - dict => tel quel
        - list => heuristique vers edits / entities / notes
        - autre => objet vide
        """
        if isinstance(resp, dict):
            return resp
        if isinstance(resp, list):
            if all(isinstance(x, dict) for x in resp):
                # ressemble à une liste d'edits ?
                if all(("span" in x and "with" in x) for x in resp):
                    return {"entities": [], "relations": [], "generalizations": [], "edits": resp, "notes": []}
                # sinon traiter comme entities
                return {"entities": resp, "relations": [], "generalizations": [], "edits": [], "notes": []}
            if all(isinstance(x, str) for x in resp):
                return {"entities": [], "relations": [], "generalizations": [], "edits": [], "notes": resp}
        return {}

    def detect_and_plan(self, text: str, seeds: List[SeedSpan], policy: Dict[str, Any]) -> DetectionPlan:
        seeds_json = _json([s.__dict__ for s in seeds])
        policy_json = _json(policy)
        user = DETECTION_USER.format(text=text, seeds_json=seeds_json, policy_json=policy_json)
        resp_raw = self.client.call_json(DETECTION_SYSTEM, user, model=self.model_detect, temperature=0.1)
        resp = self._coerce_detect_response(resp_raw)

        # Minimal structure validation + defaults
        entities = _safe_list(resp.get("entities"))
        relations = _safe_list(resp.get("relations"))
        generalizations = _safe_list(resp.get("generalizations"))
        edits = _safe_list(resp.get("edits"))
        notes = _safe_list(resp.get("notes"))

        # Ensure edits are well-formed: {span:[s,e], with: "..."} and non-overlapping
        valid_edits: List[Dict[str, Any]] = []
        for ed in edits:
            span = ed.get("span")
            rep = ed.get("with")
            if isinstance(span, list) and len(span) == 2 and isinstance(span[0], int) and isinstance(span[1], int) and isinstance(rep, str):
                valid_edits.append({"span": [span[0], span[1]], "with": rep})

        # Sort and de-overlap edits (keep earliest non-overlapping)
        valid_edits = sorted(valid_edits, key=lambda x: (x["span"][0], x["span"][1]))
        merged: List[Dict[str, Any]] = []
        last_end = -1
        for ed in valid_edits:
            s, e = ed["span"]
            if s >= last_end:
                merged.append(ed)
                last_end = e
        return DetectionPlan(
            entities=entities,
            relations=relations,
            generalizations=generalizations,
            edits=merged,
            notes=notes,
        )

    def paraphrase(self, text: str, temperature: float = 0.3, ensure_placeholders_preserved: bool = True) -> str:
        frozen_json = _json(self.frozen_patterns)
        user = PARAPHRASE_USER.format(text_json=_json(text), frozen_json=frozen_json)
        resp = self.client.call_json(
            PARAPHRASE_SYSTEM,
            user,
            model=self.model_paraphrase,
            temperature=temperature,
        )
        out = text
        if isinstance(resp, dict) and isinstance(resp.get("text"), str):
            out = resp["text"]
        elif isinstance(resp, str):
            out = resp
        elif isinstance(resp, list):
            for item in resp:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    out = item["text"]
                    break
                if isinstance(item, str):
                    out = item
                    break

        if not isinstance(out, str):
            return text

        if ensure_placeholders_preserved:
            before = list(PLACEHOLDER_RE.findall(text))
            after = list(PLACEHOLDER_RE.findall(out))
            # All placeholders present before must remain present (multiset inclusion)
            from collections import Counter
            cb, ca = Counter(before), Counter(after)
            for k, v in cb.items():
                if ca.get(k, 0) < v:
                    # Fallback to the original if placeholders altered
                    return text
        return out

    def audit(self, anonymized_text: str) -> Dict[str, Any]:
        user = AUDIT_USER.format(text_json=_json(anonymized_text))
        report = self.client.call_json(AUDIT_SYSTEM, user, model=self.model_audit, temperature=0.1)
        # Coercition en dict si list / str
        if not isinstance(report, dict):
            if isinstance(report, list):
                findings = []
                for it in report:
                    if isinstance(it, dict):
                        findings.append(it)
                    else:
                        findings.append({"type": "note", "description": str(it)})
                report = {"risk_score": 80, "findings": findings, "recommendations": []}
            elif isinstance(report, str):
                report = {"risk_score": 80, "findings": [{"type": "note", "description": report}], "recommendations": []}
            else:
                report = {"risk_score": 100, "findings": [], "recommendations": []}
        # Normalize fields
        score = report.get("risk_score")
        if not isinstance(score, int):
            try:
                score = int(score)
            except Exception:
                score = 100
        report["risk_score"] = max(0, min(100, score))
        report["findings"] = _safe_list(report.get("findings"))
        if "recommended_policy_bump" in report and not isinstance(report["recommended_policy_bump"], str):
            report.pop("recommended_policy_bump", None)
        return report