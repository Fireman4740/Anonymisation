import json
from typing import Dict, Optional, List, Any, Tuple

from guardrails.validator_base import Validator, ValidationResult, PassResult, FailResult, register_validator
from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification

from text_sanitizer import ner_based_replacements, regexes_based_replacements
from orchestrator import anonymize_text
from policy import preset

NER_MODEL_PATH = "Davlan/bert-base-multilingual-cased-ner-hrl"
nlp = pipeline(
    "ner",
    model=AutoModelForTokenClassification.from_pretrained(NER_MODEL_PATH),
    tokenizer=AutoTokenizer.from_pretrained(NER_MODEL_PATH),
    aggregation_strategy="simple",
)

# --- NER fenêtré avec chevauchement ---
MONTHS = {m.lower(): i for i, m in enumerate([
    "", "January","February","March","April","May","June","July","August","September","October","November","December"
])}

def run_ner_chunked(
    text: str,
    max_tokens: int = 384,   # < 512 pour BERT
    stride: int = 64,        # chevauchement pour capter les entités à cheval
    min_conf: float = 0.55,  # seuil de confiance minimal
) -> List[Dict[str, Any]]:
    """Fenêtrage par tokens via le tokenizer déjà chargé dans la pipeline HF.
    Retourne des entités globales (start/end sur le texte complet)."""
    tok = nlp.tokenizer  # AutoTokenizer
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc.get("offset_mapping") or []
    if not offsets:
        return []

    # Construction des fenêtres tokenisées (avec chevauchement)
    windows: List[Tuple[int, int]] = []
    i = 0
    while i < len(offsets):
        j = min(i + max_tokens, len(offsets))
        char_start = int(offsets[i][0])
        char_end = int(offsets[j - 1][1])
        if char_end > char_start:
            windows.append((char_start, char_end))
        if j == len(offsets):
            break
        i = max(0, j - stride)

    # NER par fenêtre + dédoublonnage global sur (start,end,label)
    seen = set()
    ents: List[Dict[str, Any]] = []
    for cs, ce in windows:
        chunk = text[cs:ce]
        try:
            out = nlp(chunk)
        except Exception:
            out = []
        for ent in out:
            if ent.get("score", 0.0) < min_conf:
                continue
            s = cs + int(ent.get("start", 0))
            e = cs + int(ent.get("end", 0))
            label = str(ent.get("entity_group") or ent.get("entity") or "").upper()
            if not label or e <= s:
                continue
            key = (s, e, label)
            if key in seen:
                continue
            seen.add(key)
            ents.append({
                "start": s,
                "end": e,
                "entity_group": label,
                "word": ent.get("word"),
                "score": float(ent.get("score", 0.0))
            })

    ents.sort(key=lambda x: (x["start"], x["end"]))
    return ents

@register_validator(name="sanitize_pii", data_type="string")
class PersonalInfoSanitizer(Validator):
    """
    Validator Guardrails qui applique plusieurs couches d’anonymisation (regex, NER, LLM)
    selon la politique choisie (L0 → L4).

    Metadata possibles:
      - level (str): niveau de politique (défaut "L1")
      - scope_id (str): identifiant pour pseudonymisation déterministe
      - secret_salt (str): sel HMAC
      - use_llm (bool): force on/off LLM (écrase la policy)
    """

    def validate(self, value: str, metadata: Optional[Dict] = None) -> ValidationResult:
        metadata = metadata or {}
        try:
            level = metadata.get("level", "L1")
            scope_id = metadata.get("scope_id", "SCOPE")
            secret = metadata.get("secret_salt", "secret")

            policy = preset(level)
            if "use_llm" in metadata:
                flag = bool(metadata["use_llm"])
                policy.llm_detection = flag
                policy.llm_paraphrase = flag
                policy.llm_audit = flag

            # Chemin rapide regex + NER
            regex_replacements = regexes_based_replacements(value)
            try:
                # Utilise le NER fenêtré pour meilleure couverture et offsets stables
                ner_results = run_ner_chunked(value)
            except Exception:
                ner_results = []
            ner_replacements = ner_based_replacements(value, ner_results)
            base_replacements = regex_replacements + ner_replacements

            if not base_replacements and not policy.llm_detection:
                return PassResult()

            # Orchestration complète (LLM/généralisation/audit)
            out = anonymize_text(
                value=value,
                scope_id=scope_id,
                secret_salt=secret,
                level=level,
                ner_results=ner_results,
            )

            sanitized_value: str = out["text"]
            if sanitized_value == value:
                return PassResult()

            return FailResult(
                error_message=json.dumps(out.get("audit", {}), ensure_ascii=False),
                fix_value=sanitized_value,
            )

        except Exception as exc:
            return FailResult(
                error_message=f"Exception during anonymisation: {type(exc).__name__}: {exc}"
            )