from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, Iterable, Mapping, Optional, Sequence, Tuple

Span = Tuple[int, int, str]

CANONICAL_LABELS: FrozenSet[str] = frozenset(
    {
        "PER",
        "ORG",
        "LOC",
        "DATE",
        "DOB",
        "EMAIL",
        "PHONE",
        "ADDRESS",
        "USERNAME",
        "IP",
        "URL",
        "CREDIT_CARD",
        "IBAN",
        "SSN",
        "NATIONAL_ID",
        "PASSPORT",
        "OCCUPATION",
        "DEMOGRAPHIC",
        "QUASI_ID",
        "TECH_ID",
        "AMOUNT",
        "MISC",
        "SENSITIVE",
    }
)

_BASE_CANONICAL_LABEL_MAP: Dict[str, str] = {
    "PERSON": "PER",
    "PERSONNE": "PER",
    "NAME": "PER",
    "NOM": "PER",
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "COMPANY": "ORG",
    "ENTREPRISE": "ORG",
    "LOCATION": "LOC",
    "GPE": "LOC",
    "CITY": "LOC",
    "COUNTRY": "LOC",
    "FACILITY": "LOC",
    "ADDRESS": "ADDRESS",
    "ADDR": "ADDRESS",
    "ADRESSE": "ADDRESS",
    "MAIL": "EMAIL",
    "E-MAIL": "EMAIL",
    "EMAIL ADDRESS": "EMAIL",
    "TELEPHONE": "PHONE",
    "PHONE NUMBER": "PHONE",
    "MOBILE PHONE NUMBER": "PHONE",
    "LANDLINE PHONE NUMBER": "PHONE",
    "IP ADDRESS": "IP",
    "DATE OF BIRTH": "DOB",
    "DATE_OF_BIRTH": "DOB",
    "BIRTHDATE": "DOB",
    "BIRTH_DATE": "DOB",
    "DATE DE NAISSANCE": "DOB",
    "CREDIT CARD": "CREDIT_CARD",
    "CREDIT CARD NUMBER": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "CARD": "CREDIT_CARD",
    "BANK ACCOUNT NUMBER": "IBAN",
    "SOCIAL SECURITY NUMBER": "SSN",
    "SOCIAL_SECURITY_NUMBER": "SSN",
    "NIR": "SSN",
    "NATIONAL ID NUMBER": "NATIONAL_ID",
    "NATIONAL_ID_NUMBER": "NATIONAL_ID",
    "IDENTITY CARD NUMBER": "NATIONAL_ID",
    "PASSPORT NUMBER": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "AGE": "DEMOGRAPHIC",
    "SEX": "DEMOGRAPHIC",
    "GENDER": "DEMOGRAPHIC",
    "RACE": "DEMOGRAPHIC",
    "ETHNICITY": "DEMOGRAPHIC",
    "NATIONALITY": "DEMOGRAPHIC",
    "MARITAL STATUS": "DEMOGRAPHIC",
    "MARITAL_STATUS": "DEMOGRAPHIC",
    "EMPLOYMENT STATUS": "DEMOGRAPHIC",
    "EMPLOYMENT_STATUS": "DEMOGRAPHIC",
    "EDUCATIONAL BACKGROUND": "DEMOGRAPHIC",
    "EDUCATIONAL_BACKGROUND": "DEMOGRAPHIC",
    "CITIZENSHIP STATUS": "DEMOGRAPHIC",
    "CITIZENSHIP_STATUS": "DEMOGRAPHIC",
    "POSTAL CODE": "ADDRESS",
    "POSTAL_CODE": "ADDRESS",
    "ZIP CODE": "ADDRESS",
    "ZIP_CODE": "ADDRESS",
    "JOB": "OCCUPATION",
    "PROFESSION": "OCCUPATION",
    "ROLE": "OCCUPATION",
    "COURT": "ORG",
    "JUDGE": "PER",
    "LAWYER": "PER",
    "APPLICANT": "PER",
    "LEGAL CASE NUMBER": "MISC",
    "LEGAL_CASE_NUMBER": "MISC",
    "CASE ID": "MISC",
    "CASE_ID": "MISC",
    "LAW": "MISC",
    "MONTANT": "AMOUNT",
    "MONEY": "AMOUNT",
    "COREF": "QUASI_ID",
    "ID": "TECH_ID",
    "API_KEY": "TECH_ID",
    "API KEY": "TECH_ID",
    "API": "TECH_ID",
    "AWS_KEY": "TECH_ID",
    "ERROR_CODE": "TECH_ID",
    "CONFIG": "TECH_ID",
    "LICENSE": "TECH_ID",
    "SECRET": "TECH_ID",
    "TICKET": "TECH_ID",
    "UUID": "TECH_ID",
    "MAC": "TECH_ID",
    "PATH": "TECH_ID",
    "HOST": "TECH_ID",
    "QUASI IDENTIFIER": "QUASI_ID",
    "QUASI_IDENTIFIER": "QUASI_ID",
    "DATE_REL": "DATE",
    "TEAM": "MISC",
    "PROJECT": "MISC",
    "TOOL": "MISC",
    "MODEL": "MISC",
    "STYLE": "MISC",
    "EVENT": "MISC",
    "PRODUCT": "MISC",
    "WORK OF ART": "MISC",
    "WORK_OF_ART": "MISC",
}

_PRODUCTION_MASK_POLICY: Dict[str, str] = {
    "PER": "pseudo",
    "EMAIL": "mask",
    "PHONE": "mask",
    "IP": "mask",
    "IBAN": "redact",
    "SSN": "redact",
    "CREDIT_CARD": "redact",
    "NATIONAL_ID": "redact",
    "PASSPORT": "redact",
    "ADDRESS": "generalize",
    "LOC": "generalize",
    "DATE": "generalize",
    "DOB": "generalize",
    "OCCUPATION": "generalize",
    "DEMOGRAPHIC": "generalize",
    "QUASI_ID": "generalize",
    "TECH_ID": "redact",
    "USERNAME": "pseudo",
    "URL": "mask",
    "AMOUNT": "generalize",
    "MISC": "generalize",
    "SENSITIVE": "redact",
}

_TAB_BENCHMARK_MAP: Dict[str, str] = {label: "SENSITIVE" for label in CANONICAL_LABELS}
_RATBENCH_BENCHMARK_MAP: Dict[str, str] = {
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ADDRESS": "ADDRESS",
    "DATE": "DATE",
    "DOB": "DATE",
    "EMAIL": "EMAIL",
    "PHONE": "PHONE",
    "SSN": "SSN",
    "CREDIT_CARD": "CREDIT_CARD",
    "IBAN": "SENSITIVE",
    "NATIONAL_ID": "SENSITIVE",
    "PASSPORT": "SENSITIVE",
    "OCCUPATION": "OCCUPATION",
    "DEMOGRAPHIC": "DEMOGRAPHIC",
    "QUASI_ID": "DEMOGRAPHIC",
    "USERNAME": "SENSITIVE",
    "IP": "SENSITIVE",
    "URL": "SENSITIVE",
    "TECH_ID": "SENSITIVE",
    "AMOUNT": "SENSITIVE",
    "ORG": "SENSITIVE",
    "MISC": "SENSITIVE",
    "SENSITIVE": "SENSITIVE",
}
_CONLL_BENCHMARK_MAP: Dict[str, str] = {
    "PER": "PER",
    "ORG": "ORG",
    "LOC": "LOC",
    "MISC": "MISC",
    "ADDRESS": "LOC",
}
_DBBIO_BENCHMARK_MAP: Dict[str, str] = {"PER": "PERSON", "PERSON": "PERSON"}

TAB_LEGAL_GLINER_LABELS: Tuple[str, ...] = (
    "Person",
    "Organization",
    "Location",
    "GPE",
    "Facility",
    "Date",
    "Law",
    "Legal case number",
    "Court",
    "Judge",
    "Lawyer",
    "Applicant",
    "Role",
)

RATBENCH_GLINER_LABELS: Tuple[str, ...] = (
    "Person",
    "Email address",
    "Phone number",
    "Address",
    "Credit card number",
    "Social security number",
    "National ID number",
    "Passport number",
    "Date of birth",
    "Date",
    "Age",
    "Sex",
    "Race",
    "Nationality",
    "Occupation",
    "Marital status",
    "Employment status",
    "Educational background",
    "Citizenship status",
    "Location",
    "Postal code",
)

SYNTHETIC_GLINER_LABELS: Tuple[str, ...] = (
    "Person",
    "Organization",
    "Location",
    "Date",
    "Email address",
    "Phone number",
    "Address",
    "IP address",
    "URL",
    "Username",
    "Credit card number",
    "IBAN",
    "Social security number",
    "Passport number",
    "National ID number",
    "Occupation",
    "Quasi identifier",
    "API key",
    "Ticket",
    "Amount",
)


@dataclass(frozen=True)
class EvalProfile:
    name: str
    dataset_key: str
    detection_profile: str
    gliner_labels: Tuple[str, ...]
    canonical_label_map: Mapping[str, str]
    benchmark_label_map: Mapping[str, str]
    allowed_labels: Optional[FrozenSet[str]]
    mask_policy: Mapping[str, str]
    default_mask_action: str = "generalize"


def _upper_map(mapping: Mapping[str, str]) -> Dict[str, str]:
    return {str(k).strip().upper(): str(v).strip().upper() for k, v in mapping.items()}


def _profile(
    *,
    name: str,
    dataset_key: str,
    detection_profile: str,
    gliner_labels: Sequence[str],
    benchmark_label_map: Mapping[str, str],
    allowed_labels: Optional[Iterable[str]],
    mask_policy: Optional[Mapping[str, str]] = None,
    default_mask_action: str = "generalize",
) -> EvalProfile:
    return EvalProfile(
        name=name,
        dataset_key=dataset_key,
        detection_profile=detection_profile,
        gliner_labels=tuple(gliner_labels),
        canonical_label_map=_upper_map(_BASE_CANONICAL_LABEL_MAP),
        benchmark_label_map=_upper_map(benchmark_label_map),
        allowed_labels=frozenset(str(x).upper() for x in allowed_labels) if allowed_labels else None,
        mask_policy={str(k).upper(): str(v).lower() for k, v in (mask_policy or _PRODUCTION_MASK_POLICY).items()},
        default_mask_action=default_mask_action,
    )


PROFILES: Dict[str, EvalProfile] = {
    "production_pii": _profile(
        name="production_pii",
        dataset_key="production",
        detection_profile="pii",
        gliner_labels=RATBENCH_GLINER_LABELS,
        benchmark_label_map={label: label for label in CANONICAL_LABELS},
        allowed_labels=None,
        default_mask_action="generalize",
    ),
    "tab_legal": _profile(
        name="tab_legal",
        dataset_key="tab",
        detection_profile="hybrid",
        gliner_labels=TAB_LEGAL_GLINER_LABELS,
        benchmark_label_map=_TAB_BENCHMARK_MAP,
        allowed_labels=None,
        mask_policy={label: "sensitive" for label in CANONICAL_LABELS},
        default_mask_action="sensitive",
    ),
    "ratbench_pii": _profile(
        name="ratbench_pii",
        dataset_key="ratbench",
        detection_profile="hybrid",
        gliner_labels=RATBENCH_GLINER_LABELS,
        benchmark_label_map=_RATBENCH_BENCHMARK_MAP,
        allowed_labels=None,
        mask_policy={
            **_PRODUCTION_MASK_POLICY,
            "PER": "redact",
            "ADDRESS": "redact",
            "EMAIL": "redact",
            "PHONE": "redact",
            "SSN": "redact",
            "CREDIT_CARD": "redact",
            "LOC": "generalize",
            "DATE": "generalize",
            "DOB": "generalize",
            "OCCUPATION": "generalize",
            "DEMOGRAPHIC": "generalize",
            "QUASI_ID": "generalize",
        },
        default_mask_action="generalize",
    ),
    "conll2003_news": _profile(
        name="conll2003_news",
        dataset_key="conll2003",
        detection_profile="news_ner",
        gliner_labels=(
            "Person",
            "Organization",
            "Location",
            "GPE",
            "Facility",
            "Event",
            "Product",
            "Language",
            "Law",
            "Work of Art",
            "Nationality",
        ),
        benchmark_label_map=_CONLL_BENCHMARK_MAP,
        allowed_labels={"PER", "ORG", "LOC", "MISC"},
        mask_policy={label: "generalize" for label in ("PER", "ORG", "LOC", "MISC")},
        default_mask_action="generalize",
    ),
    "dbbio_person": _profile(
        name="dbbio_person",
        dataset_key="dbbio",
        detection_profile="pii",
        gliner_labels=("Person",),
        benchmark_label_map=_DBBIO_BENCHMARK_MAP,
        allowed_labels={"PERSON"},
        mask_policy={"PER": "pseudo", "PERSON": "pseudo"},
        default_mask_action="pseudo",
    ),
    "anonymization_synthetic": _profile(
        name="anonymization_synthetic",
        dataset_key="anonymization",
        detection_profile="hybrid",
        gliner_labels=SYNTHETIC_GLINER_LABELS,
        benchmark_label_map={label: label for label in CANONICAL_LABELS},
        allowed_labels=None,
        default_mask_action="generalize",
    ),
    "personalreddit_pii": _profile(
        name="personalreddit_pii",
        dataset_key="personalreddit",
        detection_profile="hybrid",
        gliner_labels=RATBENCH_GLINER_LABELS,
        benchmark_label_map={
            **{label: label for label in CANONICAL_LABELS},
            "AGE": "DEMOGRAPHIC",
            "LOCATION": "LOC",
            "AMOUNT": "AMOUNT",
            "SENSITIVE_ATTR": "QUASI_ID",
        },
        allowed_labels=None,
        mask_policy={
            **_PRODUCTION_MASK_POLICY,
            "DEMOGRAPHIC": "generalize",
            "LOC": "generalize",
            "OCCUPATION": "generalize",
            "AMOUNT": "generalize",
            "QUASI_ID": "generalize",
        },
        default_mask_action="generalize",
    ),
}

DATASET_DEFAULT_PROFILES: Dict[str, str] = {
    "tab": "tab_legal",
    "ratbench": "ratbench_pii",
    "dbbio": "dbbio_person",
    "db-bio": "dbbio_person",
    "conll2003": "conll2003_news",
    "cleanconll2003": "conll2003_news",
    "anonymization": "anonymization_synthetic",
    "json": "anonymization_synthetic",
    "personalreddit": "personalreddit_pii",
    "personal-reddit": "personalreddit_pii",
    "reddit": "personalreddit_pii",
}


def normalize_profile_dataset_key(dataset_key: Optional[str]) -> str:
    raw = str(dataset_key or "").strip().lower()
    if raw in {"rat-bench", "rat_bench"}:
        return "ratbench"
    if raw in {"db-bio", "db_bio"}:
        return "dbbio"
    if raw in {"cleanconll2003", "cleanconll", "conll03"}:
        return "conll2003"
    if raw == "json":
        return "anonymization"
    if raw in {"personal-reddit", "reddit"}:
        return "personalreddit"
    return raw


def resolve_eval_profile(profile_name: Optional[str] = "auto", *, dataset_key: Optional[str] = None) -> EvalProfile:
    requested = str(profile_name or "auto").strip().lower()
    if requested == "auto":
        normalized_dataset = normalize_profile_dataset_key(dataset_key)
        requested = DATASET_DEFAULT_PROFILES.get(normalized_dataset, "production_pii")
    if requested not in PROFILES:
        known = ", ".join(["auto", *sorted(PROFILES)])
        raise ValueError(f"Unknown evaluation profile {profile_name!r}. Expected one of: {known}")
    return PROFILES[requested]


def canonicalize_label(label: Any, profile: Optional[EvalProfile] = None) -> str:
    raw = str(label or "SENSITIVE").strip().upper().replace("-", "_")
    raw = " ".join(raw.split())
    active_profile = profile or PROFILES["production_pii"]
    return active_profile.canonical_label_map.get(raw, _BASE_CANONICAL_LABEL_MAP.get(raw, raw))


def project_label(label: Any, profile: EvalProfile, *, target: str) -> str:
    canonical = canonicalize_label(label, profile)
    if target == "canonical":
        return canonical
    if target == "benchmark":
        return profile.benchmark_label_map.get(canonical, canonical)
    raise ValueError(f"Unsupported projection target: {target!r}")


def project_spans(spans: Sequence[Span], profile: EvalProfile, *, target: str) -> list[Span]:
    seen: set[Span] = set()
    out: list[Span] = []
    for start, end, label in spans:
        projected = (int(start), int(end), project_label(label, profile, target=target))
        if projected in seen:
            continue
        seen.add(projected)
        out.append(projected)
    out.sort(key=lambda item: (item[0], item[1], item[2]))
    return out


def apply_profile_to_config(
    config: Optional[Mapping[str, Any]],
    *,
    dataset_key: Optional[str],
    profile_name: Optional[str] = "auto",
    eval_mode: Optional[str] = None,
    masking_mode: Optional[str] = None,
) -> Dict[str, Any]:
    runtime = dict(config or {})
    active_profile = resolve_eval_profile(
        profile_name or runtime.get("profile") or runtime.get("eval_profile") or "auto",
        dataset_key=dataset_key or runtime.get("dataset_key"),
    )
    runtime["dataset_key"] = normalize_profile_dataset_key(dataset_key or runtime.get("dataset_key"))
    runtime["eval_profile"] = active_profile.name
    runtime["profile"] = active_profile.name
    runtime["entity_profile"] = active_profile.detection_profile
    runtime["gliner_label_profile"] = active_profile.detection_profile
    runtime["gliner_labels"] = list(active_profile.gliner_labels)
    runtime["eval_mode"] = str(eval_mode or runtime.get("eval_mode") or "both").lower()
    runtime["masking_mode"] = str(masking_mode or runtime.get("masking_mode") or "benchmark").lower()
    policy_profile = PROFILES["production_pii"] if runtime["masking_mode"] == "production" else active_profile
    runtime.setdefault("anon_policy", dict(policy_profile.mask_policy))
    runtime.setdefault("masking_profile", policy_profile.name)
    return runtime


def _replacement(label: str, action: str) -> str:
    upper = label.upper()
    if action == "sensitive":
        return "[SENSITIVE]"
    if action == "redact":
        return f"[{upper}_REDACTED]"
    if action == "mask":
        return f"[{upper}_MASKED]"
    if action == "pseudo":
        return f"[{upper}_PSEUDO]"
    return f"[{upper}]"


def mask_text_with_profile(
    text: str,
    spans: Sequence[Span],
    profile: EvalProfile,
) -> tuple[str, Dict[str, int]]:
    masked = text
    counts: Dict[str, int] = {}
    for start, end, label in sorted(spans, key=lambda item: item[0], reverse=True):
        if start < 0 or end > len(text) or start >= end:
            continue
        canonical = canonicalize_label(label, profile)
        action = profile.mask_policy.get(canonical, profile.mask_policy.get(label.upper(), profile.default_mask_action))
        replacement = _replacement(label, action)
        masked = masked[:start] + replacement + masked[end:]
        counts[label] = counts.get(label, 0) + 1
    return masked, counts


def profile_diagnostics(profile: EvalProfile) -> Dict[str, Any]:
    return {
        "name": profile.name,
        "dataset_key": profile.dataset_key,
        "detection_profile": profile.detection_profile,
        "gliner_labels": list(profile.gliner_labels),
        "allowed_labels": sorted(profile.allowed_labels) if profile.allowed_labels else None,
        "mask_policy": dict(profile.mask_policy),
    }


PROFILE_CHOICES: Tuple[str, ...] = ("auto", *tuple(sorted(PROFILES)))
EVAL_MODE_CHOICES: Tuple[str, ...] = ("canonical", "benchmark", "both")
MASKING_MODE_CHOICES: Tuple[str, ...] = ("production", "benchmark")
