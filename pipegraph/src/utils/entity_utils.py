"""
Entity type normalization utilities.
Centralizes label mapping between different detection sources.
"""

from typing import Any, Dict, Optional

# Centralized entity type mapping
# Maps various source labels to standardized types
ENTITY_TYPE_MAPPING: Dict[str, str] = {
    # Deterministic/Regex variations
    "TELEPHONE": "PHONE",
    "MAIL": "EMAIL",
    "E-MAIL": "EMAIL",
    "TELEPHONE NUMBER": "PHONE",
    # GLiNER / NER common labels
    "PHONE NUMBER": "PHONE",
    "MOBILE PHONE NUMBER": "PHONE",
    "LANDLINE PHONE NUMBER": "PHONE",
    "EMAIL ADDRESS": "EMAIL",
    "IP ADDRESS": "IP",
    # Person variations
    "PERSON": "PER",
    "PERSONNE": "PER",
    "NOM": "PER",
    "NAME": "PER",
    # Organization variations
    "ORGANIZATION": "ORG",
    "ORGANISATION": "ORG",
    "COMPANY": "ORG",
    "ENTREPRISE": "ORG",
    # Location variations
    "LOCATION": "LOC",
    "GPE": "LOC",
    "FACILITY": "LOC",
    "LIEU": "LOC",
    "VILLE": "LOC",
    "CITY": "LOC",
    "COUNTRY": "LOC",
    "PAYS": "LOC",
    "ADDRESS": "ADDRESS",
    "ADDR": "ADDRESS",
    "ADRESSE": "ADDRESS",
    # Date/Time
    "DATE OF BIRTH": "DOB",
    "DATE_OF_BIRTH": "DOB",
    "BIRTHDATE": "DOB",
    "BIRTH_DATE": "DOB",
    "DATE DE NAISSANCE": "DOB",
    "DATE_REL": "DATE",
    # Financial
    "CREDIT CARD NUMBER": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD",
    "BANK ACCOUNT NUMBER": "BANK_ACCOUNT",
    "CREDIT CARD": "CREDIT_CARD",
    "CARD": "CREDIT_CARD",
    # ID documents
    "NATIONAL ID NUMBER": "NATIONAL_ID",
    "NATIONAL_ID_NUMBER": "NATIONAL_ID",
    "IDENTITY CARD NUMBER": "NATIONAL_ID",
    "PASSPORT NUMBER": "PASSPORT",
    "PASSPORT_NUMBER": "PASSPORT",
    "DRIVER'S LICENSE NUMBER": "DRIVER_LICENSE",
    "SOCIAL SECURITY NUMBER": "SSN",
    "SOCIAL_SECURITY_NUMBER": "SSN",
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
    "OCCUPATION": "OCCUPATION",
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
    "DEMOGRAPHIC": "DEMOGRAPHIC",
    "QUASI IDENTIFIER": "QUASI_ID",
    "QUASI_IDENTIFIER": "QUASI_ID",
    "COREF": "QUASI_ID",
    "ID": "TECH_ID",
    "API KEY": "TECH_ID",
    "API_KEY": "TECH_ID",
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
    "MONTANT": "AMOUNT",
    "MONEY": "AMOUNT",
    "TEAM": "MISC",
    "PROJECT": "MISC",
    "TOOL": "MISC",
    "MODEL": "MISC",
    "STYLE": "MISC",
}

PROFILE_TYPE_MAPPING: Dict[str, Dict[str, str]] = {
    # Projection vers le schéma CoNLL / news NER.
    "news_ner": {
        "PERSON": "PER",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOC",
        "GPE": "LOC",
        "FACILITY": "LOC",
        "EVENT": "MISC",
        "PRODUCT": "MISC",
        "LANGUAGE": "MISC",
        "LAW": "MISC",
        "WORK OF ART": "MISC",
        "WORK_OF_ART": "MISC",
        "NATIONALITY": "MISC",
        "RACE": "MISC",
        "RELIGION": "MISC",
        "IDEOLOGY": "MISC",
        "NORP": "MISC",
        "AWARD": "MISC",
    },
    "conll2003": {
        "PERSON": "PER",
        "ORGANIZATION": "ORG",
        "LOCATION": "LOC",
        "GPE": "LOC",
        "FACILITY": "LOC",
        "EVENT": "MISC",
        "PRODUCT": "MISC",
        "LANGUAGE": "MISC",
        "LAW": "MISC",
        "WORK OF ART": "MISC",
        "WORK_OF_ART": "MISC",
        "NATIONALITY": "MISC",
        "RACE": "MISC",
        "RELIGION": "MISC",
        "IDEOLOGY": "MISC",
        "NORP": "MISC",
        "AWARD": "MISC",
    },
}


def normalize_entity_profile(profile: Optional[str]) -> Optional[str]:
    if not profile:
        return None
    raw = str(profile).strip().lower()
    if raw in {"conll", "conll03", "cleanconll"}:
        return "conll2003"
    if raw in {"news", "news_ner", "news-ner"}:
        return "news_ner"
    if raw in {"pii", "privacy"}:
        return "pii"
    return raw


def normalize_entity_type(raw_type: Any, profile: Optional[str] = None) -> str:
    """
    Normalize an entity type to a standardized format.

    Args:
        raw_type: The raw entity type from any source

    Returns:
        Normalized uppercase entity type
    """
    if not isinstance(raw_type, str):
        normalized = str(raw_type).upper() if raw_type else "UNKNOWN"
        return project_entity_type(normalized, profile=profile)

    t = raw_type.strip().upper()
    normalized = ENTITY_TYPE_MAPPING.get(t, t)
    return project_entity_type(normalized, profile=profile)


def project_entity_type(raw_type: Any, profile: Optional[str] = None) -> str:
    """
    Project a normalized type into a profile-specific label space.

    Examples:
      - ``news_ner`` / ``conll2003`` → PER/ORG/LOC/MISC
      - ``pii`` / ``None``           → keep fine-grained labels
    """
    if not isinstance(raw_type, str):
        normalized = str(raw_type).upper() if raw_type else "UNKNOWN"
    else:
        normalized = raw_type.strip().upper() or "UNKNOWN"

    norm_profile = normalize_entity_profile(profile)
    if not norm_profile or norm_profile == "pii":
        return normalized

    mapping = PROFILE_TYPE_MAPPING.get(norm_profile)
    if not mapping:
        return normalized
    return mapping.get(normalized, normalized)


def normalize_entity(entity: Dict[str, Any], profile: Optional[str] = None) -> Dict[str, Any]:
    """
    Normalize an entity dict, standardizing the type field.

    Args:
        entity: Entity dict with 'type' or 'entity_group' field

    Returns:
        New dict with normalized type
    """
    normalized = dict(entity)

    # Get type from either field
    raw_type = entity.get("type") or entity.get("entity_group") or "UNKNOWN"
    normalized["type"] = normalize_entity_type(raw_type, profile=profile)

    # Ensure entity_group is also normalized if present
    if "entity_group" in normalized:
        normalized["entity_group"] = normalized["type"]

    return normalized
