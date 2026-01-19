"""
Entity type normalization utilities.
Centralizes label mapping between different detection sources.
"""

from typing import Any, Dict

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
    "LIEU": "LOC",
    "VILLE": "LOC",
    "CITY": "LOC",
    "COUNTRY": "LOC",
    "PAYS": "LOC",
    "ADDRESS": "ADDR",
    "ADRESSE": "ADDR",
    # Date/Time
    "DATE OF BIRTH": "DOB",
    "BIRTHDATE": "DOB",
    "DATE DE NAISSANCE": "DOB",
    # Financial
    "CREDIT CARD NUMBER": "CREDIT_CARD",
    "BANK ACCOUNT NUMBER": "BANK_ACCOUNT",
    "CREDIT CARD": "CREDIT_CARD",
    # ID documents
    "NATIONAL ID NUMBER": "NATIONAL_ID",
    "IDENTITY CARD NUMBER": "NATIONAL_ID",
    "PASSPORT NUMBER": "PASSPORT",
    "DRIVER'S LICENSE NUMBER": "DRIVER_LICENSE",
    "SOCIAL SECURITY NUMBER": "SSN",
}


def normalize_entity_type(raw_type: Any) -> str:
    """
    Normalize an entity type to a standardized format.

    Args:
        raw_type: The raw entity type from any source

    Returns:
        Normalized uppercase entity type
    """
    if not isinstance(raw_type, str):
        return str(raw_type).upper() if raw_type else "UNKNOWN"

    t = raw_type.strip().upper()
    return ENTITY_TYPE_MAPPING.get(t, t)


def normalize_entity(entity: Dict[str, Any]) -> Dict[str, Any]:
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
    normalized["type"] = normalize_entity_type(raw_type)

    # Ensure entity_group is also normalized if present
    if "entity_group" in normalized:
        normalized["entity_group"] = normalized["type"]

    return normalized
