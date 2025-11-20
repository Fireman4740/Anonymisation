
"""
Détecteur basé sur Regex avec Validation

Ce module fournit des patterns regex pour détecter les PII et autres entités sensibles,
avec des validateurs pour s'assurer de la validité des détections.

Pattern Categories:
- PII: Email, Phone, NIR (SSN français), IBAN, BIC
- Technical: IP, URL, UUID, Path, MAC, etc.
- Financial: Montants avec devises
- Legal: Articles, lois, codes
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, List
import re
import geonamescache

# Import optionnel de schwifty (IBAN/BIC)
try:
    from schwifty import IBAN, BIC
    _SCHWIFTY_AVAILABLE = True
except Exception:
    _SCHWIFTY_AVAILABLE = False
    class IBAN:
        def __init__(self, *_a, **_k):
            self.spec = {"iban_length": 34}
            self.is_valid = False
    class BIC:
        def __init__(self, *_a, **_k):
            self.exists = False

# Import whitelist
try:
    from ...utils.whitelist import get_whitelist
except Exception:
    try:
        from utils.whitelist import get_whitelist
    except Exception:
        def get_whitelist():
            return set()


# ====
# Validator Strategy Pattern
# ====

class ValidatorStrategy(ABC):
    @abstractmethod
    def validate(self, match: re.Match, start: int, tag: str) -> Optional[Tuple[int, int, str]]:
        pass


def _clean_identifier(text: str) -> str:
    return text.replace(" ", "").replace("-", "")


def _span_to_n_alnum(text: str, length: int) -> int:
    count = 0
    for i, char in enumerate(text, 1):
        if char.isalnum():
            count += 1
            if count == length:
                return i
    return len(text)


class IbanValidator(ValidatorStrategy):
    def validate(self, match: re.Match, start: int, tag: str) -> Optional[Tuple[int, int, str]]:
        if not _SCHWIFTY_AVAILABLE:
            return None
        raw = match.group()
        candidate = _clean_identifier(raw)
        iban_length = IBAN(candidate, allow_invalid=True).spec["iban_length"]
        candidate = candidate[:iban_length]
        validated = IBAN(candidate, allow_invalid=True)
        end = start + _span_to_n_alnum(raw, iban_length)
        return (start, end, tag) if validated.is_valid else None


class BicValidator(ValidatorStrategy):
    def validate(self, match: re.Match, start: int, tag: str) -> Optional[Tuple[int, int, str]]:
        if not _SCHWIFTY_AVAILABLE:
            return None
        candidate = _clean_identifier(match.group())
        validated = BIC(candidate, allow_invalid=True)
        return (start, match.end(), tag) if validated.exists else None


class FrenchSSNValidator(ValidatorStrategy):
    def validate(self, match: re.Match, start: int, tag: str) -> Optional[Tuple[int, int, str]]:
        raw = match.group()
        candidate = _clean_identifier(raw)
        key = int(candidate[-2:])
        try:
            num = int(candidate[:-2])
        except ValueError:
            return None
        expected_key = 97 - (num % 97)
        if expected_key == key:
            end = start + _span_to_n_alnum(raw, 15)
            return (start, end, tag)
        return None


class NoValidator(ValidatorStrategy):
    def validate(self, match: re.Match, start: int, tag: str) -> Tuple[int, int, str]:
        return (start, match.end(), tag)


class LuhnValidator(ValidatorStrategy):
    """Generic Luhn validator for credit card numbers."""
    def validate(self, match: re.Match, start: int, tag: str) -> Optional[Tuple[int, int, str]]:
        digits = re.sub(r"\D", "", match.group())
        if len(digits) < 13:
            return None
        total = 0
        rev = digits[::-1]
        for i, d in enumerate(rev):
            n = int(d)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        return (start, match.end(), tag) if total % 10 == 0 else None


class GroupSpanValidator(ValidatorStrategy):
    """Valide et remplace uniquement le groupe capturé."""
    def __init__(self, group_index: int = 1):
        self.group_index = group_index

    def validate(self, match: re.Match, start: int, tag: str) -> Optional[Tuple[int, int, str]]:
        try:
            s = match.start(self.group_index)
            e = match.end(self.group_index)
            return (s, e, tag) if 0 <= s < e else None
        except IndexError:
            return None


# ====
# Patterns de Détection
# ====

PII_PATTERNS = [
    # Email
    (
        re.compile(r"(?<![\w@])[a-zA-Z0-9._%+-]+@(?:[a-zA-Z0-9-]+\.)+[a-zA-Z]{2,}", re.VERBOSE),
        "EMAIL",
        NoValidator(),
    ),
    # FR phone
    (
        re.compile(r"(?<!\w)(?:\+\d{1,3}|0{1,2}\s?-?\d{1,3}|0)[\s.-]?\d(?:[\s.-]?\d{2}){4}(?!\w)", re.IGNORECASE),
        "TELEPHONE",
        NoValidator(),
    ),
    # NIR (Sécurité Sociale FR)
    (
        re.compile(r"\b[12](?:[ -]?)[0-9]{2}(?:[ -]?)(0[1-9]|1[0-2])(?:[ -]?)(2[AB]|[0-9]{2})(?:[ -]?)[0-9]{3}(?:[ -]?)[0-9]{3}(?:[ -]?)([0-9]{2})\b", re.IGNORECASE),
        "NIR",
        FrenchSSNValidator(),
    ),
]

# Dates (FR/EN)
DATE_PATTERNS = [
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b"), "DATE", NoValidator()),
    (re.compile(r"\b\d{1,2}\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE), "DATE", NoValidator()),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s*\d{4}\b", re.IGNORECASE), "DATE", NoValidator()),
    (re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b", re.IGNORECASE), "DATE", NoValidator()),
    (re.compile(r"\b(\d{1,2}|1er)\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|janv\.?,?|févr\.?,?|fevr\.?,?|avr\.?,?|juil\.?,?|sept\.?,?|oct\.?,?|nov\.?,?|déc\.?,?|dec\.?)\s+\d{4}\b", re.IGNORECASE), "DATE", NoValidator()),
    (re.compile(r"\b(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|janv\.?|févr\.?|fevr\.?|avr\.?|juil\.?|sept\.?|oct\.?|nov\.?|déc\.?|dec\.?)\s+\d{4}\b", re.IGNORECASE), "DATE", NoValidator()),
    (re.compile(r"\b\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}\b"), "DATE", NoValidator()),
]

if _SCHWIFTY_AVAILABLE:
    PII_PATTERNS.extend([
        (re.compile(r"\b([A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]){11,30})\b", re.IGNORECASE), "IBAN", IbanValidator()),
        (re.compile(r"\b[A-Z]{4}[A-Z]{2}[A-Z0-9]{2}(?:[A-Z0-9]{3})?\b", re.IGNORECASE), "BIC", BicValidator()),
    ])

OTHER_PATTERNS = [
    # IPv4
    (re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)"), "IP", NoValidator()),
    # IPv6
    (re.compile(r"\b[0-9a-fA-F:]{3,}:[0-9a-fA-F:]{2,}\b"), "IP", NoValidator()),
    # URL/URI
    (re.compile(r"\bhttps?://[\w\-._~:/?#\[\]@!$&'()*+,;=%]+", re.IGNORECASE), "URL", NoValidator()),
    # Hostname/FQDN
    (re.compile(r"\b(?:[a-zA-Z0-9-]{1,63}\.)+[a-zA-Z]{2,63}\b"), "HOST", NoValidator()),
    # Unix path
    (re.compile(r"(?<!\w)/(?!/)[^\s]+"), "PATH", NoValidator()),
    # Windows path
    (re.compile(r"\b[A-Za-z]:\\(?:[^\\\s]+\\)*[^\\\s]*"), "PATH", NoValidator()),
    # UUID
    (re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"), "UUID", NoValidator()),
    # Carte bancaire (Luhn)
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "CARD", LuhnValidator()),
    # AWS Access Key
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "AWS_KEY", NoValidator()),
    # Secret générique
    (re.compile(r"\bsk-[A-Za-z0-9]{16,64}\b"), "SECRET", NoValidator()),
    # MAC address
    (re.compile(r"\b[0-9A-Fa-f]{2}(:[0-9A-Fa-f]{2}){5}\b"), "MAC", NoValidator()),
    # Ticket (JIRA)
    (re.compile(r"\b[A-Z][A-Z0-9]{1,9}-\d+\b"), "TICKET", NoValidator()),
    # Username
    (re.compile(r"\b(?:user|username|login|utilisateur)\s*[:=]\s*([A-Za-z0-9._-]{3,64})", re.IGNORECASE), "USERNAME", GroupSpanValidator(1)),
    # Montants avec devises
    (re.compile(r"\b(?:EUR|USD|SEK|GBP|CHF|CAD|AUD|JPY|CNY)\s+\d{1,3}(?:[.,\s\u202f]\d{3})*(?:[.,]\d{2})?\b"), "MONTANT", NoValidator()),
    (re.compile(r"(?<![\w<])(?:€|\$|£)\s?\d{1,3}(?:[.,\s\u202f]\d{3})*(?:[.,]\d{2})?\b"), "MONTANT", NoValidator()),
    (re.compile(r"\b\d{1,3}(?:[.,\s\u202f]\d{3})*(?:[.,]\d{2})?\s?(?:€|euros?|EUR|USD|dollars?|SEK|GBP|pounds?|livres|CHF|CAD|AUD|JPY|CNY)\b", re.IGNORECASE), "MONTANT", NoValidator()),
    # Codes légaux
    (re.compile(r"\b(?:application\s+)?no\.?\s*\d{2,8}/\d{2,4}\b", re.IGNORECASE), "CODE", NoValidator()),
    (re.compile(r"\b(?:Article|Art\.?)\s+\d+[A-Za-z]?\b", re.IGNORECASE), "CODE", NoValidator()),
    (re.compile(r"\b(?:Law|Decree|Act)\s+no\.?\s*\d+[A-Za-z]?\b", re.IGNORECASE), "CODE", NoValidator()),
    (re.compile(r"\bRule\s+\d+\s*§\s*\d+\b", re.IGNORECASE), "CODE", NoValidator()),
    (re.compile(r"\bProtocol\s+No\.?\s*\d+\b", re.IGNORECASE), "CODE", NoValidator()),
]


def regexes_based_replacements(text: str) -> List[Tuple[int, int, str]]:
    """
    Détecte les entités PII en utilisant les patterns regex.
    
    Args:
        text: Texte à analyser
        
    Returns:
        Liste de tuples (start, end, tag) pour chaque entité détectée
    """
    results: List[Tuple[int, int, str]] = []
    all_patterns = PII_PATTERNS + DATE_PATTERNS + OTHER_PATTERNS
    
    for pattern, tag, validator in all_patterns:
        for match in pattern.finditer(text):
            start = match.start()
            try:
                result = validator.validate(match, start, tag)
                if result:
                    results.append(result)
            except Exception:
                continue
    
    return results


__all__ = ["regexes_based_replacements", "PII_PATTERNS", "DATE_PATTERNS", "OTHER_PATTERNS"]
