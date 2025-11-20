"""
Generalization Service - Policy-driven data generalization

This module provides services for generalizing detected entities
according to policy settings (dates, organizations, etc.).
"""
import re
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass

from .policy import AnonymizationPolicy


@dataclass
class Generalization:
    """Represents a generalization transformation."""
    start: int
    end: int
    surface: str
    replacement: str
    etype: str
    policy_rule: str  # e.g., "month", "quarter", "year"


class GeneralizationService:
    """
    Service for applying policy-driven generalizations.
    
    Handles:
    - Date generalization (to month/quarter/year)
    - Organization placeholder generalization
    - Other policy-driven transformations
    """
    
    def __init__(self, policy: AnonymizationPolicy):
        """
        Initialize generalization service.
        
        Args:
            policy: AnonymizationPolicy defining generalization rules
        """
        self.policy = policy
    
    def generalize_dates(self, text: str) -> Tuple[str, List[Generalization]]:
        """
        Generalize dates according to policy granularity.
        
        Supports French and English date formats.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (generalized_text, list of generalizations)
        """
        generalizations: List[Generalization] = []
        
        if self.policy.date_granularity in {"none"}:
            return text, generalizations

        MONTHS = {
            # English
            "january": 1, "february": 2, "march": 3, "april": 4, "may": 5, "june": 6,
            "july": 7, "august": 8, "september": 9, "october": 10, "november": 11, "december": 12,
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "jun": 6, "jul": 7,
            "aug": 8, "sept": 9, "oct": 10, "nov": 11, "dec": 12,
            # French
            "janvier": 1, "février": 2, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
            "juillet": 7, "août": 8, "aout": 8, "septembre": 9, "octobre": 10, "novembre": 11,
            "décembre": 12, "decembre": 12,
            "janv": 1, "févr": 2, "fevr": 2, "avr": 4, "juil": 7,
            "sept": 9, "oct": 10, "nov": 11, "déc": 12, "dec": 12,
        }

        def month_to_num(name: str) -> int:
            return MONTHS.get(name.strip(". ").lower(), 0)

        def make_replacement(year: str, month: str) -> str:
            if self.policy.date_granularity in {"week", "month"}:
                return f"[DATE_{year}-{month}]"
            if self.policy.date_granularity == "quarter":
                q = (int(month) - 1) // 3 + 1
                return f"[DATE_{year}-Q{q}]"
            if self.policy.date_granularity == "year":
                return f"[DATE_{year}]"
            return "[DATE]"

        patterns = [
            re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b"),  # ISO format
            re.compile(r"\b(\d{1,2})\s+(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})\b", re.IGNORECASE),
            re.compile(r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),\s*(\d{4})\b", re.IGNORECASE),
            re.compile(r"\b(\d{1,2}|1er)\s+(janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|janv\.?,?|févr\.?,?|fevr\.?,?|avr\.?,?|juil\.?,?|sept\.?,?|oct\.?,?|nov\.?,?|déc\.?,?|dec\.?)\s+(\d{4})\b", re.IGNORECASE),
            re.compile(r"\b(\d{1,2})[\/.\-](\d{1,2})[\/.\-](\d{2,4})\b"),
        ]

        out = text
        delta = 0
        matches = []
        
        for pat in patterns:
            for m in pat.finditer(text):
                matches.append((pat, m.start(), m.end(), m))
        matches.sort(key=lambda x: x[1])

        for pat, s0, e0, m in matches:
            s = s0 + delta
            e = e0 + delta
            g = m.groups()
            y = None
            mo = None
            
            # Parse different date formats
            if len(g) == 3 and re.match(r"^\d{4}$", g[0] or ""):
                y, mo = g[0], g[1]
            elif len(g) == 3 and g[1] and g[1][0].isalpha() and (g[0].isdigit() or g[0].lower() == "1er"):
                y = g[2]
                mo = f"{month_to_num(g[1]):02d}"
            elif len(g) == 3 and g[0] and g[0][0].isalpha() and (g[1].isdigit() or g[1].lower() == "1er"):
                y = g[2]
                mo = f"{month_to_num(g[0]):02d}"
            elif len(g) == 3 and g[0].isdigit() and g[1].isdigit():
                d1, m2, y3 = int(g[0]), int(g[1]), g[2]
                if len(y3) == 2:
                    y3 = f"20{y3}" if int(y3) <= 29 else f"19{y3}"
                y, mo = y3, f"{m2:02d}"
            
            if not (y and mo and mo.isdigit() and 1 <= int(mo) <= 12):
                continue
            
            rep = make_replacement(y, mo)
            out = out[:s] + rep + out[e:]
            diff = len(rep) - (e - s)
            delta += diff
            
            generalizations.append(Generalization(
                start=s,
                end=e,
                surface=m.group(0),
                replacement=rep,
                etype="DATE",
                policy_rule=self.policy.date_granularity,
            ))
        
        return out, generalizations
    
    def generalize_org_placeholders(self, text: str) -> Tuple[str, List[Generalization]]:
        """
        Generalize organization placeholders according to policy.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (generalized_text, list of generalizations)
        """
        generalizations: List[Generalization] = []
        
        if self.policy.org_policy not in {"generalize", "redact"}:
            return text, generalizations

        def make_replacement(_m: re.Match) -> str:
            return "[ORG]" if self.policy.org_policy == "generalize" else "[REDACTED]"

        pattern = re.compile(r"\[ORG_[A-Z0-9]+\]")
        out = text
        
        for m in list(pattern.finditer(text)):
            s, e = m.start(), m.end()
            new = make_replacement(m)
            if new != m.group(0):
                out = out[:s] + new + out[e:]
                generalizations.append(Generalization(
                    start=s,
                    end=e,
                    surface=m.group(0),
                    replacement=new,
                    etype="ORG",
                    policy_rule=self.policy.org_policy,
                ))
        
        return out, generalizations
    
    def apply_all(self, text: str) -> Tuple[str, List[Generalization]]:
        """
        Apply all generalizations according to policy.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (generalized_text, list of all generalizations)
        """
        all_generalizations = []
        
        # Apply date generalization
        text, date_gens = self.generalize_dates(text)
        all_generalizations.extend(date_gens)
        
        # Apply org generalization
        text, org_gens = self.generalize_org_placeholders(text)
        all_generalizations.extend(org_gens)
        
        return text, all_generalizations


def escalate_policy(policy: AnonymizationPolicy) -> AnonymizationPolicy:
    """
    Create a new policy with escalated privacy settings.
    
    This is used for hardening when risk is too high.
    
    Args:
        policy: Current policy
        
    Returns:
        New policy with escalated settings
    """
    # Create a copy to avoid mutating the original
    from dataclasses import replace
    
    order_date = ["none", "week", "month", "quarter", "year", "redact"]
    try:
        i = order_date.index(policy.date_granularity)
        new_date_gran = order_date[i + 1] if i + 1 < len(order_date) else "redact"
    except Exception:
        new_date_gran = "month"

    order_ip = ["exact", "public_private", "cidr24", "redact"]
    try:
        i = order_ip.index(policy.ip_policy)
        new_ip_policy = order_ip[i + 1] if i + 1 < len(order_ip) else "redact"
    except Exception:
        new_ip_policy = "cidr24"

    order_org = ["replace", "categorize", "generalize", "redact"]
    try:
        i = order_org.index(policy.org_policy)
        new_org_policy = order_org[i + 1] if i + 1 < len(order_org) else "redact"
    except Exception:
        new_org_policy = "generalize"

    new_paraphrase_intensity = min(3, int(policy.paraphrase_intensity or 1) + 1)
    
    return replace(
        policy,
        date_granularity=new_date_gran,
        ip_policy=new_ip_policy,
        org_policy=new_org_policy,
        paraphrase_intensity=new_paraphrase_intensity,
    )


__all__ = [
    "Generalization",
    "GeneralizationService",
    "escalate_policy",
]
