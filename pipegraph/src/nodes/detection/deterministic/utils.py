import os
from typing import Set, Optional, Iterable

DEFAULT_WHITELIST = {
    "alteca",
    "cse",
    "cpf",
    "miaou",
    "alte",
    "kalidéa",
}

_CACHE: Optional[Set[str]] = None

def get_whitelist(extra_words: Optional[Iterable[str]] = None) -> Set[str]:
    global _CACHE
    if _CACHE is not None and not extra_words:
        return _CACHE

    wl = set(DEFAULT_WHITELIST)
    # Enrichissement via fichier externe si défini (variable d'env)
    ext_path = os.getenv("PII_WHITELIST_PATH", "").strip()
    if ext_path and os.path.exists(ext_path):
        try:
            with open(ext_path, "r", encoding="utf-8") as f:
                for line in f:
                    word = line.strip()
                    if word and not word.startswith("#"):
                        wl.add(word.lower())
        except Exception:
            pass

    if extra_words:
        wl.update(w.lower() for w in extra_words)

    if not extra_words:
        _CACHE = wl
    
    return wl

def is_whitelisted(word: str) -> bool:
    return word.lower().strip() in get_whitelist()

def clean_identifier(text: str) -> str:
    """Supprime les espaces et tirets pour la validation."""
    return text.replace(" ", "").replace("-", "")
