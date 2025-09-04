# whitelist_words.py
import os
import json
from typing import Iterable, Set, Optional

# Whitelist par défaut (mettez ici vos termes tolérés: acronymes internes, etc.)
DEFAULT_WHITELIST = {
    "alteca",
    "cse",
    "cpf",
    "miaou",
    "alte",
    "kalidéa",
}

_CACHE: Optional[Set[str]] = None


def _load_from_file(path: str) -> Set[str]:
    path = path.strip()
    if not path or not os.path.exists(path):
        return set()

    # JSON (array) => ["mot1","mot2",...]
    if path.lower().endswith(".json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return {str(x).strip().lower() for x in data if str(x).strip()}
        # clé "words" si c'est un objet
        if isinstance(data, dict) and "words" in data and isinstance(data["words"], list):
            return {str(x).strip().lower() for x in data["words"] if str(x).strip()}
        return set()

    # Texte: un mot par ligne, lignes vides/# ignorées
    words: Set[str] = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            words.add(s.lower())
    return words


def get_whitelist(extra_words: Optional[Iterable[str]] = None) -> Set[str]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    wl = set(DEFAULT_WHITELIST)
    # Enrichissement via fichier externe si défini
    ext_path = os.getenv("PII_WHITELIST_PATH", "").strip()
    if ext_path:
        wl |= _load_from_file(ext_path)

    if extra_words:
        wl |= {str(w).strip().lower() for w in extra_words if str(w).strip()}

    _CACHE = wl
    return _CACHE


def reload_whitelist() -> None:
    global _CACHE
    _CACHE = None
    _ = get_whitelist()


def is_whitelisted(word: str) -> bool:
    return word.lower().strip() in get_whitelist()