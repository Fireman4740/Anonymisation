"""
PUMS population loader for RAT-Bench re-identification risk calculation.

Downloads and caches the US Census PUMS 1-Year microdata, extracts relevant
columns, and maps PUMS numeric codes to the human-readable values used by
RAT-Bench profiles.

Setup:
    Option A (recommended): Run `python eval/pums_loader.py` to download
        automatically from the US Census Bureau FTP (~620 MB download,
        ~50 MB cached parquet).
    Option B (manual): Place the PUMS CSV files (psam_pusa.csv, psam_pusb.csv)
        in eval/datasets/RAT-Bench/cache/pums_raw/ and run the loader.

Reference population: ~3.2 M person records (ACS 2022 1-Year).
US Census total N ≈ 309,349,689.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional

import pandas as pd

logger = logging.getLogger("PUMS-Loader")

# ---------------------------------------------------------------------------
# PUMS column → profile key mapping
# ---------------------------------------------------------------------------

PUMS_COL_TO_PROFILE_KEY: Dict[str, str] = {
    "SEX": "sex",
    "RAC2P": "race",
    "CIT": "citizenship status",
    "SCHL": "educational attainment",
    "ESR": "employment status",
    "ST": "state of residence",
    "MAR": "marital status",
    "OCCP": "occupation",
}

PROFILE_KEY_TO_PUMS_COL: Dict[str, str] = {v: k for k, v in PUMS_COL_TO_PROFILE_KEY.items()}

PUMS_COLUMNS_NEEDED: List[str] = ["SERIALNO", "PWGTP"] + list(PUMS_COL_TO_PROFILE_KEY.keys())

US_POPULATION_N: int = 309_349_689

# ---------------------------------------------------------------------------
# Code → human-readable mappings (from ACS 2022 PUMS Data Dictionary)
# ---------------------------------------------------------------------------

SEX_MAP: Dict[int, str] = {
    1: "Male",
    2: "Female",
}

CIT_MAP: Dict[int, str] = {
    1: "Born in the US",
    2: "Born in Puerto Rico, Guam, the U.S. Virgin Islands, or the Northern Marianas Islands",
    3: "Born abroad of American parent(s)",
    4: "U.S. citizen by naturalization",
    5: "Not a citizen of the U.S.",
}

MAR_MAP: Dict[int, str] = {
    1: "Married",
    2: "Widowed",
    3: "Divorced",
    4: "Separated",
    5: "Never married or under 15 years old",
}

ESR_MAP: Dict[int, str] = {
    1: "Civilian employed, at work",
    2: "Civilian employed, with a job but not at work",
    3: "Unemployed",
    4: "Armed forces, at work",
    5: "Armed forces, with a job but not at work",
    6: "Not in labor force",
}

SCHL_MAP: Dict[int, str] = {
    1: "No schooling completed",
    2: "Nursery school, preschool",
    3: "Kindergarten",
    4: "Grade 1",
    5: "Grade 2",
    6: "Grade 3",
    7: "Grade 4",
    8: "Grade 5",
    9: "Grade 6",
    10: "Grade 7",
    11: "Grade 8",
    12: "Grade 9",
    13: "Grade 10",
    14: "Grade 11",
    15: "12th grade - no diploma",
    16: "Regular high school diploma",
    17: "GED or alternative credential",
    18: "Some college, but less than 1 year",
    19: "1 or more years of college credit, no degree",
    20: "Associate's degree",
    21: "Bachelor's degree",
    22: "Master's degree",
    23: "Professional degree beyond a bachelor's degree",
    24: "Doctorate degree",
}

# State FIPS → "Full Name/Abbreviation"  (RAT-Bench profile format)
_STATE_DATA: Dict[int, tuple] = {
    1: ("Alabama", "AL"), 2: ("Alaska", "AK"), 4: ("Arizona", "AZ"),
    5: ("Arkansas", "AR"), 6: ("California", "CA"), 8: ("Colorado", "CO"),
    9: ("Connecticut", "CT"), 10: ("Delaware", "DE"),
    11: ("District of Columbia", "DC"), 12: ("Florida", "FL"),
    13: ("Georgia", "GA"), 15: ("Hawaii", "HI"), 16: ("Idaho", "ID"),
    17: ("Illinois", "IL"), 18: ("Indiana", "IN"), 19: ("Iowa", "IA"),
    20: ("Kansas", "KS"), 21: ("Kentucky", "KY"), 22: ("Louisiana", "LA"),
    23: ("Maine", "ME"), 24: ("Maryland", "MD"), 25: ("Massachusetts", "MA"),
    26: ("Michigan", "MI"), 27: ("Minnesota", "MN"), 28: ("Mississippi", "MS"),
    29: ("Missouri", "MO"), 30: ("Montana", "MT"), 31: ("Nebraska", "NE"),
    32: ("Nevada", "NV"), 33: ("New Hampshire", "NH"), 34: ("New Jersey", "NJ"),
    35: ("New Mexico", "NM"), 36: ("New York", "NY"),
    37: ("North Carolina", "NC"), 38: ("North Dakota", "ND"),
    39: ("Ohio", "OH"), 40: ("Oklahoma", "OK"), 41: ("Oregon", "OR"),
    42: ("Pennsylvania", "PA"), 44: ("Rhode Island", "RI"),
    45: ("South Carolina", "SC"), 46: ("South Dakota", "SD"),
    47: ("Tennessee", "TN"), 48: ("Texas", "TX"), 49: ("Utah", "UT"),
    50: ("Vermont", "VT"), 51: ("Virginia", "VA"), 53: ("Washington", "WA"),
    54: ("West Virginia", "WV"), 55: ("Wisconsin", "WI"),
    56: ("Wyoming", "WY"), 72: ("Puerto Rico", "PR"),
}
ST_MAP: Dict[int, str] = {k: f"{v[0]}/{v[1]}" for k, v in _STATE_DATA.items()}

# RAC2P — Detailed race (2-digit recoded).  Full ACS 2022 mapping.
RAC2P_MAP: Dict[int, str] = {
    1: "White alone",
    2: "Black or African American alone",
    3: "American Indian alone",
    4: "Alaska Native alone",
    5: "American Indian and Alaska Native tribes specified; or American Indian or Alaska Native, not specified and no other races",
    6: "Asian alone",
    7: "Native Hawaiian and Other Pacific Islander alone",
    8: "Some Other Race alone",
    9: "Two or More Races",
}

# OCCP — Occupation codes.  We load a full mapping from a CSV if available,
# otherwise provide a reverse-lookup helper that uses the RAT-Bench dataset
# itself to build the map.
_OCCP_MAP: Optional[Dict[int, str]] = None


def _occp_map() -> Dict[int, str]:
    """Lazy-load OCCP code→description mapping."""
    global _OCCP_MAP
    if _OCCP_MAP is not None:
        return _OCCP_MAP

    cache = _cache_dir()
    occp_path = os.path.join(cache, "occp_map.csv")
    if os.path.exists(occp_path):
        df = pd.read_csv(occp_path)
        _OCCP_MAP = dict(zip(df["code"], df["description"]))
        return _OCCP_MAP

    # Fallback: build from the RAT-Bench profiles (limited but workable)
    _OCCP_MAP = _build_occp_from_ratbench()
    return _OCCP_MAP


def _build_occp_from_ratbench() -> Dict[int, str]:
    """Build a partial OCCP map from the RAT-Bench dataset profiles."""
    import json

    cache = _cache_dir()
    rb_path = os.path.join(cache, "ratbench_english.json")
    mapping: Dict[int, str] = {}

    if not os.path.exists(rb_path):
        return mapping

    with open(rb_path, "r", encoding="utf-8") as f:
        profiles = json.load(f)

    for p in profiles:
        occ_val = (p.get("profile") or {}).get("occupation")
        occ_code = (p.get("indirect_identifiers") or {}).get("OCCP")
        if occ_val and occ_code is not None:
            try:
                code_int = int(occ_code)
                mapping[code_int] = str(occ_val)
            except (ValueError, TypeError):
                # OCCP might be a string description already
                pass

    logger.info(f"Built partial OCCP map from RAT-Bench data: {len(mapping)} entries")
    return mapping


# ---------------------------------------------------------------------------
# Reverse mappings  (human-readable → PUMS code)
# ---------------------------------------------------------------------------

def _build_reverse_map(forward: Dict[int, str]) -> Dict[str, int]:
    return {v.strip().lower(): k for k, v in forward.items()}


_REVERSE_SEX = _build_reverse_map(SEX_MAP)
_REVERSE_CIT = _build_reverse_map(CIT_MAP)
_REVERSE_MAR = _build_reverse_map(MAR_MAP)
_REVERSE_ESR = _build_reverse_map(ESR_MAP)
_REVERSE_SCHL = _build_reverse_map(SCHL_MAP)
_REVERSE_RAC2P = _build_reverse_map(RAC2P_MAP)

# State: allow match on "Oregon/OR", "Oregon", or "OR"
_REVERSE_ST: Dict[str, int] = {}
for _fips, (_name, _abbr) in _STATE_DATA.items():
    _REVERSE_ST[f"{_name}/{_abbr}".lower()] = _fips
    _REVERSE_ST[_name.lower()] = _fips
    _REVERSE_ST[_abbr.lower()] = _fips


def value_to_pums_code(pums_col: str, human_value: str) -> Optional[int]:
    """Convert a human-readable attribute value to its PUMS numeric code."""
    v = human_value.strip().lower()
    if not v or v in ("none", "n/a", "null"):
        return None

    lookup: Dict[str, int] = {
        "SEX": _REVERSE_SEX,
        "CIT": _REVERSE_CIT,
        "MAR": _REVERSE_MAR,
        "ESR": _REVERSE_ESR,
        "SCHL": _REVERSE_SCHL,
        "ST": _REVERSE_ST,
        "RAC2P": _REVERSE_RAC2P,
    }.get(pums_col, {})

    code = lookup.get(v)
    if code is not None:
        return code

    # For OCCP, try reverse lookup from description
    if pums_col == "OCCP":
        occp = _occp_map()
        rev = {desc.strip().lower(): c for c, desc in occp.items()}
        return rev.get(v)

    return None


# ---------------------------------------------------------------------------
# Cache directory
# ---------------------------------------------------------------------------

def _cache_dir() -> str:
    eval_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(eval_dir, "datasets", "RAT-Bench", "cache")


# ---------------------------------------------------------------------------
# Download & process PUMS data
# ---------------------------------------------------------------------------

_CENSUS_FTP_BASE = "https://www2.census.gov/programs-surveys/acs/data/pums"
_PUMS_YEAR = 2022
_PUMS_HORIZON = "1-Year"


def download_pums_csv(cache_dir: Optional[str] = None, year: int = _PUMS_YEAR) -> str:
    """
    Download combined PUMS person CSV from the US Census Bureau FTP.

    Returns the path to the downloaded (and extracted) CSV directory.
    """
    import io
    import urllib.request
    import zipfile

    cache = cache_dir or _cache_dir()
    raw_dir = os.path.join(cache, "pums_raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Check if already downloaded
    existing = [f for f in os.listdir(raw_dir) if f.startswith("psam_p") and f.endswith(".csv")]
    if existing:
        logger.info(f"PUMS CSV already downloaded: {raw_dir} ({len(existing)} files)")
        return raw_dir

    url = f"{_CENSUS_FTP_BASE}/{year}/{_PUMS_HORIZON}/csv_pus.zip"
    logger.info(f"Downloading PUMS data from {url} ... (this may take several minutes)")
    print(f"📥 Téléchargement des données PUMS US Census ({year}) ...")
    print(f"   URL: {url}")
    print(f"   ⚠️  Taille approximative: ~620 MB — patience requise.")

    req = urllib.request.Request(url, headers={"User-Agent": "Anonymisation-Eval/1.0"})
    with urllib.request.urlopen(req, timeout=600) as resp:
        data = resp.read()

    logger.info(f"Downloaded {len(data) / 1e6:.0f} MB — extracting...")
    print(f"   ✅ Téléchargé ({len(data) / 1e6:.0f} MB). Extraction...")

    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        for member in zf.namelist():
            if member.startswith("psam_p") and member.endswith(".csv"):
                zf.extract(member, raw_dir)
                logger.info(f"Extracted: {member}")

    return raw_dir


def process_pums_to_parquet(
    raw_dir: Optional[str] = None,
    cache_dir: Optional[str] = None,
) -> str:
    """
    Read raw PUMS CSV(s), extract relevant columns, map codes to
    human-readable values, and save as a compact parquet file.

    Returns the path to the cached parquet file.
    """
    cache = cache_dir or _cache_dir()
    parquet_path = os.path.join(cache, "pums_population.parquet")

    if os.path.exists(parquet_path):
        logger.info(f"PUMS parquet already exists: {parquet_path}")
        return parquet_path

    raw = raw_dir or os.path.join(cache, "pums_raw")
    csv_files = sorted(
        os.path.join(raw, f)
        for f in os.listdir(raw)
        if f.startswith("psam_p") and f.endswith(".csv")
    )
    if not csv_files:
        raise FileNotFoundError(
            f"No PUMS CSV files found in {raw}. "
            f"Run `python eval/pums_loader.py` to download, or place "
            f"psam_pusa.csv and psam_pusb.csv in {raw}."
        )

    print(f"📊 Traitement des fichiers PUMS : {[os.path.basename(f) for f in csv_files]}")

    frames: List[pd.DataFrame] = []
    for csv_path in csv_files:
        logger.info(f"Reading {csv_path} ...")
        # Read only needed columns (saves memory)
        available_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()
        use_cols = [c for c in PUMS_COLUMNS_NEEDED if c in available_cols]
        df = pd.read_csv(csv_path, usecols=use_cols, low_memory=False)
        frames.append(df)

    population = pd.concat(frames, ignore_index=True)
    logger.info(f"Raw PUMS records: {len(population):,}")

    # Map codes to human-readable values
    column_maps: Dict[str, Dict[int, str]] = {
        "SEX": SEX_MAP,
        "CIT": CIT_MAP,
        "MAR": MAR_MAP,
        "ESR": ESR_MAP,
        "SCHL": SCHL_MAP,
        "ST": ST_MAP,
        "RAC2P": RAC2P_MAP,
    }

    for col, mapping in column_maps.items():
        if col in population.columns:
            population[col] = (
                population[col]
                .apply(lambda x: mapping.get(int(x), None) if pd.notna(x) else None)
            )

    # OCCP: keep as-is (numeric) — we'll use the description map for matching
    occp = _occp_map()
    if "OCCP" in population.columns and occp:
        population["OCCP"] = (
            population["OCCP"]
            .apply(lambda x: occp.get(int(x), None) if pd.notna(x) else None)
        )

    # Rename columns to match profile keys
    population = population.rename(columns=PUMS_COL_TO_PROFILE_KEY)

    # Save
    os.makedirs(cache, exist_ok=True)
    population.to_parquet(parquet_path, index=False)
    print(f"   ✅ Population PUMS sauvegardée: {parquet_path} ({len(population):,} enregistrements)")
    logger.info(f"Saved PUMS population: {parquet_path} ({len(population):,} rows)")

    return parquet_path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def load_pums_population(cache_dir: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Load the PUMS population DataFrame.

    Returns None if PUMS data is not available (with a warning).
    Columns use human-readable profile key names.
    """
    cache = cache_dir or _cache_dir()
    parquet_path = os.path.join(cache, "pums_population.parquet")

    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded PUMS population: {len(df):,} records from {parquet_path}")
        return df

    # Try to process from raw CSV if available
    raw_dir = os.path.join(cache, "pums_raw")
    csv_exists = os.path.isdir(raw_dir) and any(
        f.startswith("psam_p") and f.endswith(".csv")
        for f in os.listdir(raw_dir)
    )
    if csv_exists:
        try:
            process_pums_to_parquet(raw_dir, cache)
            return pd.read_parquet(parquet_path)
        except Exception as e:
            logger.warning(f"Failed to process PUMS CSV: {e}")

    logger.warning(
        "⚠️  PUMS population data NOT available. Risk calculation will use "
        "the RAT-Bench dataset profiles as a VERY SMALL proxy population "
        f"(~300 instead of ~3.2M). Results will be UNRELIABLE.\n"
        f"Run `python eval/pums_loader.py` to download PUMS data."
    )
    return None


# ---------------------------------------------------------------------------
# CLI: download + process
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print("=" * 60)
    print("PUMS Population Data — Download & Process")
    print("=" * 60)

    cache = _cache_dir()
    raw_dir = download_pums_csv(cache_dir=cache)
    parquet_path = process_pums_to_parquet(raw_dir, cache)

    df = pd.read_parquet(parquet_path)
    print(f"\n📊 Population: {len(df):,} enregistrements")
    print(f"   Colonnes: {list(df.columns)}")
    for col in df.columns:
        if col not in ("SERIALNO", "PWGTP"):
            n_unique = df[col].nunique()
            n_null = df[col].isna().sum()
            print(f"   {col}: {n_unique} valeurs uniques, {n_null} null")
    print("\n✅ Terminé!")