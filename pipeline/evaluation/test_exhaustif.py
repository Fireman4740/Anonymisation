"""
Script d'évaluation exhaustive du pipeline d'anonymisation.

Ce script charge désormais un dataset JSON décrivant les cas de test, afin de
faciliter l'ajout de nouveaux jeux de données ou la mise à jour des paramètres.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# Ajouter le dossier src au path (legacy support si nécessaire)
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

EVALUATION_DIR = PROJECT_ROOT / "evaluation"
DEFAULT_DATASET_PATH = EVALUATION_DIR / "datasets" / "default_cases.json"
DEFAULT_REPORT_PATH = EVALUATION_DIR / "reports" / "test_results.json"

DEFAULT_API_URL = os.getenv("PIPELINE_API_URL", "http://localhost:8000")
DEFAULT_API_TIMEOUT = float(os.getenv("PIPELINE_API_TIMEOUT", "120"))


EMBEDDED_DATASET = {
    "metadata": {
        "name": "Embedded demo dataset",
        "description": "Cas de test historiques embarqués dans le script",
        "case_count": 12,
    },
    "defaults": {
        "scope_prefix": "test",
        "secret_salt": "test-secret-123",
        "level": "L0",
    },
    "global_overrides": {},
    "cases": [
        {
            "id": "case_1_regex_pii",
            "description": "Test détection regex PII (email, téléphone, NIR)",
            "text": "Jean Dupont (jean.dupont@example.com) a appelé le 06 12 34 56 78. NIR: 1 89 05 75 123 456 78",
            "level": "L0",
            "expected_patterns": ["[MAIL_", "[TELEPHONE_", "[NIR_"],
        },
        {
            "id": "case_2_iban_bic",
            "description": "Test détection IBAN et BIC (nécessite schwifty)",
            "text": "Virement vers IBAN FR76 1234 5678 9012 3456 7890 123, BIC BNPAFRPP",
            "level": "L0",
            "expected_patterns": ["[IBAN_", "[BIC_"],
        },
        {
            "id": "case_3_dates_fr",
            "description": "Test détection dates françaises",
            "text": "Né le 13 septembre 1988, démission en janvier 2020, retraite prévue pour 2050-06-15.",
            "level": "L0",
            "expected_patterns": ["[DATE_"],
        },
        {
            "id": "case_4_dates_en",
            "description": "Test détection dates anglaises",
            "text": "Born on September 13, 1988. Retired in December 2020.",
            "level": "L0",
            "expected_patterns": ["[DATE_"],
        },
        {
            "id": "case_5_tech_patterns",
            "description": "Test patterns techniques (IP, URL, UUID, PATH)",
            "text": "Serveur 192.168.1.100 accessible via https://example.com/api. UUID: 550e8400-e29b-41d4-a716-446655440000. Logs dans /var/log/app.log",
            "level": "L0",
            "expected_patterns": ["[IP_", "[URL_", "[UUID_", "[PATH_"],
        },
        {
            "id": "case_6_financial",
            "description": "Test montants et carte bancaire",
            "text": "Facture de 15,800 euros payée par carte 4532 1234 5678 9010.",
            "level": "L0",
            "expected_patterns": ["[MONTANT_", "[CARD_"],
        },
        {
            "id": "case_7_secrets",
            "description": "Test détection secrets (AWS, API keys)",
            "text": "AWS_ACCESS_KEY=AKIAIOSFODNN7EXAMPLE, API_KEY=sk-1234567890abcdefghijklmnopqrstuvwxyz",
            "level": "L0",
            "expected_patterns": ["[AWS_KEY_", "[SECRET_"],
        },
        {
            "id": "case_8_mixed_complex",
            "description": "Test cas complexe avec multiples entités",
            "text": "Marie Curie (marie@curie.fr, +33 6 12 34 56 78) a travaillé à l'Institut Radium, 1 rue Pierre et Marie Curie, 75005 Paris. NIR: 2 89 05 75 987 654 32. Salaire: €85,000/an. Compte IBAN: FR14 2004 1010 0505 0001 3M02 606",
            "level": "L0",
            "expected_patterns": ["[MAIL_", "[TELEPHONE_", "[NIR_", "[IBAN_"],
            "expected_count": {"min_entities": 5},
        },
        {
            "id": "case_9_generalization_dates",
            "description": "Test généralisation des dates (policy date_granularity=month)",
            "text": "Contrat signé le 2024-06-15, début prévu pour septembre 2024.",
            "level": "L0",
            "overrides": {"date_granularity": "month"},
            "expected_patterns": ["[DATE_"],
            "expected_pattern_counts": {"[DATE_": 2},
        },
        {
            "id": "case_10_llm_level",
            "description": "Test niveau L1 (LLM activé) - paraphrase + audit",
            "text": "Alice travaille chez TechCorp depuis janvier 2020.",
            "level": "L1",
            "overrides": {
                "llm_paraphrase": True,
                "llm_audit": True,
                "rupta_enabled": False,
            },
            "expected_audit_keys": ["paraphrase_applied"],
        },
        {
            "id": "case_11_skip_tags",
            "description": "Test exclusion de tags regex (skip EMAIL)",
            "text": "Contact: bob@example.com, tel: 06 98 76 54 32",
            "level": "L0",
            "overrides": {"skip_regex_tags": ["MAIL"]},
            "forbidden_patterns": ["[MAIL_", "bob@example.com"],
        },
        {
            "id": "case_12_scope_consistency",
            "description": "Test cohérence des placeholders dans le même scope",
            "text": "Jean Dupont (jean.dupont@example.com) et Jean Dupont ont signé.",
            "level": "L0",
            "verify_placeholder_consistency": True,
        },
    ],
}


def parse_global_overrides_arg(value: Optional[str]) -> Dict[str, Any]:
    if not value:
        return {}
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        candidate = Path(value)
        if candidate.exists():
            with open(candidate, "r", encoding="utf-8") as f:
                return json.load(f)
        raise ValueError(f"Impossible d'interpréter --global-overrides: {value}")


def load_dataset_config(path: Optional[Path]) -> tuple[Dict[str, Any], str]:
    if path and path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f), str(path)
    if path:
        print(f"[WARN] Dataset '{path}' introuvable, utilisation du dataset embarqué.")
    return EMBEDDED_DATASET, "embedded"


def ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Évalue le pipeline d'anonymisation sur un dataset JSON.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH,
                        help="Chemin du dataset JSON à charger.")
    parser.add_argument("--output", type=Path, default=DEFAULT_REPORT_PATH,
                        help="Chemin du fichier de rapport JSON.")
    parser.add_argument("--scope-prefix", dest="scope_prefix", default=None,
                        help="Préfixe utilisé pour les scope_id (par défaut: dataset.defaults.scope_prefix).")
    parser.add_argument("--secret-salt", dest="secret_salt", default=None,
                        help="Sel secret utilisé pour les placeholders (défaut: dataset.defaults.secret_salt).")
    parser.add_argument("--default-level", dest="default_level", default=None,
                        help="Niveau appliqué aux cas sans level explicite.")
    parser.add_argument("--global-overrides", dest="global_overrides", default=None,
                        help="JSON ou chemin vers un fichier JSON d'overrides applicables à tous les cas.")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Liste d'identifiants de cas à exécuter (filtre).")
    parser.add_argument("--fail-fast", action="store_true",
                        help="Arrête l'exécution dès qu'un test échoue.")
    parser.add_argument("--api-url", dest="api_url", default=None,
                        help="URL de base de l'API FastAPI (défaut: PIPELINE_API_URL ou http://localhost:8000).")
    parser.add_argument("--api-timeout", dest="api_timeout", type=float, default=None,
                        help="Timeout de requête API en secondes (défaut: PIPELINE_API_TIMEOUT ou 30s).")
    return parser.parse_args()


# ============================================================================
# APPEL API
# ============================================================================


def call_api_anonymize(
    *,
    api_url: str,
    timeout: float,
    text: str,
    scope_id: str,
    level: str,
    secret_salt: str,
    overrides: Dict[str, Any],
    ner_results: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    endpoint = f"{api_url.rstrip('/')}/anonymize"
    payload = {
        "text": text,
        "scope_id": scope_id,
        "level": level,
        "secret_salt": secret_salt,
        "overrides": overrides or {},
        "ner_results": ner_results,
    }
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
    except requests.RequestException as exc:  # pragma: no cover - réseau
        raise RuntimeError(f"Échec de l'appel API {endpoint}: {exc}") from exc

    if response.status_code != 200:
        snippet = response.text[:500]
        raise RuntimeError(
            f"API {endpoint} a retourné {response.status_code}: {snippet}"
        )

    try:
        return response.json()
    except ValueError as exc:  # pragma: no cover - réponse invalide
        raise RuntimeError(f"Réponse API non JSON: {response.text[:200]}") from exc


# ============================================================================
# FONCTIONS D'ANALYSE
# ============================================================================

def check_patterns(
    anonymized: str,
    expected: list,
    forbidden: list = None,
    min_counts: dict | None = None,
) -> dict:
    """Vérifie la présence de patterns attendus et l'absence de patterns interdits."""
    results = {
        "success": True,
        "found": [],
        "missing": [],
        "forbidden_found": [],
        "counts": {},
        "count_missing": [],
    }
    
    for pattern in expected:
        if pattern in anonymized:
            results["found"].append(pattern)
        else:
            results["missing"].append(pattern)
            results["success"] = False
    
    if forbidden:
        for pattern in forbidden:
            if pattern in anonymized:
                results["forbidden_found"].append(pattern)
                results["success"] = False

    if min_counts:
        for pattern, min_count in min_counts.items():
            actual = anonymized.count(pattern)
            results["counts"][pattern] = actual
            if actual < min_count:
                results["success"] = False
                results["count_missing"].append(
                    {
                        "pattern": pattern,
                        "expected": min_count,
                        "found": actual,
                    }
                )
    
    return results


def check_audit_keys(audit: dict, expected_keys: list) -> dict:
    """Vérifie la présence de clés attendues dans l'audit."""
    results = {
        "success": True,
        "found": [],
        "missing": [],
    }
    
    for key in expected_keys:
        if key in audit:
            results["found"].append(key)
        else:
            results["missing"].append(key)
            results["success"] = False
    
    return results


def check_placeholder_consistency(result: dict) -> dict:
    """Vérifie que les mêmes entités ont les mêmes placeholders."""
    replacements = result["audit"]["replacements"]
    surface_to_placeholder = {}
    inconsistencies = []
    
    for repl in replacements:
        surface = repl["surface"]
        placeholder = repl["replacement"]
        
        if surface in surface_to_placeholder:
            if surface_to_placeholder[surface] != placeholder:
                inconsistencies.append({
                    "surface": surface,
                    "expected": surface_to_placeholder[surface],
                    "got": placeholder,
                })
        else:
            surface_to_placeholder[surface] = placeholder
    
    return {
        "success": len(inconsistencies) == 0,
        "inconsistencies": inconsistencies,
    }


# ============================================================================
# EXÉCUTION DES TESTS
# ============================================================================

def run_test_case(
    test_case: dict,
    *,
    scope_prefix: str,
    secret_salt: str,
    default_level: str,
    global_overrides: Dict[str, Any],
    api_url: str,
    api_timeout: float,
) -> dict:
    """Exécute un cas de test et retourne les résultats."""
    test_id = test_case["id"]
    description = test_case["description"]
    text = test_case["text"]
    level = test_case.get("level") or default_level or "L0"
    overrides = {}
    overrides.update(global_overrides or {})
    overrides.update(test_case.get("overrides") or {})
    if test_case.get("forbidden_patterns"):
        overrides.setdefault("forbidden_patterns", test_case["forbidden_patterns"])
    if test_case.get("expected_pattern_counts"):
        overrides.setdefault("expected_placeholder_counts", test_case["expected_pattern_counts"])
    
    print(f"\n{'='*80}")
    print(f"🧪 Test: {test_id}")
    print(f"📝 Description: {description}")
    print(f"📄 Texte original: {text}")
    print(f"🎚️  Niveau: {level}")
    if overrides:
        print(f"⚙️  Overrides: {json.dumps(overrides, indent=2)}")
    
    try:
        # Exécuter l'anonymisation
        scope_id = f"{scope_prefix}-{test_id}" if scope_prefix else test_id
        result = call_api_anonymize(
            api_url=api_url,
            timeout=api_timeout,
            text=text,
            scope_id=scope_id,
            level=level,
            secret_salt=secret_salt or "test-secret-123",
            overrides=overrides,
        )
        
        anonymized = result["anonymized_text"]
        audit = result["audit"]
        evaluation = result["evaluation"]
        
        print(f"\n✅ Texte anonymisé: {anonymized}")
        print(f"📊 Entités détectées: {len(audit['entities'])}")
        print(f"🔄 Remplacements: {len(audit['replacements'])}")
        print(f"📐 Généralisations: {len(audit['generalizations'])}")
        
        # Vérifications
        checks = {"success": True, "details": {}}
        
        # Vérifier les patterns attendus
        if "expected_patterns" in test_case:
            pattern_check = check_patterns(
                anonymized,
                test_case["expected_patterns"],
                test_case.get("forbidden_patterns"),
                test_case.get("expected_pattern_counts"),
            )
            checks["details"]["patterns"] = pattern_check
            if not pattern_check["success"]:
                checks["success"] = False
                print(f"\n❌ ÉCHEC - Patterns:")
                if pattern_check["missing"]:
                    print(f"   Manquants: {pattern_check['missing']}")
                if pattern_check["forbidden_found"]:
                    print(f"   Interdits trouvés: {pattern_check['forbidden_found']}")
                if pattern_check["count_missing"]:
                    for shortage in pattern_check["count_missing"]:
                        print(
                            "   Occurrences insuffisantes pour {pattern}: attendu {expected}, trouvé {found}".format(
                                pattern=shortage["pattern"],
                                expected=shortage["expected"],
                                found=shortage["found"],
                            )
                        )
        
        # Vérifier les clés d'audit
        if "expected_audit_keys" in test_case:
            audit_check = check_audit_keys(audit, test_case["expected_audit_keys"])
            checks["details"]["audit_keys"] = audit_check
            if not audit_check["success"]:
                checks["success"] = False
                print(f"\n❌ ÉCHEC - Clés audit manquantes: {audit_check['missing']}")
        
        # Vérifier la cohérence des placeholders
        if test_case.get("verify_placeholder_consistency"):
            consistency_check = check_placeholder_consistency(result)
            checks["details"]["consistency"] = consistency_check
            if not consistency_check["success"]:
                checks["success"] = False
                print(f"\n❌ ÉCHEC - Incohérences placeholders:")
                for inc in consistency_check["inconsistencies"]:
                    print(f"   '{inc['surface']}': attendu {inc['expected']}, obtenu {inc['got']}")
        
        # Afficher les erreurs LLM si présentes
        if audit.get("llm_errors"):
            print(f"\n⚠️  Erreurs LLM: {audit['llm_errors']}")
        
        # Afficher le statut final
        if checks["success"]:
            print(f"\n✅ TEST RÉUSSI")
        else:
            print(f"\n❌ TEST ÉCHOUÉ")
        
        return {
            "test_id": test_id,
            "success": checks["success"],
            "result": result,
            "checks": checks,
            "error": None,
        }
    
    except Exception as e:
        print(f"\n❌ ERREUR CRITIQUE: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        return {
            "test_id": test_id,
            "success": False,
            "result": None,
            "checks": None,
            "error": str(e),
        }


def main():
    """Exécute tous les cas de test et affiche un résumé."""
    start_time = time.time()
    args = parse_args()
    dataset, dataset_source = load_dataset_config(args.dataset)
    metadata = dataset.get("metadata") or {}
    dataset_defaults = dataset.get("defaults") or {}
    dataset_cases: List[Dict[str, Any]] = dataset.get("cases") or []

    scope_prefix = args.scope_prefix or dataset_defaults.get("scope_prefix") or "test"
    secret_salt = args.secret_salt or dataset_defaults.get("secret_salt") or "test-secret-123"
    default_level = args.default_level or dataset_defaults.get("level") or "L0"
    api_url = args.api_url or DEFAULT_API_URL
    api_timeout = args.api_timeout if args.api_timeout is not None else DEFAULT_API_TIMEOUT

    dataset_global_overrides = dataset.get("global_overrides") or {}
    cli_global_overrides = parse_global_overrides_arg(args.global_overrides)
    global_overrides = {**dataset_global_overrides, **cli_global_overrides}

    selected_ids = set(args.only) if args.only else None
    selected_cases: List[Dict[str, Any]] = []
    for case in dataset_cases:
        if selected_ids and case["id"] not in selected_ids:
            continue
        if case.get("skip"):
            continue
        selected_cases.append(case)

    if not selected_cases:
        print("Aucun cas à exécuter (filtre trop restrictif ?).")
        return 0

    print("=" * 80)
    print("🚀 TESTS EXHAUSTIFS DU PIPELINE D'ANONYMISATION")
    print("=" * 80)
    dataset_name = metadata.get("name", "(dataset sans nom)")
    print(f"Dataset: {dataset_name} (source: {dataset_source})")
    if metadata.get("description"):
        print(f"📝 {metadata['description']}")
    print(f"Cas sélectionnés: {len(selected_cases)} / {len(dataset_cases)}")
    print(f"API cible: {api_url.rstrip('/')}/anonymize (timeout {api_timeout}s)")

    results = []
    for test_case in selected_cases:
        result = run_test_case(
            test_case,
            scope_prefix=scope_prefix,
            secret_salt=secret_salt,
            default_level=default_level,
            global_overrides=global_overrides,
            api_url=api_url,
            api_timeout=api_timeout,
        )
        results.append(result)
        if args.fail_fast and not result["success"]:
            print("Arrêt anticipé (--fail-fast).")
            break

    print("\n" + "=" * 80)
    print("📊 RÉSUMÉ DES TESTS")
    print("=" * 80)

    total = len(results)
    passed = sum(1 for r in results if r["success"])
    failed = total - passed

    print(f"\nTotal: {total} tests")
    print(f"✅ Réussis: {passed}")
    print(f"❌ Échoués: {failed}")

    if failed > 0:
        print("\n❌ Tests échoués:")
        for r in results:
            if not r["success"]:
                error_msg = r["error"] or "Vérifications échouées"
                print(f"   - {r['test_id']}: {error_msg}")

    print("\n" + "=" * 80)

    duration_seconds = time.time() - start_time

    output_file = args.output
    ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        clean_results: List[Dict[str, Any]] = []
        entities_total = 0
        replacements_total = 0
        generalizations_total = 0
        llm_error_events = 0

        for r in results:
            clean_r: Dict[str, Any] = {
                "test_id": r["test_id"],
                "success": r["success"],
                "error": r["error"],
            }
            if r["result"]:
                audit = r["result"]["audit"]
                clean_r["anonymized_text"] = r["result"]["anonymized_text"]
                clean_r["entities_count"] = len(audit["entities"])
                clean_r["replacements_count"] = len(audit["replacements"])
                entities_total += len(audit["entities"])
                replacements_total += len(audit["replacements"])
                generalizations_total += len(audit["generalizations"])
                if audit.get("llm_errors"):
                    llm_error_events += len(audit["llm_errors"])
            clean_results.append(clean_r)

        summary = {
            "dataset_name": metadata.get("name", dataset_source),
            "dataset_source": dataset_source,
            "scope_prefix": scope_prefix,
            "cases_total": len(dataset_cases),
            "cases_executed": total,
            "cases_passed": passed,
            "cases_failed": failed,
            "success_rate": (passed / total) if total else 0,
            "duration_seconds": round(duration_seconds, 3),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "entities_detected_total": entities_total,
            "replacements_total": replacements_total,
            "generalizations_total": generalizations_total,
            "llm_error_events": llm_error_events,
        }

        report_payload = {
            "summary": summary,
            "results": clean_results,
        }

        json.dump(report_payload, f, indent=2, ensure_ascii=False)

    print(f"📁 Résultats détaillés sauvegardés dans: {output_file}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
