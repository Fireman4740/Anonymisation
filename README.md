# Anonymisation de Tickets & Conversations Support

## 1. Problème

Transformer des logs, tickets et dialogues de support en texte exploitable tout en empêchant la (ré)identification directe (emails, NIR, IBAN, téléphones, adresses…) et indirecte (quasi‑identifiants combinés, contexte métier, co‑références, style d’écriture) face à des attaquants capables de raisonnement (humains ou LLM) et de recoupements externes.

Contraintes: minimisation (GDPR), traçabilité, très faible risque résiduel, préservation de la valeur technique/narrative (chronologie, rôles, flux API, codes d’erreurs).

## 2. But

Obtenir un texte anonymisé mais utile pour: analyse, entraînement, animation/scénarisation. Préserver structure, relations entre acteurs/artefacts et cohérence technique via des placeholders typés et stables, avec réduction stylométrique contrôlée et audit automatisé du risque.

## 3. Architecture Résumée

- Couche Détection forte signal: regex + validateurs (emails, téléphones, NIR, IBAN, BIC…).
- NER multilingue (modèle local) pour PER / ORG / LOC.
- Pseudonymisation déterministe HMAC (`PseudoMapper`) → placeholders cohérents: `[PER_ABC]`, `[ORG_XYZ]`…
- Raisonner LLM (OpenRouter) pour: complétion de détection, co‑références implicites, paraphrase normalisée (stylométrie), audit de risque.
- Orchestrateur: fusion des spans, application hiérarchisée des remplacements, généralisation (dates, etc.), paraphrase optionnelle, audit + escalade de politique.
- Policy binaire actuelle (L0 sans LLM / L1 avec LLM) simplifiée mais extensible vers niveaux L2–L4.

## 4. Pipeline (vue logique)

1. Entrée texte + métadonnées (scope_id, secret_salt, level).
2. Détection regex + validation (filtrage chevauchements).
3. NER (agrégation simple) → spans supplémentaires.
4. Fusion + génération placeholders stables (HMAC(secret, scope, type, surface)).
5. (Optionnel LLM) Plan de détection enrichi / déductions / SAME_AS.
6. Généralisation (dates, lieux, montants…) selon politique.
7. Paraphrase contrôlée (placeholders gelés) pour réduire signature stylométrique.
8. Audit LLM (score risque + findings) → durcissement adaptatif si seuil dépassé.
9. Sortie: texte anonymisé + audit structuré + mapping éphémère (selon policy).

## 5. Placeholders & Pseudonymisation

Format: `[TYPE_CODE]` où TYPE ∈ {PER, ORG, LOC, API, HOST, TICKET…}. Le code est dérivé HMAC → stable intra scope, non réversible sans secret. Favorise la cohérence narrative (mêmes entités reconnues).

## 6. Généralisation (concept)

- Dates: précision graduelle (jour → semaine → mois → trimestre → année → redact).
- Lieux: adresse → ville → région → pays → redact.
- IP: exact → /24 → public/private → redact.
- Montants: exact → arrondi → tranche (10/100) → redact.
- Artefacts rares (build IDs, versions internes) → catégorisation ou redact.

## 7. Stylométrie

Paraphrase contrôlée: normalisation des formulations, ponctuation, politesses. Interdiction de modifier placeholders, codes d’erreurs ou structure factuelle. Objectif: réduire la probabilité d’attribution d’auteur tout en maintenant la fidélité technique.

## 8. Audit & Durcissement

LLM "auditeur" produit: score 0–100, findings (combinaisons rares, fuites implicites). Si score > seuil: escalade (plus de généralisation, placeholder style plus générique, augmentation paraphrase, suppression ORG, etc.).

## 9. État Actuel du Code

✔ Regex/validateurs de base.  
✔ NER multilingue basique.  
✔ Client OpenRouter + squelette Reasoner (détection / paraphrase / audit JSON strict).  
✔ Pseudonymisation HMAC stable (`src/utils_pseudo.py`).  
✔ Orchestrateur (fusion spans, hooks généralisation).  
✔ Validator Guardrails (`sanitize_pii`).  
✖ Couverture regex incomplète (IPs, URLs, hostnames internes, UUID, tickets, secrets…).  
✖ Généralisation fine partielle.  
✖ Co‑référence avancée manquante.  
✖ Métriques formelles (k‑anonymity, l‑diversity) absentes.  
✖ Export structuré "animation" non finalisé.

## 10. Roadmap Courte

1. Étendre les regex + validateurs (IP, URL, UUID, tickets, clés, montants, MAC, chemins internes).
2. Implémenter généralisation hiérarchique complète (dates, lieux, IP, montants).
3. Ajout co‑références (fusion des mentions, pronoms, alias).
4. Finaliser Reasoner LLM (plan → actions → audit) + boucle durcissement.
5. Export JSON narratif (agents, événements, mapping éphémère).
6. Ensemble métriques sécurité/utilité + tests adversariaux FR/EN.
7. Réactiver niveaux L0–L4 riches (intensité paraphrase, granularités, seuil audit, destruction mapping).

## 11. Installation (développement)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Exporter la clé OpenRouter si LLM utilisé
export OPENROUTER_API_KEY=sk-...
```

## 12. API Rapide

Démarrer le service:

```bash
python main_eval.py  # Flask
```

Requête:

```bash
curl -X POST http://localhost:8000/anonymize \
  -H 'Content-Type: application/json' \
  -d '{"text": "Bonjour, je suis Jean Dupont <jean@example.com>", "level": "L1"}'
```

## 13. Sécurité & Bonnes Pratiques

- Utiliser un `secret_salt` fort par lot pour limiter corrélations croisées.
- Séparer environnements (dev vs prod) pour éviter fuite de clés.
- Journaliser l’audit mais purger le mapping si policy le demande.
- Vérifier que la paraphrase ne réintroduit pas de fuites (tests régression).

## 14. Limites / Avertissements

La protection stylométrique est probabiliste. Les attaques par corrélation externe massives ou modèles spécialisés peuvent réduire l’efficacité. Toujours évaluer sur données représentatives et ajuster la politique.

## 15. Licence / Usage

Prototype interne orienté R&D anonymisation avancée; compléter selon politique de distribution souhaitée.

## 16. Modèles IA Utilisés

### NER (détection d'entités nommées)

- Modèle: `Davlan/bert-base-multilingual-cased-ner-hrl` (variable `NER_MODEL_PATH` dans `src/personal_info.py`).
- Chargé via `transformers` (pipeline `aggregation_strategy="simple"`).
- Rôle: fournir des entités PER / ORG / LOC (multilingue) avant enrichissement LLM.

### Raisonner / Paraphrase / Audit (LLM via OpenRouter)

Dans `src/orchestrator.py`, si la policy active le LLM (`policy.llm_detection`, etc.), un `LLMReasoner` est instancié avec les modèles (paramètre `openrouter_models` ou valeurs par défaut) :

```python
models = openrouter_models or {
  "detect": "qwen/qwen3-235b-a22b-thinking-2507",
  "paraphrase": "qwen/qwen3-235b-a22b-thinking-2507",
  "audit": "qwen/qwen3-235b-a22b-thinking-2507",
}
```

Un unique grand modèle (famille Qwen 3 235B) est utilisé pour les trois rôles par défaut :

- detect : production du plan JSON (entités supplémentaires, actions, généralisation fine)
- paraphrase : réduction stylométrique contrôlée (température ajustée `0.2 + 0.1 * intensity`)
- audit : évaluation du risque (score + findings) et déclenchement du durcissement.

### Formats & Contraintes

- Appels via `OpenRouterClient` (`src/openrouter_client.py`) avec `response_format={"type": "json_object"}` pour forcer le JSON strict.
- Si la sortie n'est pas strictement JSON : extraction regex du dernier objet (`re.search("{.*}" ...)`).
- Clé d'API attendue : variable d'environnement `OPENROUTER_API_KEY`.

### Personnalisation des modèles

Injection possible en passant `openrouter_models` à `anonymize_text` :

```python
out = anonymize_text(
  value=texte,
  scope_id="S1",
  secret_salt="secret",
  level="L1",
  openrouter_models={
    "detect": "meta-llama/llama-3.1-70b-instruct",
    "paraphrase": "google/gemma-2-27b-it",
    "audit": "qwen/qwen3-235b-a22b-thinking-2507",
  }
)
```

Clés manquantes → retombent sur les valeurs par défaut.

### Justification du choix actuel

Un seul gros modèle maximise cohérence des clusters / placeholders et couverture des signaux faibles (co‑références, quasi‑identifiants composites). Des spécialisations (modèle plus léger pour paraphrase) pourront être introduites une fois les métriques formelles en place.

---

Contributions: ouvrir PR avec description des entités ajoutées, règles de collision, et tests. Bug: fournir texte d’entrée + sortie + attendu.

## 17. Champs / Types Anonymisés (Actuels & Prévus)

Cette section récapitule tous les éléments que le pipeline traite (déjà implémentés ou prévus à court terme), leur méthode de détection, la forme de sortie et les remarques de sûreté.

### 17.1 PII & Identifiants techniques (Regex + Validateurs)

| Catégorie                       | Exemple d'entrée                       | Détection (regex / logique)                   | Placeholder / Transformation                                      | Remarques                                                       |
| ------------------------------- | -------------------------------------- | --------------------------------------------- | ----------------------------------------------------------------- | --------------------------------------------------------------- |
| Email                           | `jean.dupont@example.com`              | Regex stricte domaine + TLD                   | `<MAIL>` (puis potentiellement `[MAIL_XXX]` si futur typing HMAC) | Conservation structure phrase, supprime corrélation nom/domaine |
| Téléphone (FR & formes souples) | `+33 6 12 34 56 78`                    | Regex flexible numéros FR                     | `<TELEPHONE>`                                                     | Normalisation, formats variés supportés                         |
| NIR (Sécurité Sociale FR)       | `1 84 12 75 123 456 78`                | Regex + validation basique                    | `<NIR>`                                                           | Identifiant très sensible, toujours remplacé                    |
| IBAN\*                          | `FR76....`                             | Regex + (si `schwifty` dispo) validation IBAN | `<IBAN>`                                                          | Optionnel (dépend lib)                                          |
| BIC\*                           | `AGRIFRPP`                             | Regex + (si `schwifty`) validation BIC        | `<BIC>`                                                           | Optionnel                                                       |
| IPv4                            | `192.168.1.10`                         | Regex simple                                  | `<IP>` puis possible généralisation (priv/public, /24)            | Généralisation pilotée par policy future                        |
| IPv6 (simplifié)                | `fe80::1ff:fe23:4567:890a`             | Regex large                                   | `<IP>`                                                            | Pas encore de compression/expansion                             |
| URL                             | `https://host/app?id=9`                | Regex protocole + chars sûrs                  | `<URL>`                                                           | Paramètres non filtrés finement pour l'instant                  |
| Host / FQDN                     | `dev-23.internal.local`                | Regex labels DNS                              | `<HOST>`                                                          | Peut devenir `[HOST_XXX]` (HMAC)                                |
| Paths Unix                      | `/var/log/auth.log`                    | Regex chemin absolu                           | `<PATH>`                                                          | Risque de fuite structure interne                               |
| Paths Windows                   | `C:\Users\John\file.txt`               | Regex                                         | `<PATH>`                                                          | Idem                                                            |
| UUID v1–5                       | `550e8400-e29b-41d4-a716-446655440000` | Regex versions 1-5                            | `<UUID>`                                                          | Peut être généralisé en pattern stable                          |
| N° carte (13–19, Luhn)          | `4539 1488 0343 6467`                  | Regex + Luhn léger                            | `<CARD>`                                                          | Aucune troncature partielle (redact total)                      |
| AWS Access Key ID               | `AKIAxxxxxxxxxxxxxx`                   | Regex préfixe `AKIA`                          | `<AWS_KEY>`                                                       | Secret associé non capturé                                      |
| Secret générique                | `sk-abcDEF123...`                      | Regex `sk-` + longueur                        | `<SECRET>`                                                        | Peut couvrir API keys variées                                   |
| MAC Address                     | `00:11:22:33:44:55`                    | Regex hex paires                              | `<MAC>`                                                           | Pas de généralisation pour l'instant                            |
| Ticket ID                       | `PROJ-1234`                            | Regex JIRA-like                               | `<TICKET>`                                                        | Peut devenir `[TICKET_XXX]` (HMAC)                              |
| Username (clé explicite)        | `user:alice` / `login=foo_bar`         | Regex groupe capturé                          | `<USERNAME>`                                                      | Seul le token de valeur remplacé                                |
| Username (logs Linux)           | `uid=1001(alice)`                      | Regex parenthèse                              | `<USERNAME>`                                                      | Conserve le reste de la ligne                                   |

(\*): activé seulement si la bibliothèque `schwifty` est installée.

### 17.2 Entités NER (Multilingue)

| Type               | Exemple       | Source       | Placeholder généré | Détails pseudonymisation                                               |
| ------------------ | ------------- | ------------ | ------------------ | ---------------------------------------------------------------------- |
| Personne (PER)     | `Jean Dupont` | Pipeline NER | `[PER_ABC]`        | HMAC(secret, scope, "PER", surface normalisée) → code stable 3 lettres |
| Organisation (ORG) | `GlobalInc`   | NER          | `[ORG_XYZ]`        | Peut être ensuite généralisé/purgé selon policy future                 |
| Lieu (LOC)         | `Paris`       | NER          | `[LOC_QWE]`        | Granularité future (ville → pays)                                      |

Pronoms / co‑références: prévus via plan LLM (non encore pleinement fusionné).

### 17.3 Généralisation (Policy)

Actuel (simplifié):

- Dates (mode L1): conversion jour → mois (`2024-05-15` → `[DATE_2024-05]`).
- IP (placeholder simple `<IP>` pour l'instant; enrichissement prévu: public/private, /24, redact).
- Montants: détectés dans prompts LLM mais pas encore regexifiés systématiquement (prévu: binning).

Plan futur (escalade / durcissement):

- Dates: jour → semaine → mois → trimestre → année → redact.
- Lieux: adresse → ville → région → pays → redact.
- Montants: exact → arrondi → tranche (10 / 100) → large tranche → redact.
- Identifiants techniques rares (versions internes, build IDs) → catégorisation / redact.

### 17.4 Paraphrase Stylométrique

Ne modifie jamais:

- Placeholders `[TYPE_CODE]`
- Codes d'erreur (`ERR_FOO`, HTTP 4xx/5xx, `HTTP/1.1`)
- Tokens gelés (patterns dans `DEFAULT_FROZEN_PATTERNS`).

Transforme style lexical / ponctuation afin de réduire l'empreinte auteur tout en gardant la structure factuelle.

### 17.5 Audit & Score de Risque

Audit LLM produit:

- `risk_score` (0–100)
- `findings`: objets décrivant vecteurs de ré‑identification (combinaisons quasi‑uniques, fuites implicites)
- `recommendations` (placeholder, généralisation supplémentaire, paraphrase intensifiée)

### 17.6 Pseudonymisation Déterministe

`[TYPE_XXX]` où `XXX` = base26 dérivée des 32 premiers bits d'un HMAC(SHA‑256) sur `(secret_salt, scope_id, TYPE, surface_normalisée)`:

- Stable intra scope → cohérence conversationnelle.
- Changement de `secret_salt` ou `scope_id` casse le lien → protection contre corrélation externe.
- Aucune inversion possible sans les secrets.

### 17.7 Limitations Actuelles

- Pas encore: détection systématique des montants (`$25000`), versions (`v4.2.1`), adresses postales structurées.
- Co‑références pronoms / alias partielle (dépend des réponses LLM et intégration future du plan).
- Pas de métriques formelles (k‑anonymity) — à instrumenter.
- Pas de mapping export stable (reverse minimal) fourni par défaut; ephemeral en mémoire.

### 17.8 Extension Facile

Ajouter un nouveau pattern:

1. Définir regex (+ éventuel validateur) dans `text_sanitizer.py`.
2. Taguer avec un label `<TYPE>`.
3. (Optionnel) Ajouter logique NER ou LLM si contexte.
4. Documenter ici dans un PR.

---

git clone https://github.com/NorskRegnesentral/text-anonymization-benchmark
python -m spacy download en_core_web_md

python eval_tab.py \
 --tab_repo Dataset/text-anonymization-benchmark \
 --gold_json Dataset/text-anonymization-benchmark/echr_test.json \
 --level L0 \
 --split test \
 --no_paraphrase \
 --save_preds ./debug_dump/preds_tab_L0_test.json \
 --dump_dir ./debug_dump/tab_test_L0 \
 --dump_k 30 \
 --dump_strategy errors

## 18. Ensemble NER (Transformers, DeepPavlov, GLiNER)

Le projet supporte jusqu'à trois sources d'entités nommées combinables :

1. HF Transformers (par défaut) : modèle multilingue `Davlan/bert-base-multilingual-cased-ner-hrl`.
2. DeepPavlov (configurations Ontonotes / autres) : ex. `ner_ontonotes_bert`.
3. GLiNER (zero-shot / multi-label) : ex. `urchade/gliner_large-v2.1`, `urchade/gliner_multi-v2.1`.

L'agrégation permet d'augmenter rappel et robustesse multi‑langue avant pseudonymisation.

### 18.1 Installation des dépendances optionnelles

Par défaut seules les dépendances de base sont installées. Pour activer toutes les variantes NER :

```bash
# (Dans votre venv déjà activé)
# HF est déjà couvert par transformers dans requirements.txt (vérifier).

pip install deeppavlov
python -m deeppavlov install ner_ontonotes_bert  # télécharge poids + dépendances spécifiques

pip install gliner
```

Notes DeepPavlov : l'appel à `build_model(..., download=True, install=True)` dans le code gère aussi l'installation si vous ne lancez pas la commande explicite `python -m deeppavlov install ...`.

### 18.2 Variables d'environnement de performance

| Variable             | Effet                                                                  |
| -------------------- | ---------------------------------------------------------------------- |
| `NER_FORCE_DEVICE`   | Forcer `cuda`, `mps` ou `cpu` (auto-détection sinon).                  |
| `NER_HALF_PRECISION` | `1/true` : tente le float16 sur GPU pour HF/GLiNER.                    |
| `NER_FAST`           | `1/true` : active un mode accéléré (moins de vérifications, batching). |
| `NER_DEBUG`          | `1/true` : logs internes de `src/ner_ensemble.py`.                     |

Exemple (PowerShell Windows) :

```powershell
$env:NER_FORCE_DEVICE = "cuda"
$env:NER_HALF_PRECISION = "1"
$env:NER_FAST = "1"
```

### 18.3 API interne (fonctions disponibles)

Les fonctions principales (dans `src/ner_ensemble.py`) :

- `run_hf_ner_chunked(text, max_tokens=384, stride=64, min_conf=0.55)`
- `run_deeppavlov_ner_ensemble(text, dp_config_names=[...], mode='union'|'consensus', min_votes=1)`
- `run_gliner(text, model_names=[...], labels=[...], threshold=0.35)`
- `merge_ner_lists(*lists)` pour fusionner sans doublons.

### 18.4 Exemple minimal (script)

```python
from src.ner_ensemble import (
    run_hf_ner_chunked,
    run_deeppavlov_ner_ensemble,
    run_gliner,
    merge_ner_lists,
)

text = "Bob Ross lived in Florida. Elon Musk founded Tesla."

# 1. HF Transformers
hf_ents = run_hf_ner_chunked(text)

# 2. DeepPavlov (installer préalablement: pip install deeppavlov ; python -m deeppavlov install ner_ontonotes_bert)
#    On peut passer plusieurs configs ex: ['ner_ontonotes_bert','ner_ontonotes_bert_mult'] si existantes
try:
    dp_ents = run_deeppavlov_ner_ensemble(text, dp_config_names=['ner_ontonotes_bert'], mode='union')
except Exception:
    dp_ents = []  # lib non installée → silencieux

# 3. GLiNER (pip install gliner)
try:
    gln_ents = run_gliner(text, labels=["Person","Organization","Location"], threshold=0.35)
except Exception:
    gln_ents = []

all_ents = merge_ner_lists(hf_ents, dp_ents, gln_ents)
print(all_ents)
```

### 18.5 Modes d'agrégation

- `union` : conserve toute entité atteignant `min_votes` (>=1 par défaut) dans une source.
- `consensus` : exige que l'entité soit validée par toutes les sources chargées (votes == nombre de modèles). Utile pour augmenter la précision quand rappel ≈ suffisant.

Les fonctions DeepPavlov et GLiNER ajoutent un champ `votes` par entité.

### 18.6 Sélection / normalisation des labels

- DeepPavlov : conversion BIO → spans, normalisation (PERSON, ORGANIZATION, LOCATION, GPE, etc.) + quelques PII dérivés (EMAIL→MAIL...).
- GLiNER : labels textuels adaptés vers le même espace (PERSON, ORGANIZATION, LOCATION...).
- HF : `entity_group` déjà agrégé via `aggregation_strategy="simple"`.

### 18.7 Bonnes pratiques

1. Activer `NER_FAST=1` pour lots volumineux (réduction overhead Python) après validation qualité.
2. Utiliser `consensus` si vous observez trop de faux positifs sur données internes.
3. Limiter le nombre de modèles GLiNER (grands checkpoints → RAM GPU) ou réduire `labels`.
4. Sur GPU mémoire limitée : commencer par HF seul, puis ajouter GLiNER; DeepPavlov peut être CPU.
5. Journaliser temps d'inférence par composant pour guider l'arbitrage performance/qualité.

### 18.8 Dépannage rapide

| Problème                 | Cause probable               | Action                                                       |
| ------------------------ | ---------------------------- | ------------------------------------------------------------ |
| OOM CUDA HF              | fenêtre trop large           | Réduire `max_tokens` (ex 256) ou désactiver half si instable |
| Pas d'entités DeepPavlov | Modèle non installé          | `python -m deeppavlov install ner_ontonotes_bert`            |
| ImportError gliner       | Dépendance manquante         | `pip install gliner`                                         |
| Lenteur extrême          | Empilement 3 sources sur CPU | Désactiver une source ou activer GPU                         |
| Labels incohérents       | Variantes label GLiNER       | Vérifier normalisation / compléter `_normalize_gliner_label` |

### 18.9 Preset GLiNER "best" & Pondération des modèles

Nouveauté : un preset `best` combine plusieurs checkpoints GLiNER récents et complémentaires pour maximiser le rappel tout en conservant une bonne précision.

Modèles utilisés (ordre = priorité / confiance décroissante) :

1. `EmergentMethods/gliner_medium_news-v2.1`
2. `numind/NuNER_Zero-span`
3. `urchade/gliner_large-v2.1`
4. `urchade/gliner_multi-v2.1`

Une pondération de vote (somme des poids) est appliquée si la variable d'environnement suivante est active :

- `GLINER_WEIGHTING=1` (activé par défaut dans le code si non précisé explicitement)

Poids (peuvent évoluer) :

- EmergentMethods/gliner_medium_news-v2.1 → 1.25
- numind/NuNER_Zero-span → 1.20
- urchade/gliner_large-v2.1 → 1.10
- urchade/gliner_multi-v2.1 → 1.05
- (Autres modèles non listés → 1.0 par défaut)

Activation du preset :

- Via variable d'env : `GLINER_PRESET=best`
- Ou automatiquement injecté par `main_eval.py` (API Flask) et `eval_tab.py` (script d'évaluation) si :
  - Aucune configuration `GLINER_PRESET` n'est déjà définie dans l'environnement
  - Aucun override `gliner_models` explicite n'est fourni dans la requête / CLI

Désactivation de la pondération :

```powershell
# Windows PowerShell
$env:GLINER_WEIGHTING = "0"
```

Ou bash :

```bash
export GLINER_WEIGHTING=0
```

Override manuel dans un appel API :

```json
{
	"text": "Alice et Bob travaillent chez SpaceZ à Paris.",
	"level": "L2",
	"overrides": {
		"gliner_models": ["urchade/gliner_medium-v2.1"],
		"ner_use_gliner": true
	}
}
```

CLI (évaluation TAB) pour désactiver le preset auto :

```bash
python eval_tab.py --tab_repo Dataset/text-anonymization-benchmark \
  --gold_json Dataset/text-anonymization-benchmark/echr_test.json \
  --level L0 --no_gliner_best
```

Cas d'usage recommandés :

- Données hétérogènes multi‑domaines : laisser `best` + pondération activée.
- Contrainte GPU mémoire stricte : réduire à 1–2 modèles (`gliner_medium` ou `gliner_large`).
- Latence critique : utiliser preset `fast` ou désactiver GLiNER (`overrides: {"ner_use_gliner": false}`).

---
