import json

def calculate_offsets(text, spans):
    annotations = []
    for span_text, label, replacement, risk_note in spans:
        start = text.find(span_text)
        if start == -1:
            print(f"Warning: '{span_text}' not found in text.")
            continue
        # In case of multiple occurrences, take the first one for these examples
        annotations.append({
            "type": label,
            "start": start,
            "end": start + len(span_text),
            "text": span_text,
            "replacement": replacement,
            "coref_id": None,
            "risk_note": risk_note
        })
    return annotations

hard_examples = []

# 1. The Startup Founder in a small ecosystem
text_1 = "L'ancien CEO de la startup qui a inventé le distributeur de croquettes connecté en 2018 à Nantes, maintenant consultant en IA pour le secteur portuaire, a demandé un accès VPN."
spans_1 = [
    ("ancien CEO de la startup qui a inventé le distributeur de croquettes connecté", "QUASI_ID", "[ROLE_FOUNDER_UNIQUE]", "Description de poste unique via un produit spécifique"),
    ("2018", "DATE", "[DATE_YEAR]", "Année de création/évènement"),
    ("Nantes", "LOC", "[LOC_CITY]", "Ville"),
    ("consultant en IA pour le secteur portuaire", "QUASI_ID", "[ROLE_NICHE]", "Métier actuel très spécifique")
]
hard_examples.append({
    "id": "hard_001",
    "langue": "FR",
    "original_text": text_1,
    "annotations": calculate_offsets(text_1, spans_1),
    "anonymized_text": "L'[ROLE_FOUNDER_UNIQUE] en [DATE_YEAR] à [LOC_CITY], maintenant [ROLE_NICHE], a demandé un accès VPN.",
    "niveau_anonymisation": "L4 Max",
    "risk_score": 95,
    "metadata": {"type": "quasi_id_hard", "case": "startup_entrepreneur"}
})

# 2. Local Celebrity / Unique Position
text_2 = "Le seul habitant de l'île aux Moines possédant une Tesla Model S Plaid rouge a signalé une panne de recharge sur la borne publique du port."
spans_2 = [
    ("seul habitant de l'île aux Moines possédant une Tesla Model S Plaid rouge", "QUASI_ID", "[INDIVIDUAL_UNIQUE_ASSET]", "Identification par possession unique dans une petite zone géographique")
]
hard_examples.append({
    "id": "hard_002",
    "langue": "FR",
    "original_text": text_2,
    "annotations": calculate_offsets(text_2, spans_2),
    "anonymized_text": "Un [INDIVIDUAL_UNIQUE_ASSET] a signalé une panne de recharge sur la borne publique du port.",
    "niveau_anonymisation": "L4 Max",
    "risk_score": 100,
    "metadata": {"type": "quasi_id_hard", "case": "unique_asset_small_loc"}
})

# 3. Precise Health/Personal Habits
text_3 = "Une femme de 45 ans, marathonienne, qui a subi une triple fracture du métatarse lors du Marathon de Paris 2023, demande le remboursement de ses semelles orthopédiques de chez Podoplus."
spans_3 = [
    ("45 ans", "QUASI_ID", "[AGE]", "Âge"),
    ("marathonienne", "QUASI_ID", "[ACTIVITY]", "Activité physique"),
    ("triple fracture du métatarse", "QUASI_ID", "[MEDICAL_SPECIFIC]", "Blessure précise"),
    ("Marathon de Paris 2023", "EVENT", "[EVENT_DATE]", "Événement daté"),
    ("Podoplus", "ORG", "[ORG_PROVIDER]", "Fournisseur spécialisé")
]
hard_examples.append({
    "id": "hard_003",
    "langue": "FR",
    "original_text": text_3,
    "annotations": calculate_offsets(text_3, spans_3),
    "anonymized_text": "Une personne de [AGE], [ACTIVITY], ayant subi une [MEDICAL_SPECIFIC] lors d'un [EVENT_DATE], demande le remboursement de matériel chez [ORG_PROVIDER].",
    "niveau_anonymisation": "L3 Fort",
    "risk_score": 85,
    "metadata": {"type": "quasi_id_hard", "case": "medical_event_combo"}
})

# 4. Niche Technical Expert (Internal Data Leak)
text_4 = "Le développeur qui a maintenu le code legacy du module COBOL 'Z-TRANS-99' chez BanquePopulaire de 1995 à 2005 pense qu'il y a une faille de sécurité."
spans_4 = [
    ("développeur qui a maintenu le code legacy du module COBOL 'Z-TRANS-99'", "QUASI_ID", "[ROLE_OLD_TECH]", "Expertise sur un module interne obscur"),
    ("BanquePopulaire", "ORG", "[ORG_COMPANY]", "Entreprise"),
    ("1995 à 2005", "DATE", "[DATE_RANGE]", "Période d'activité précise")
]
hard_examples.append({
    "id": "hard_004",
    "langue": "FR",
    "original_text": text_4,
    "annotations": calculate_offsets(text_4, spans_4),
    "anonymized_text": "Un [ROLE_OLD_TECH] chez [ORG_COMPANY] durant [DATE_RANGE] pense qu'il y a une faille de sécurité.",
    "niveau_anonymisation": "L4 Max",
    "risk_score": 90,
    "metadata": {"type": "quasi_id_hard", "case": "legacy_tech_expert"}
})

# 5. Victim of a specific news event
text_5 = "Le propriétaire du garage incendié lors des émeutes de la rue de Rivoli le 1er mai dernier a contacté son assurance pour un véhicule de collection Jaguar Type E."
spans_5 = [
    ("propriétaire du garage incendié", "QUASI_ID", "[VICTIM_INCIDENT]", "Statut lié à un fait divers"),
    ("rue de Rivoli", "LOC", "[LOC_STREET]", "Localisation précise"),
    ("1er mai dernier", "DATE", "[DATE_EVENT]", "Date précise"),
    ("Jaguar Type E", "QUASI_ID", "[RARE_VEHICLE]", "Véhicule rare")
]
hard_examples.append({
    "id": "hard_005",
    "langue": "FR",
    "original_text": text_5,
    "annotations": calculate_offsets(text_5, spans_5),
    "anonymized_text": "Une personne [VICTIM_INCIDENT] à [LOC_STREET] le [DATE_EVENT] a contacté son assurance pour un [RARE_VEHICLE].",
    "niveau_anonymisation": "L4 Max",
    "risk_score": 98,
    "metadata": {"type": "quasi_id_hard", "case": "news_event_link"}
})

output_file = "eval/datasets/data/hard_quasi_id_dataset.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump({"examples": hard_examples}, f, indent=4, ensure_ascii=False)

print(f"Created {output_file} with {len(hard_examples)} hard Quasi-ID examples.")
