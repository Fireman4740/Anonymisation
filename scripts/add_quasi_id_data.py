import json

def calculate_offsets(text, spans):
    annotations = []
    for span_text, label, replacement, risk_note in spans:
        start = text.find(span_text)
        if start == -1:
            print(f"Warning: '{span_text}' not found in text.")
            continue
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

dataset_path = "eval/datasets/data/anonymization_dataset.json"

# Example 16: Research/University (k-anonymity focus)
text_016 = "Je suis doctorant en astrophysique à l'Université de Grenoble, j'ai 26 ans et je voudrais savoir comment déclarer mon accident de vélo survenu hier devant le bâtiment PhITEM."
spans_016 = [
    ("doctorant en astrophysique", "QUASI_ID", "[ROLE_RESEARCH_SPECIALIZED]", "Poste très spécifique dans une petite équipe"),
    ("Université de Grenoble", "ORG", "[ORG_UNIVERSITY]", "Lieu de travail"),
    ("26 ans", "QUASI_ID", "[AGE_RANGE_20-30]", "Tranche d'âge"),
    ("accident de vélo", "QUASI_ID", "[INCIDENT_TYPE]", "Événement marquant récent"),
    ("bâtiment PhITEM", "LOC", "[LOC_BUILDING]", "Localisation précise augmentant le risque de ré-identification")
]

# Example 17: Medical (Rare disease + Loc)
text_017 = "Patient de 82 ans souffrant du syndrome de Wolfram, suivi au service neurologie de l'hôpital de Poitiers par le Dr. Vallet. Il demande une prise en charge pour son transport en ambulance depuis Châtellerault."
spans_017 = [
    ("82 ans", "QUASI_ID", "[AGE_EXTREME]", "Âge avancé combiné à pathologie"),
    ("syndrome de Wolfram", "QUASI_ID", "[RARE_DISEASE]", "Maladie rare (faible k-anonymat)"),
    ("service neurologie", "ORG", "[DEPT_MEDICAL]", "Département hospitalier"),
    ("hôpital de Poitiers", "LOC", "[LOC_HOSPITAL]", "Hôpital régional"),
    ("Dr. Vallet", "PER", "[PER_DOCTOR]", "Nom du médecin"),
    ("Châtellerault", "LOC", "[LOC_TOWN]", "Ville de résidence")
]

# Example 18: HR/Corporate (Unique attributes)
text_018 = "La seule femme ingénieure du département Cloud chez TechSolution à Lyon, qui a travaillé sur le projet 'Icare' l'an dernier, a soumis une plainte concernant les horaires de nuit commencés en décembre."
spans_018 = [
    ("seule femme ingénieure", "QUASI_ID", "[UNIQUE_GENDER_ROLE]", "Attribut unique dans le département (k=1)"),
    ("département Cloud", "ORG", "[DEPT_TECH]", "Département"),
    ("TechSolution", "ORG", "[ORG_COMPANY]", "Entreprise"),
    ("Lyon", "LOC", "[LOC_CITY]", "Ville"),
    ("projet 'Icare'", "PROJECT", "[PROJECT_CONFIDENTIAL]", "Nom du projet"),
    ("horaires de nuit", "QUASI_ID", "[SPECIFIC_STRESS]", "Condition de travail spécifique"),
    ("décembre", "DATE", "[DATE_MONTH]", "Mois")
]

# Example 19: Public Figure/Local (Combination of roles)
text_019 = "Le boulanger de Saint-Cirq-Lapopie, qui est aussi le premier adjoint à la mairie, s'inquiète du retard de livraison de son nouveau four commandé chez FourPro France."
spans_019 = [
    ("boulanger", "QUASI_ID", "[ROLE_COMMERCIAL]", "Métier local"),
    ("Saint-Cirq-Lapopie", "LOC", "[LOC_SMALL_VILLAGE]", "Village à faible population (k-anonymat critique)"),
    ("premier adjoint à la mairie", "QUASI_ID", "[ROLE_PUBLIC]", "Fonction publique locale unique"),
    ("FourPro France", "ORG", "[ORG_SUPPLIER]", "Fournisseur")
]

new_examples = [
    {
        "id": "ticket_016",
        "langue": "FR",
        "original_text": text_016,
        "annotations": calculate_offsets(text_016, spans_016),
        "anonymized_text": "Je suis [ROLE_RESEARCH_SPECIALIZED] dans une [ORG_UNIVERSITY], j'ai [AGE_RANGE_20-30] et je voudrais savoir comment déclarer mon [INCIDENT_TYPE] survenu hier devant un [LOC_BUILDING].",
        "niveau_anonymisation": "L3 Fort",
        "risk_score": 85,
        "metadata": {"domain": "education", "reason": "quasi-id_detection"}
    },
    {
        "id": "ticket_017",
        "langue": "FR",
        "original_text": text_017,
        "annotations": calculate_offsets(text_017, spans_017),
        "anonymized_text": "Patient de [AGE_RANGE_80+] souffrant d'une [RARE_DISEASE], suivi dans un [DEPT_MEDICAL] d'un [LOC_HOSPITAL] par [PER_DOCTOR]. Il demande une prise en charge pour son transport depuis une [LOC_TOWN].",
        "niveau_anonymisation": "L3 Fort",
        "risk_score": 95,
        "metadata": {"domain": "health", "reason": "quasi-id_detection"}
    },
    {
        "id": "ticket_018",
        "langue": "FR",
        "original_text": text_018,
        "annotations": calculate_offsets(text_018, spans_018),
        "anonymized_text": "Une [ROLE_ENGINEER] du [DEPT_TECH] chez [ORG_COMPANY] à [LOC_CITY], ayant travaillé sur le [PROJECT_CONFIDENTIAL] l'an dernier, a soumis une plainte concernant les [SPECIFIC_STRESS] commencés en [DATE_SEASON].",
        "niveau_anonymisation": "L4 Max",
        "risk_score": 90,
        "metadata": {"domain": "hr", "reason": "quasi-id_detection"}
    },
    {
        "id": "ticket_019",
        "langue": "FR",
        "original_text": text_019,
        "annotations": calculate_offsets(text_019, spans_019),
        "anonymized_text": "Un [ROLE_COMMERCIAL] d'une [LOC_SMALL_VILLAGE], qui occupe aussi une [ROLE_PUBLIC], s'inquiète du retard de livraison de matériel commandé chez [ORG_SUPPLIER].",
        "niveau_anonymisation": "L3 Fort",
        "risk_score": 80,
        "metadata": {"domain": "public_sector", "reason": "quasi-id_detection"}
    }
]

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

data["examples"].extend(new_examples)

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Added {len(new_examples)} examples focusing on Quasi-IDs.")
