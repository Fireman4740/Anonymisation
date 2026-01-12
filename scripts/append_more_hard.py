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


dataset_path = "eval/datasets/data/hard_quasi_id_dataset.json"

new_examples = []

# hard_006: Small-city unique medical role
text_6 = (
    "La seule cardiologue pédiatrique de Villeneuve-sur-Lot, âgée de 39 ans, a signalé un manque "
    "de pacemakers infantiles au service de nuit."
)
spans_6 = [
    ("seule cardiologue pédiatrique de Villeneuve-sur-Lot", "QUASI_ID", "[ROLE_MED_UNIQUE]", "Poste médical unique dans une petite ville"),
    ("39 ans", "QUASI_ID", "[AGE_39]", "Âge précis"),
    ("pacemakers infantiles", "QUASI_ID", "[MED_DEVICE]", "Dispositif médical rare")
]
new_examples.append({
    "id": "hard_006",
    "langue": "FR",
    "original_text": text_6,
    "annotations": calculate_offsets(text_6, spans_6),
    "anonymized_text": "Une [ROLE_MED_UNIQUE], âgée de [AGE_39], a signalé un manque de [MED_DEVICE] au service de nuit.",
    "niveau_anonymisation": "L4 Max",
    "risk_score": 97,
    "metadata": {"type": "quasi_id_hard", "case": "medical_unique_city"}
})

# hard_007: Unique asset in tiny community
text_7 = (
    "À l'Île-aux-Oiseaux, le seul propriétaire d'une Aston Martin DB5 verte a demandé une expertise après un vol "
    "suspect sur le port local."
)
spans_7 = [
    ("seul propriétaire d'une Aston Martin DB5 verte", "QUASI_ID", "[UNIQUE_CAR_OWNER]", "Possession unique d'un véhicule de collection dans une petite île"),
    ("Île-aux-Oiseaux", "LOC", "[LOC_ISLAND]", "Localité très petite")
]
new_examples.append({
    "id": "hard_007",
    "langue": "FR",
    "original_text": text_7,
    "annotations": calculate_offsets(text_7, spans_7),
    "anonymized_text": "Un [UNIQUE_CAR_OWNER] à [LOC_ISLAND] a demandé une expertise après un vol suspect.",
    "niveau_anonymisation": "L4 Max",
    "risk_score": 100,
    "metadata": {"type": "quasi_id_hard", "case": "unique_asset_island"}
})

# hard_008: Employment + rare hobby + neighborhood
text_8 = (
    "Le maître-chien bénévole, instructeur du seul club cynophile de la ZAC des Mimosas, ancien champion de pistage, "
    "a été contacté par la gendarmerie locale."
)
spans_8 = [
    ("maître-chien bénévole, instructeur du seul club cynophile de la ZAC des Mimosas", "QUASI_ID", "[ROLE_VOLUNTEER_UNIQUE]", "Combinaison métier/activité unique localement"),
    ("ZAC des Mimosas", "LOC", "[LOC_NEIGH]", "Quartier spécifique")
]
new_examples.append({
    "id": "hard_008",
    "langue": "FR",
    "original_text": text_8,
    "annotations": calculate_offsets(text_8, spans_8),
    "anonymized_text": "Un [ROLE_VOLUNTEER_UNIQUE] du [LOC_NEIGH] a été contacté par la gendarmerie.",
    "niveau_anonymisation": "L3 Fort",
    "risk_score": 88,
    "metadata": {"type": "quasi_id_hard", "case": "volunteer_unique_activity"}
})

# hard_009: Academic niche + age + project
text_9 = (
    "Doctorante en cryptographie post-quantique, membre d'un groupe de recherche de 5 personnes à Clermont, âgée de 27 ans, "
    "cherche un hébergement collaboratif sur campus."
)
spans_9 = [
    ("Doctorante en cryptographie post-quantique", "QUASI_ID", "[ROLE_PHD_CRYPTO]", "Domaine académique très spécifique"),
    ("groupe de recherche de 5 personnes", "QUASI_ID", "[GROUP_SMALL]", "Taille d'équipe très réduite"),
    ("Clermont", "LOC", "[LOC_CITY]", "Ville universitaire"),
    ("27 ans", "QUASI_ID", "[AGE_27]", "Âge précis")
]
new_examples.append({
    "id": "hard_009",
    "langue": "FR",
    "original_text": text_9,
    "annotations": calculate_offsets(text_9, spans_9),
    "anonymized_text": "Une [ROLE_PHD_CRYPTO] membre d'un [GROUP_SMALL] à [LOC_CITY], âgée de [AGE_27], cherche un hébergement.",
    "niveau_anonymisation": "L3 Fort",
    "risk_score": 93,
    "metadata": {"type": "quasi_id_hard", "case": "academic_niche_small_group"}
})

# hard_010: Local politician + business + rare combination
text_10 = (
    "Le caviste et conseiller municipal de 52 ans, propriétaire du bar-épicerie 'Le Cep Romain' à Montrouge, a annoncé "
    "une fermeture temporaire suite à un contrôle sanitaire."
)
spans_10 = [
    ("caviste et conseiller municipal", "QUASI_ID", "[ROLE_POLITICO_BUSINESS]", "Cumul de rôle local visible"),
    ("52 ans", "QUASI_ID", "[AGE_52]", "Âge"),
    ("'Le Cep Romain'", "QUASI_ID", "[BUSINESS_NAME]", "Nom commercial local"),
    ("Montrouge", "LOC", "[LOC_TOWN]", "Ville/commune")
]
new_examples.append({
    "id": "hard_010",
    "langue": "FR",
    "original_text": text_10,
    "annotations": calculate_offsets(text_10, spans_10),
    "anonymized_text": "Un [ROLE_POLITICO_BUSINESS] de [AGE_52], propriétaire d'un [BUSINESS_NAME] à [LOC_TOWN], a annoncé une fermeture.",
    "niveau_anonymisation": "L3 Fort",
    "risk_score": 90,
    "metadata": {"type": "quasi_id_hard", "case": "local_politician_business"}
})

# Load existing dataset and append
with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

if "examples" not in data:
    data = {"examples": []}

for ex in new_examples:
    data["examples"].append(ex)

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Appended {len(new_examples)} examples to {dataset_path}")
