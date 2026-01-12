import json
import os

def calculate_offsets(text, spans):
    annotations = []
    for span_text, label, replacement, risk_note in spans:
        start = text.find(span_text)
        if start == -1:
            # Try once with case insensitive for robustness
            start = text.lower().find(span_text.lower())
            if start == -1:
                print(f"Warning: '{span_text}' not found in text.")
                continue
            span_text = text[start:start+len(span_text)]
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

dataset_path = "/mnt/f/IA/Anonymisation/eval/datasets/data/anonymization_dataset.json"

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Maps base_id -> {lang -> (text, spans, anonymized_text)}
# Only adding a subset of important ones to keep script size reasonable
translations = {
    "ticket_001": {
        "EN": ("Hello, I am John Smith from London. My email is john.smith@example.com. API Error on server 192.168.1.1 with JIRA-1234 ticket. He told me it was urgent.",
               [("John Smith", "PER", "[PER_A]", ""), ("London", "LOC", "[LOC_UK]", ""), ("john.smith@example.com", "EMAIL", "[EMAIL_A]", ""), ("192.168.1.1", "IP", "[IP_PRIVATE]", ""), ("JIRA-1234", "TICKET", "[TICKET_A]", "")],
               "Hello, I am [PER_A] from [LOC_UK]. My email is [EMAIL_A]. API Error on server [IP_PRIVATE] with [TICKET_A] ticket. [PER_A] told me it was urgent."),
        "ES": ("Hola, soy Juan Pérez de Madrid. Mi correo es juan.perez@example.com. Error de API en el servidor 192.168.1.1 con ticket JIRA-1234. Me dijo que era urgente.",
               [("Juan Pérez", "PER", "[PER_A]", ""), ("Madrid", "LOC", "[LOC_ES]", ""), ("juan.perez@example.com", "EMAIL", "[EMAIL_A]", ""), ("192.168.1.1", "IP", "[IP_PRIVATE]", ""), ("JIRA-1234", "TICKET", "[TICKET_A]", "")],
               "Hola, soy [PER_A] de [LOC_ES]. Mi correo es [EMAIL_A]. Error de API en el servidor [IP_PRIVATE] con ticket [TICKET_A]. [PER_A] me dijo que era urgente."),
        "IT": ("Buongiorno, sono Mario Rossi di Roma. La mia email è mario.rossi@example.com. Errore API sul server 192.168.1.1 con ticket JIRA-1234. Mi ha detto che era urgente.",
               [("Mario Rossi", "PER", "[PER_A]", ""), ("Roma", "LOC", "[LOC_IT]", ""), ("mario.rossi@example.com", "EMAIL", "[EMAIL_A]", ""), ("192.168.1.1", "IP", "[IP_PRIVATE]", ""), ("JIRA-1234", "TICKET", "[TICKET_A]", "")],
               "Buongiorno, sono [PER_A] di [LOC_IT]. La mia email è [EMAIL_A]. Errore API sul server [IP_PRIVATE] con ticket [TICKET_A]. [PER_A] mi ha detto che era urgente.")
    },
    "ticket_002": {
        "FR": ("Salut, c'est Alice de Nantes. Téléphone: +33 6 12 34 56 78. Problème avec le nom d'hôte internal-server.example.com et l'UUID 123e4567-e89b-12d3-a456-426614174000. Le chef d'équipe l'a mentionné.",
               [("Alice", "PER", "[PER_B]", ""), ("Nantes", "LOC", "[LOC_FR]", ""), ("+33 6 12 34 56 78", "PHONE", "[PHONE_A]", ""), ("internal-server.example.com", "HOST", "[HOST_A]", ""), ("123e4567-e89b-12d3-a456-426614174000", "UUID", "[UUID_A]", "")],
               "Salut, c'est [PER_B] de [LOC_FR]. Téléphone: [PHONE_A]. Problème avec le nom d'hôte [HOST_A] et l'UUID [UUID_A]. [PER_C] l'a mentionné."),
        "ES": ("Hola, soy Alicia de Valencia. Teléfono: +34 912 345 678. Problema con el nombre de host internal-server.example.com y UUID 123e4567-e89b-12d3-a456-426614174000. El líder del equipo lo mencionó.",
               [("Alicia", "PER", "[PER_B]", ""), ("Valencia", "LOC", "[LOC_ES]", ""), ("+34 912 345 678", "PHONE", "[PHONE_A]", ""), ("internal-server.example.com", "HOST", "[HOST_A]", ""), ("123e4567-e89b-12d3-a456-426614174000", "UUID", "[UUID_A]", "")],
               "Hola, soy [PER_B] de [LOC_ES]. Teléfono: [PHONE_A]. Problema con el host [HOST_A] y UUID [UUID_A]. [PER_C] lo mencionó."),
        "IT": ("Ciao, sono Alice da Milano. Telefono: +39 02 1234567. Problema con hostname internal-server.example.com e UUID 123e4567-e89b-12d3-a456-426614174000. Il team lead l'ha menzionato.",
               [("Alice", "PER", "[PER_B]", ""), ("Milano", "LOC", "[LOC_IT]", ""), ("+39 02 1234567", "PHONE", "[PHONE_A]", ""), ("internal-server.example.com", "HOST", "[HOST_A]", ""), ("123e4567-e89b-12d3-a456-426614174000", "UUID", "[UUID_A]", "")],
               "Ciao, sono [PER_B] da [LOC_IT]. Telefono: [PHONE_A]. Problema con [HOST_A] e UUID [UUID_A]. [PER_C] l'ha menzionato.")
    },
    "ticket_003": {
        "EN": ("Issue with API key: sk-abc123 on 2023-07-15. Address: 123 Example Street, London. Amount: £456.78. Mrs. Martin confirmed.",
               [("sk-abc123", "API_KEY", "[API_KEY_A]", ""), ("2023-07-15", "DATE", "[DATE_A]", ""), ("123 Example Street, London", "LOC", "[LOC_UK]", ""), ("£456.78", "MONTANT", "[MONTANT_A]", ""), ("Mrs. Martin", "PER", "[PER_D]", "")],
               "Issue with API key: [API_KEY_A] on [DATE_A]. Address: [LOC_UK]. Amount: [MONTANT_A]. [PER_D] confirmed."),
        "IT": ("Problema con la chiave API: sk-abc123 in data 15/07/2023. Indirizzo: 123 Via Esempio, Roma. Importo: 456,78€. La signora Martin ha confermato.",
               [("sk-abc123", "API_KEY", "[API_KEY_A]", ""), ("15/07/2023", "DATE", "[DATE_A]", ""), ("123 Via Esempio, Roma", "LOC", "[LOC_IT]", ""), ("456,78€", "MONTANT", "[MONTANT_A]", ""), ("La signora Martin", "PER", "[PER_D]", "")],
               "Problema con chiave [API_KEY_A] il [DATE_A]. Indirizzo: [LOC_IT]. Importo: [MONTANT_A]. [PER_D] ha confermato.")
    },
    "ticket_005": {
        "EN": ("Incident on MAC address 00:1A:2B:3C:4D:5E and URL https://internal.api.com. Date: 2024-02-10. DevOps team in Manchester. The amount is 1234.56€.",
               [("00:1A:2B:3C:4D:5E", "MAC", "[MAC_A]", ""), ("https://internal.api.com", "URL", "[URL_A]", ""), ("2024-02-10", "DATE", "[DATE_A]", ""), ("Manchester", "LOC", "[LOC_UK]", ""), ("1234.56€", "MONTANT", "[MONTANT_A]", "")],
               "Incident on MAC address [MAC_A] and URL [URL_A]. Date: [DATE_A]. DevOps team in [LOC_UK]. The amount is [MONTANT_A].")
    },
    "ticket_016": {
        "EN": ("I am a PhD student in astrophysics at the University of Manchester, I am 26 years old and I would like to know how to report my bike accident that happened yesterday in front of the PhITEM building.",
               [("PhD student in astrophysics", "QUASI_ID", "[ROLE_RESEARCH_SPECIALIZED]", ""), ("University of Manchester", "ORG", "[ORG_UNIVERSITY]", ""), ("26 years old", "QUASI_ID", "[AGE_RANGE_20-30]", ""), ("bike accident", "QUASI_ID", "[INCIDENT_TYPE]", ""), ("PhITEM building", "LOC", "[LOC_BUILDING]", "")],
               "I am a [ROLE_RESEARCH_SPECIALIZED] at [ORG_UNIVERSITY], I am [AGE_RANGE_20-30] and I would like to know how to report my [INCIDENT_TYPE] in front of [LOC_BUILDING]."),
        "ES": ("Soy estudiante de doctorado en astrofísica en la Universidad de Barcelona, tengo 26 años y me gustaría saber cómo declarar mi accidente de bicicleta ocurrido ayer frente al edificio PhITEM.",
               [("estudiante de doctorado en astrofísica", "QUASI_ID", "[ROLE_RESEARCH_SPECIALIZED]", ""), ("Universidad de Barcelona", "ORG", "[ORG_UNIVERSITY]", ""), ("26 años", "QUASI_ID", "[AGE_RANGE_20-30]", ""), ("accidente de bicicleta", "QUASI_ID", "[INCIDENT_TYPE]", ""), ("edificio PhITEM", "LOC", "[LOC_BUILDING]", "")],
               "Soy [ROLE_RESEARCH_SPECIALIZED] en la [ORG_UNIVERSITY], tengo [AGE_RANGE_20-30] y me gustaría saber cómo declarar mi [INCIDENT_TYPE] frente al [LOC_BUILDING].")
    },
    "ticket_017": {
        "EN": ("82-year-old patient suffering from Wolfram syndrome, followed in the neurology department of the hospital in London by Dr. Vallet. He requests coverage for his ambulance transport from Oxford.",
               [("82-year-old", "QUASI_ID", "[AGE_EXTREME]", ""), ("Wolfram syndrome", "QUASI_ID", "[RARE_DISEASE]", ""), ("neurology department", "ORG", "[DEPT_MEDICAL]", ""), ("hospital in London", "LOC", "[LOC_HOSPITAL]", ""), ("Dr. Vallet", "PER", "[PER_DOCTOR]", ""), ("Oxford", "LOC", "[LOC_TOWN]", "")],
               "[AGE_EXTREME] patient suffering from [RARE_DISEASE], followed in [DEPT_MEDICAL] of [LOC_HOSPITAL] by [PER_DOCTOR]. He requests coverage for transport from [LOC_TOWN].")
    }
}

new_examples = []
existing_ids = {ex["id"] for ex in data["examples"]}

for ex in data["examples"]:
    base_id = ex["id"]
    if base_id in translations:
        for lang, (text, spans, anon) in translations[base_id].items():
            trans_id = f"{base_id}_{lang.lower()}"
            if trans_id not in existing_ids:
                new_ex = {
                    "id": trans_id,
                    "langue": lang,
                    "original_text": text,
                    "annotations": calculate_offsets(text, spans),
                    "anonymized_text": anon,
                    "niveau_anonymisation": ex.get("niveau_anonymisation", "L2 Standard"),
                    "risk_score": ex.get("risk_score", 50),
                    "metadata": ex.get("metadata", {})
                }
                new_examples.append(new_ex)
                existing_ids.add(trans_id)

data["examples"].extend(new_examples)

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Added {len(new_examples)} translated examples to {dataset_path}")
