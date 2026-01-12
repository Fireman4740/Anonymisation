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

text_015 = """SECURITY AUDIT LOG - SYSTEM EXTERNAL ACCESS
DATE: 2025-01-12
OBJECT: Investigation on unauthorized access from external IP 45.12.33.102.

Summary:
An anomalous activity was detected by the security system on Sunday evening. The user account involved belongs to Jean-Pierre Raffarin, a consultant at Elysée Strategy (JP.Raffarin@elysee.fr).
The access originated from a residential ISP in Poitiers, France.

Events Timeline:
1. 22:15:02 - Login from 45.12.33.102 using valid credentials for 'j_raffarin'.
2. 22:16:30 - Access to /var/www/internal-portal/secrets/config.php. System path: /mnt/storage/nas-01/backup/sql_dump_2025.zip.
3. 22:17:45 - Download initiated of 4.5GB of data. 
4. 22:18:10 - Alert triggered on session ID: SID-ABC-990-112.

Personnel Involved:
- Security Analyst: Brigitte Macron (b.macron@elysee.fr)
- IT Director: Emmanuel Macron (e.macron@elysee.fr, +33 1 22 33 44 55)

The incident was reported to the local police department in Paris 8ème.
Reference case: POLICE-2025-XP-998.

Original Log Snippet:
[2025-01-12 22:15:02] SSHD: Accepted password for j_raffarin from 45.12.33.102 port 54322 ssh2
[2025-01-12 22:16:15] KERNEL: sda2: write cache: enabled, read cache: enabled, supports DPO and FUA
[2025-01-12 22:16:30] AUDIT: user=j_raffarin action=read file=/var/www/internal-portal/secrets/config.php
[2025-01-12 22:17:00] APP: Session SID-ABC-990-112 started background task for /home/j_raffarin/exports/client_list.csv

Additional Context:
The user was using a MacBook Pro with serial number C02XG123XXXX.
MAC Address: 1A:2B:3C:4D:5E:6F.
Secondary email listed: raffarin.jp@pro-consulting.com.
Home address: 12 Rue de l'Eglise, 86000 Poitiers.
"""

spans_015 = [
    ("45.12.33.102", "IP", "[IP_EXTERNAL]", "IP source suspecte"),
    ("Jean-Pierre Raffarin", "PER", "[PER_A]", "Nom du consultant"),
    ("Elysée Strategy", "ORG", "[ORG_A]", "Cabinet de conseil"),
    ("JP.Raffarin@elysee.fr", "EMAIL", "[EMAIL_A]", "Email pro"),
    ("Poitiers", "LOC", "[LOC_REGION]", "Ville d'origine"),
    ("/var/www/internal-portal/secrets/config.php", "PATH", "[PATH_A]", "Chemin vers code source"),
    ("/mnt/storage/nas-01/backup/sql_dump_2025.zip", "PATH", "[PATH_B]", "Chemin vers backup"),
    ("SID-ABC-990-112", "ID", "[SESSION_ID]", "ID de session"),
    ("Brigitte Macron", "PER", "[PER_B]", "Analyste sécurité"),
    ("b.macron@elysee.fr", "EMAIL", "[EMAIL_B]", "Email analyste"),
    ("Emmanuel Macron", "PER", "[PER_C]", "Directeur IT"),
    ("e.macron@elysee.fr", "EMAIL", "[EMAIL_C]", "Email directeur"),
    ("+33 1 22 33 44 55", "PHONE", "[PHONE_A]", "Téléphone directeur"),
    ("Paris 8ème", "LOC", "[LOC_CITY]", "Localisation police"),
    ("POLICE-2025-XP-998", "ID", "[CASE_ID]", "Référence dossier"),
    ("C02XG123XXXX", "ID", "[SERIAL_NUMBER]", "Numéro de série matériel"),
    ("1A:2B:3C:4D:5E:6F", "MAC", "[MAC_A]", "Adresse MAC"),
    ("raffarin.jp@pro-consulting.com", "EMAIL", "[EMAIL_D]", "Email perso"),
    ("12 Rue de l'Eglise, 86000 Poitiers", "LOC", "[LOC_ADDRESS]", "Adresse personnelle")
]

ticket_015 = {
    "id": "ticket_015",
    "langue": "FR",
    "original_text": text_015,
    "annotations": calculate_offsets(text_015, spans_015),
    "anonymized_text": "...",
    "niveau_anonymisation": "L4 Max destructif",
    "risk_score": 90,
    "metadata": {
        "domain": "security_audit",
        "date_creation": "2025-01-12"
    }
}

with open("eval/datasets/data/anonymization_dataset.json", 'r', encoding='utf-8') as f:
    data = json.load(f)
data["examples"].append(ticket_015)
with open("eval/datasets/data/anonymization_dataset.json", 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)
print("Added ticket_015.")
