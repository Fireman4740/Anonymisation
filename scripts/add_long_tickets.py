import json

def add_ticket(dataset_path, ticket):
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    data["examples"].append(ticket)
    
    with open(dataset_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def calculate_offsets(text, spans):
    annotations = []
    for span_text, label, replacement, risk_note in spans:
        start = text.find(span_text)
        if start == -1:
            print(f"Warning: '{span_text}' not found in text.")
            continue
        # In a real scenario we might need to handle multiple occurrences
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

# Ticket 013: Long Infrastructure Log
text_013 = """LOG ANALYSIS REPORT - INCIDENT #8829
Timestamp: 2024-10-05 08:12:44
Severity: CRITICAL
Node: srv-prod-db-master-01.internal.cloud.net
IP Address: 10.240.15.2
Maintenance Contact: Arthur Pendragon (a.pendragon@camelot-solutions.fr)

[08:12:40] ERROR: Auth failed for user 'db_admin_svc' from 192.168.100.45.
[08:12:41] WARNING: Connection limit reached on eth0 (MAC: 00:50:56:AB:CD:EF).
[08:12:42] CRITICAL: System disk /dev/sda1 is 99% full. Path: /var/lib/mysql/data/camelot_db.
[08:12:43] ALERT: Secret key 'AKIA_SQL_KEY_SAMPLE_123' exposed in plain text in /etc/mysql/my.cnf.
[08:12:44] INFO: Escalating to Head of Infrastructure, Morgane LeFay (m.lefay@camelot-solutions.fr, +33 7 88 99 00 11).

Detailed Stack Trace:
java.sql.SQLException: Access denied for user 'db_admin_svc'@'192.168.100.45' (using password: YES)
    at com.mysql.cj.jdbc.exceptions.SQLError.createSQLException(SQLError.java:129)
    at com.mysql.cj.jdbc.exceptions.SQLExceptionsMapping.translateException(SQLExceptionsMapping.java:122)
    at com.mysql.cj.jdbc.ConnectionImpl.createNewIO(ConnectionImpl.java:836)
    ...
Impacted Customers:
- Excalibur Retail (Contract ID: EX-7788)
- RoundTable Finance (Project: Avalon-v2)

Resolution Steps:
1. Contact Arthur Pendragon for disk cleanup.
2. Rotate AWS key AKIA_SQL_KEY_SAMPLE_123.
3. Update firewall rules for 192.168.100.45.
"""

spans_013 = [
    ("srv-prod-db-master-01.internal.cloud.net", "HOST", "[HOST_A]", "Nom d'hôte interne"),
    ("10.240.15.2", "IP", "[IP_INTERNAL]", "IP de base de données"),
    ("Arthur Pendragon", "PER", "[PER_L1]", "Nom de l'employé"),
    ("a.pendragon@camelot-solutions.fr", "EMAIL", "[EMAIL_L1]", "Email professionnel"),
    ("camelot-solutions.fr", "ORG", "[ORG_CAMELOT]", "Société mentionnée dans email"),
    ("192.168.100.45", "IP", "[IP_CLIENT]", "IP source attaquante"),
    ("00:50:56:AB:CD:EF", "MAC", "[MAC_L1]", "Adresse MAC"),
    ("/var/lib/mysql/data/camelot_db", "PATH", "[PATH_DB]", "Chemin système"),
    ("AKIA_SQL_KEY_SAMPLE_123", "AWS_KEY", "[AWS_KEY_L1]", "Clé AWS"),
    ("Morgane LeFay", "PER", "[PER_L2]", "Nom de la responsable"),
    ("m.lefay@camelot-solutions.fr", "EMAIL", "[EMAIL_L2]", "Email responsable"),
    ("+33 7 88 99 00 11", "PHONE", "[PHONE_L1]", "Numéro de téléphone"),
    ("Excalibur Retail", "ORG", "[ORG_EXCALIBUR]", "Nom du client"),
    ("EX-7788", "ID", "[CONTRACT_A]", "ID de contrat"),
    ("RoundTable Finance", "ORG", "[ORG_ROUNDTABLE]", "Nom du client"),
    ("Avalon-v2", "PROJECT", "[PROJECT_A]", "Nom de projet confidentiel")
]

ticket_013 = {
    "id": "ticket_013",
    "langue": "EN",
    "original_text": text_013,
    "annotations": calculate_offsets(text_013, spans_013),
    "anonymized_text": "...", # Will be filled by the pipeline later or manually if needed
    "niveau_anonymisation": "L2 Standard",
    "risk_score": 75,
    "metadata": {
        "domain": "infrastructure",
        "date_creation": "2024-10-05"
    }
}

# Ticket 014: Long Multi-turn Conversation
text_014 = """CUSTOMER SUPPORT TRANSCRIPT - TRANSCRIPTION_ID_99012
Date: 2024-11-20
Participants: John (Support Agent), Elena Gilbert (Customer), Stefan (Billing Expert)

John [10:00]: Hello! This is John from GlobalNet Support. How can I help you today?
Elena Gilbert [10:01]: Hi John. I'm calling from Mystic Falls. I have an issue with my bill. My account number is ACCT-5544-3322.
John [10:01]: I see. Could you verify your address for security purposes?
Elena Gilbert [10:02]: Sure, it's 2104 Maple Street, Mystic Falls, VA 22101. My phone is 555-010-9988.
John [10:03]: Thank you Elena. I'm looking at your last transaction of $1,250.45 dated 2024-11-15. It seems it was charged to your Visa ending in 4455.
Elena Gilbert [10:04]: Yes, but I was promised a discount because I work for the Salvatore Boarding School as a teacher.
John [10:05]: Let me bring Stefan into this chat. He handles the institutional discounts.
Stefan [10:06]: Hello Elena. Stefan here. John told me about the Salvatore Boarding School discount. 
Stefan [10:07]: I see your email registered is elena.g@mystic-high.edu. I will apply a 20% discount. 
Elena Gilbert [10:08]: Thank you Stefan!
Stefan [10:09]: You're welcome. The updated amount is $1,000.36. I've sent the confirmation to your IP 172.16.254.1 for your record.
John [10:10]: Is there anything else?
Elena Gilbert [10:10]: No, that's it. Thanks John and Stefan!
"""

spans_014 = [
    ("John", "PER", "[PER_AGENT]", "Agent support"),
    ("Elena Gilbert", "PER", "[PER_CUSTOMER]", "Nom du client"),
    ("Stefan", "PER", "[PER_EXPERT]", "Expert billing"),
    ("Mystic Falls", "LOC", "[LOC_TOWN]", "Ville"),
    ("ACCT-5544-3322", "ID", "[ACCOUNT_ID]", "Numéro de compte"),
    ("2104 Maple Street, Mystic Falls, VA 22101", "LOC", "[LOC_FULL_ADDRESS]", "Adresse complète"),
    ("555-010-9988", "PHONE", "[PHONE_B]", "Téléphone"),
    ("$1,250.45", "AMOUNT", "[AMOUNT_A]", "Montant transaction"),
    ("2024-11-15", "DATE", "[DATE_A]", "Date transaction"),
    ("4455", "ID", "[CARD_LAST_4]", "Fin de CB"),
    ("Salvatore Boarding School", "ORG", "[ORG_SCHOOL]", "Lieu de travail"),
    ("teacher", "ROLE", "[ROLE_A]", "Métier"),
    ("elena.g@mystic-high.edu", "EMAIL", "[EMAIL_C]", "Email personnel"),
    ("$1,000.36", "AMOUNT", "[AMOUNT_B]", "Montant réduit"),
    ("172.16.254.1", "IP", "[IP_PRIVATE]", "IP client")
]

ticket_014 = {
    "id": "ticket_014",
    "langue": "EN",
    "original_text": text_014,
    "annotations": calculate_offsets(text_014, spans_014),
    "anonymized_text": "...",
    "niveau_anonymisation": "L3 Fort",
    "risk_score": 65,
    "metadata": {
        "domain": "customer_service",
        "date_creation": "2024-11-20"
    }
}

dataset_path = "eval/datasets/data/anonymization_dataset.json"
add_ticket(dataset_path, ticket_013)
add_ticket(dataset_path, ticket_014)
print("Added ticket_013 and ticket_014 to dataset.")
