import json
import os

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

with open(dataset_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

# Filter out previously added translations if any (to avoid duplicates if run multiple times)
original_examples = [ex for ex in data["examples"] if not ex["id"].endswith(('_en', '_es', '_it'))]
translated_examples = []

translations_es = {
    "hard_001": ("El ex CEO de la startup que inventó el dispensador de croquetas conectado en 2018 en Nantes, ahora consultor de IA para el sector portuario, solicitó acceso VPN.", 
                [("ex CEO de la startup que inventó el dispensador de croquetas conectado", "QUASI_ID", "[ROLE_FOUNDER_UNIQUE]", "Descripción única"), ("2018", "DATE", "[DATE_YEAR]", ""), ("Nantes", "LOC", "[LOC_CITY]", ""), ("consultor de IA para el sector portuario", "QUASI_ID", "[ROLE_NICHE]", "")], 
                "El [ROLE_FOUNDER_UNIQUE] en [DATE_YEAR] en [LOC_CITY], ahora [ROLE_NICHE], solicitó acceso VPN."),
    "hard_002": ("El único habitante de Île aux Moines que posee un Tesla Model S Plaid rojo informó de un fallo de carga en la terminal pública del puerto.", 
                [("único habitante de Île aux Moines que posee un Tesla Model S Plaid rojo", "QUASI_ID", "[INDIVIDUAL_UNIQUE_ASSET]", "")], 
                "Un [INDIVIDUAL_UNIQUE_ASSET] informó de un fallo de carga."),
    "hard_003": ("Una mujer de 45 años, corredora de maratón, que sufrió una triple fractura de metatarso durante el Maratón de París 2023, solicita el reembolso de sus plantillas ortopédicas de Podoplus.", 
                [("45 años", "QUASI_ID", "[AGE]", ""), ("corredora de maratón", "QUASI_ID", "[ACTIVITY]", ""), ("triple fractura de metatarso", "QUASI_ID", "[MEDICAL_SPECIFIC]", ""), ("Maratón de París 2023", "EVENT", "[EVENT_DATE]", ""), ("Podoplus", "ORG", "[ORG_PROVIDER]", "")], 
                "Una persona de [AGE], [ACTIVITY], que sufrió una [MEDICAL_SPECIFIC] durante el [EVENT_DATE], solicita reembolso de [ORG_PROVIDER]."),
    "hard_004": ("El desarrollador que mantuvo el código heredado del módulo COBOL 'Z-TRANS-99' en BanquePopulaire de 1995 a 2005 cree que hay un fallo de seguridad.", 
                [("desarrollador que mantuvo el código heredado del módulo COBOL 'Z-TRANS-99'", "QUASI_ID", "[ROLE_OLD_TECH]", ""), ("BanquePopulaire", "ORG", "[ORG_COMPANY]", ""), ("1995 a 2005", "DATE", "[DATE_RANGE]", "")], 
                "Un [ROLE_OLD_TECH] en [ORG_COMPANY] durante [DATE_RANGE] cree que hay un fallo."),
    "hard_005": ("El dueño del taller incendiado durante los disturbios de la Rue de Rivoli el pasado 1 de mayo contactó con su seguro por un vehículo de colección Jaguar E-Type.", 
                [("dueño del taller incendiado", "QUASI_ID", "[VICTIM_INCIDENT]", ""), ("Rue de Rivoli", "LOC", "[LOC_STREET]", ""), ("1 de mayo", "DATE", "[DATE_EVENT]", ""), ("Jaguar E-Type", "QUASI_ID", "[RARE_VEHICLE]", "")], 
                "Una persona [VICTIM_INCIDENT] en [LOC_STREET] el [DATE_EVENT] contactó por un [RARE_VEHICLE]."),
    "hard_006": ("La única cardióloga pediátrica de Villeneuve-sur-Lot, de 39 años, informó de la falta de marcapasos infantiles en el servicio nocturno.", 
                [("única cardióloga pediátrica de Villeneuve-sur-Lot", "QUASI_ID", "[ROLE_MED_UNIQUE]", ""), ("39", "QUASI_ID", "[AGE_39]", ""), ("marcapasos infantiles", "QUASI_ID", "[MED_DEVICE]", "")], 
                "Una [ROLE_MED_UNIQUE], de [AGE_39], informó de la falta de [MED_DEVICE]."),
    "hard_007": ("En Île-aux-Oiseaux, el único propietario de un Aston Martin DB5 verde solicitó un peritaje tras un robo sospechoso en el puerto local.", 
                [("único propietario de un Aston Martin DB5 verde", "QUASI_ID", "[UNIQUE_CAR_OWNER]", ""), ("Île-aux-Oiseaux", "LOC", "[LOC_ISLAND]", "")], 
                "Un [UNIQUE_CAR_OWNER] en [LOC_ISLAND] solicitó un peritaje."),
    "hard_008": ("El guía canino voluntario, instructor del único club canino de la ZAC des Mimosas, ex campeón de rastreo, fue contactado por la gendarmería local.", 
                [("guía canino voluntario, instructor del único club canino de la ZAC des Mimosas", "QUASI_ID", "[ROLE_VOLUNTEER_UNIQUE]", ""), ("ZAC des Mimosas", "LOC", "[LOC_NEIGH]", "")], 
                "Un [ROLE_VOLUNTEER_UNIQUE] de [LOC_NEIGH] fue contactado."),
    "hard_009": ("Estudiante de doctorado en criptografía post-cuántica, miembro de un grupo de investigación de 5 personas en Clermont, de 27 años, busca alojamiento colaborativo en el campus.", 
                [("Estudiante de doctorado en criptografía post-cuántica", "QUASI_ID", "[ROLE_PHD_CRYPTO]", ""), ("grupo de investigación de 5 personas", "QUASI_ID", "[GROUP_SMALL]", ""), ("Clermont", "LOC", "[LOC_CITY]", ""), ("27", "QUASI_ID", "[AGE_27]", "")], 
                "Una [ROLE_PHD_CRYPTO] de un [GROUP_SMALL] en [LOC_CITY], de [AGE_27], busca alojamiento."),
    "hard_010": ("El vinatero y concejal de 52 años, dueño de la bodega-tienda 'Le Cep Romain' en Montrouge, anunció un cierre temporal tras una inspección sanitaria.", 
                [("vinatero y concejal", "QUASI_ID", "[ROLE_POLITICO_BUSINESS]", ""), ("52", "QUASI_ID", "[AGE_52]", ""), ("'Le Cep Romain'", "QUASI_ID", "[BUSINESS_NAME]", ""), ("Montrouge", "LOC", "[LOC_TOWN]", "")], 
                "Un [ROLE_POLITICO_BUSINESS] de [AGE_52], dueño de [BUSINESS_NAME] en [LOC_TOWN], anunció un cierre.")
}

translations_it = {
    "hard_001": ("L'ex CEO della startup che ha inventato il distributore di croccantini connesso nel 2018 a Nantes, ora consulente IA per il settore portuale, ha richiesto un accesso VPN.", 
                [("ex CEO della startup che ha inventato il distributore di croccantini connesso", "QUASI_ID", "[ROLE_FOUNDER_UNIQUE]", ""), ("2018", "DATE", "[DATE_YEAR]", ""), ("Nantes", "LOC", "[LOC_CITY]", ""), ("consulente IA per il settore portuale", "QUASI_ID", "[ROLE_NICHE]", "")], 
                "L'[ROLE_FOUNDER_UNIQUE] nel [DATE_YEAR] a [LOC_CITY], ora [ROLE_NICHE], ha richiesto accesso VPN."),
    "hard_002": ("L'unico abitante dell'Île aux Moines che possiede una Tesla Model S Plaid rossa ha segnalato un guasto alla ricarica presso il terminal pubblico del porto.", 
                [("unico abitante dell'Île aux Moines che possiede una Tesla Model S Plaid rossa", "QUASI_ID", "[INDIVIDUAL_UNIQUE_ASSET]", "")], 
                "Un [INDIVIDUAL_UNIQUE_ASSET] ha segnalato un guasto."),
    "hard_003": ("Una donna di 45 anni, maratoneta, che ha subito una tripla frattura del metatarso durante la Maratona di Parigi 2023, richiede il rimborso delle sue solette ortopediche da Podoplus.", 
                [("45 anni", "QUASI_ID", "[AGE]", ""), ("maratoneta", "QUASI_ID", "[ACTIVITY]", ""), ("tripla frattura del metatarso", "QUASI_ID", "[MEDICAL_SPECIFIC]", ""), ("Maratona di Parigi 2023", "EVENT", "[EVENT_DATE]", ""), ("Podoplus", "ORG", "[ORG_PROVIDER]", "")], 
                "Una persona di [AGE], [ACTIVITY], che ha subito [MEDICAL_SPECIFIC] durante [EVENT_DATE], richiede rimborso da [ORG_PROVIDER]."),
    "hard_004": ("Lo sviluppatore che ha mantenuto il codice legacy del modulo COBOL 'Z-TRANS-99' presso BanquePopulaire dal 1995 al 2005 ritiene che ci sia una falla di sicurezza.", 
                [("sviluppatore che ha mantenuto il codice legacy del modulo COBOL 'Z-TRANS-99'", "QUASI_ID", "[ROLE_OLD_TECH]", ""), ("BanquePopulaire", "ORG", "[ORG_COMPANY]", ""), ("1995 al 2005", "DATE", "[DATE_RANGE]", "")], 
                "Un [ROLE_OLD_TECH] presso [ORG_COMPANY] durante [DATE_RANGE] ritiene che ci sia una falla."),
    "hard_005": ("Il proprietario del garage bruciato durante le rivolte di Rue de Rivoli lo scorso 1° maggio ha contattato la sua assicurazione per un veicolo da collezione Jaguar E-Type.", 
                [("proprietario del garage bruciato", "QUASI_ID", "[VICTIM_INCIDENT]", ""), ("Rue de Rivoli", "LOC", "[LOC_STREET]", ""), ("1° maggio", "DATE", "[DATE_EVENT]", ""), ("Jaguar E-Type", "QUASI_ID", "[RARE_VEHICLE]", "")], 
                "Una persona [VICTIM_INCIDENT] in [LOC_STREET] il [DATE_EVENT] ha contattato per una [RARE_VEHICLE]."),
    "hard_006": ("L'unica cardiologa pediatrica di Villeneuve-sur-Lot, di 39 anni, ha segnalato la mancanza di pacemaker infantili nel servizio notturno.", 
                [("unica cardiologa pediatrica di Villeneuve-sur-Lot", "QUASI_ID", "[ROLE_MED_UNIQUE]", ""), ("39", "QUASI_ID", "[AGE_39]", ""), ("pacemaker infantili", "QUASI_ID", "[MED_DEVICE]", "")], 
                "Una [ROLE_MED_UNIQUE], di [AGE_39], ha segnalato la mancanza di [MED_DEVICE]."),
    "hard_007": ("Sull'Île-aux-Oiseaux, l'unico proprietario di una Aston Martin DB5 verde ha richiesto una perizia dopo un furto sospetto al porto locale.", 
                [("unico proprietario di una Aston Martin DB5 verde", "QUASI_ID", "[UNIQUE_CAR_OWNER]", ""), ("Île-aux-Oiseaux", "LOC", "[LOC_ISLAND]", "")], 
                "Un [UNIQUE_CAR_OWNER] sull'[LOC_ISLAND] ha richiesto una perizia."),
    "hard_008": ("L'addestratore cinofilo volontario, istruttore dell'unico club cinofilo della ZAC des Mimosas, ex campione di ricerca, è stato contattato dalla gendarmeria locale.", 
                [("addestratore cinofilo volontario, istruttore dell'unico club cinofilo della ZAC des Mimosas", "QUASI_ID", "[ROLE_VOLUNTEER_UNIQUE]", ""), ("ZAC des Mimosas", "LOC", "[LOC_NEIGH]", "")], 
                "Un [ROLE_VOLUNTEER_UNIQUE] della [LOC_NEIGH] è stato contattato."),
    "hard_009": ("Dottoranda in crittografia post-quantistica, membro di un gruppo di ricerca di 5 persone a Clermont, di 27 anni, cerca un alloggio collaborativo nel campus.", 
                [("Dottoranda in crittografia post-quantistica", "QUASI_ID", "[ROLE_PHD_CRYPTO]", ""), ("grupa di ricerca di 5 persone", "QUASI_ID", "[GROUP_SMALL]", ""), ("Clermont", "LOC", "[LOC_CITY]", ""), ("27", "QUASI_ID", "[AGE_27]", "")], 
                "Una [ROLE_PHD_CRYPTO] membro di un [GROUP_SMALL] a [LOC_CITY], di [AGE_27], cerca alloggio."),
    "hard_010": ("Il vestaio e consigliere comunale di 52 anni, titolare del bar-drogheria 'Le Cep Romain' a Montrouge, ha annunciato una chiusura temporanea a seguito di un'ispezione sanitaria.", 
                [("vestaio e consigliere comunale", "QUASI_ID", "[ROLE_POLITICO_BUSINESS]", ""), ("52", "QUASI_ID", "[AGE_52]", ""), ("'Le Cep Romain'", "QUASI_ID", "[BUSINESS_NAME]", ""), ("Montrouge", "LOC", "[LOC_TOWN]", "")], 
                "Un [ROLE_POLITICO_BUSINESS] di [AGE_52], titolare di [BUSINESS_NAME] a [LOC_TOWN], ha annunciato una chiusura.")
}

for ex in original_examples:
    base_id = ex["id"]
    
    # English
    text_en = ""
    spans_en = []
    anon_en = ""
    if base_id == "hard_001":
        text_en = "The former CEO of the startup that invented the connected kibble dispenser in 2018 in Nantes, now an AI consultant for the port sector, requested VPN access."
        spans_en = [("former CEO of the startup that invented the connected kibble dispenser", "QUASI_ID", "[ROLE_FOUNDER_UNIQUE]", ""), ("2018", "DATE", "[DATE_YEAR]", ""), ("Nantes", "LOC", "[LOC_CITY]", ""), ("AI consultant for the port sector", "QUASI_ID", "[ROLE_NICHE]", "")]
        anon_en = "The [ROLE_FOUNDER_UNIQUE] in [DATE_YEAR] in [LOC_CITY], now an [ROLE_NICHE], requested VPN access."
    elif base_id == "hard_002":
        text_en = "The only inhabitant of Île aux Moines owning a red Tesla Model S Plaid reported a charging failure at the port's public terminal."
        spans_en = [("only inhabitant of Île aux Moines owning a red Tesla Model S Plaid", "QUASI_ID", "[INDIVIDUAL_UNIQUE_ASSET]", "")]
        anon_en = "A [INDIVIDUAL_UNIQUE_ASSET] reported a charging failure."
    elif base_id == "hard_003":
        text_en = "A 45-year-old woman, a marathon runner, who suffered a triple metatarsal fracture during the 2023 Paris Marathon, is requesting reimbursement for her orthopedic insoles from Podoplus."
        spans_en = [("45-year-old", "QUASI_ID", "[AGE]", ""), ("marathon runner", "QUASI_ID", "[ACTIVITY]", ""), ("triple metatarsal fracture", "QUASI_ID", "[MEDICAL_SPECIFIC]", ""), ("2023 Paris Marathon", "EVENT", "[EVENT_DATE]", ""), ("Podoplus", "ORG", "[ORG_PROVIDER]", "")]
        anon_en = "A [AGE] person, [ACTIVITY], who suffered a [MEDICAL_SPECIFIC] during a [EVENT_DATE], is requesting reimbursement for equipment from [ORG_PROVIDER]."
    elif base_id == "hard_004":
        text_en = "The developer who maintained the legacy code for the COBOL module 'Z-TRANS-99' at BanquePopulaire from 1995 to 2005 believes there is a security flaw."
        spans_en = [("developer who maintained the legacy code for the COBOL module 'Z-TRANS-99'", "QUASI_ID", "[ROLE_OLD_TECH]", ""), ("BanquePopulaire", "ORG", "[ORG_COMPANY]", ""), ("1995 to 2005", "DATE", "[DATE_RANGE]", "")]
        anon_en = "A [ROLE_OLD_TECH] at [ORG_COMPANY] during [DATE_RANGE] believes there is a security flaw."
    elif base_id == "hard_005":
        text_en = "The owner of the garage burned down during the Rue de Rivoli riots last May 1st contacted his insurance for a Jaguar E-Type collector vehicle."
        spans_en = [("owner of the garage burned down", "QUASI_ID", "[VICTIM_INCIDENT]", ""), ("Rue de Rivoli", "LOC", "[LOC_STREET]", ""), ("May 1st", "DATE", "[DATE_EVENT]", ""), ("Jaguar E-Type", "QUASI_ID", "[RARE_VEHICLE]", "")]
        anon_en = "A person [VICTIM_INCIDENT] at [LOC_STREET] on [DATE_EVENT] contacted his insurance for a [RARE_VEHICLE]."
    elif base_id == "hard_006":
        text_en = "The only pediatric cardiologist in Villeneuve-sur-Lot, aged 39, reported a lack of infant pacemakers in the night service."
        spans_en = [("only pediatric cardiologist in Villeneuve-sur-Lot", "QUASI_ID", "[ROLE_MED_UNIQUE]", ""), ("39", "QUASI_ID", "[AGE_39]", ""), ("infant pacemakers", "QUASI_ID", "[MED_DEVICE]", "")]
        anon_en = "A [ROLE_MED_UNIQUE], aged [AGE_39], reported a lack of [MED_DEVICE]."
    elif base_id == "hard_007":
        text_en = "On Île-aux-Oiseaux, the only owner of a green Aston Martin DB5 requested an appraisal after a suspicious theft at the local port."
        spans_en = [("only owner of a green Aston Martin DB5", "QUASI_ID", "[UNIQUE_CAR_OWNER]", ""), ("Île-aux-Oiseaux", "LOC", "[LOC_ISLAND]", "")]
        anon_en = "A [UNIQUE_CAR_OWNER] on [LOC_ISLAND] requested an appraisal."
    elif base_id == "hard_008":
        text_en = "The volunteer dog handler, instructor of the only dog club in the ZAC des Mimosas, former tracking champion, was contacted by the local gendarmerie."
        spans_en = [("volunteer dog handler, instructor of the only dog club in the ZAC des Mimosas", "QUASI_ID", "[ROLE_VOLUNTEER_UNIQUE]", ""), ("ZAC des Mimosas", "LOC", "[LOC_NEIGH]", "")]
        anon_en = "A [ROLE_VOLUNTEER_UNIQUE] from [LOC_NEIGH] was contacted."
    elif base_id == "hard_009":
        text_en = "PhD student in post-quantum cryptography, member of a 5-person research group in Clermont, aged 27, looking for collaborative campus housing."
        spans_en = [("PhD student in post-quantum cryptography", "QUASI_ID", "[ROLE_PHD_CRYPTO]", ""), ("research group of 5 people", "QUASI_ID", "[GROUP_SMALL]", ""), ("Clermont", "LOC", "[LOC_CITY]", ""), ("27", "QUASI_ID", "[AGE_27]", "")]
        anon_en = "A [ROLE_PHD_CRYPTO] member of a [GROUP_SMALL] in [LOC_CITY], aged [AGE_27], is looking for housing."
    elif base_id == "hard_010":
        text_en = "The wine merchant and city councilor, aged 52, owner of the 'Le Cep Romain' bar-grocery in Montrouge, announced a temporary closure following a health inspection."
        spans_en = [("wine merchant and city councilor", "QUASI_ID", "[ROLE_POLITICO_BUSINESS]", ""), ("52", "QUASI_ID", "[AGE_52]", ""), ("'Le Cep Romain'", "QUASI_ID", "[BUSINESS_NAME]", ""), ("Montrouge", "LOC", "[LOC_TOWN]", "")]
        anon_en = "A [ROLE_POLITICO_BUSINESS] aged [AGE_52], owner of a [BUSINESS_NAME] in [LOC_TOWN], announced a closure."

    if text_en:
        translated_examples.append({
            "id": base_id + "_en",
            "langue": "EN",
            "original_text": text_en,
            "annotations": calculate_offsets(text_en, spans_en),
            "anonymized_text": anon_en,
            "niveau_anonymisation": ex["niveau_anonymisation"],
            "risk_score": ex["risk_score"],
            "metadata": ex["metadata"]
        })

    if base_id in translations_es:
        t_text, t_spans, t_anon = translations_es[base_id]
        translated_examples.append({
            "id": base_id + "_es",
            "langue": "ES",
            "original_text": t_text,
            "annotations": calculate_offsets(t_text, t_spans),
            "anonymized_text": t_anon,
            "niveau_anonymisation": ex["niveau_anonymisation"],
            "risk_score": ex["risk_score"],
            "metadata": ex["metadata"]
        })

    if base_id in translations_it:
        t_text, t_spans, t_anon = translations_it[base_id]
        translated_examples.append({
            "id": base_id + "_it",
            "langue": "IT",
            "original_text": t_text,
            "annotations": calculate_offsets(t_text, t_spans),
            "anonymized_text": t_anon,
            "niveau_anonymisation": ex["niveau_anonymisation"],
            "risk_score": ex["risk_score"],
            "metadata": ex["metadata"]
        })

# Maintain only original + newly generated translations
final_examples = original_examples + translated_examples
data["examples"] = final_examples

with open(dataset_path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Total examples now: {len(data['examples'])}")
