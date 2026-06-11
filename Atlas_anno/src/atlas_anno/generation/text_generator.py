from __future__ import annotations

from collections import Counter
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

from atlas_anno.surface_grounding import find_occurrences
from atlas_anno.generation.surface_forms import SurfaceOverride, build_surface_overrides

from atlas_anno.schemas import (
    AnnotationBundle,
    CandidatePools,
    CharacterProfile,
    DocumentRecord,
    GeneratedTextDraft,
    GroundedMention,
    ScenarioSpec,
    World,
)


MentionPart = Tuple[str, str, str]

RECIPIENT_LABELS = {
    "support_manager": "l'equipe support",
    "service_desk": "le service desk",
    "identity_support": "l'equipe identite",
    "data_ops": "l'equipe data ops",
    "customer_support": "le support client",
    "billing_support": "l'equipe facturation",
    "partner_support": "le support partenaires",
    "project_team": "l'equipe projet",
    "ops_manager": "l'equipe operations",
    "platform_team": "l'equipe plateforme",
    "client_success": "l'equipe client success",
    "vendor_contact": "le contact fournisseur",
}

SUPPORT_GOAL_TEXT = {
    "request_help": [
        "j'ai besoin d'un coup de main sur",
        "je vous sollicite pour un point qui bloque sur",
    ],
    "incident_report": [
        "je vous signale un incident encore visible sur",
        "je remonte un incident actif sur",
    ],
    "access_issue": [
        "je n'arrive plus a acceder correctement a",
        "l'acces reste bloque sur",
    ],
    "integration_issue": [
        "l'integration entre",
        "la liaison entre",
    ],
    "billing_mismatch": [
        "je constate un decalage de facturation sur",
        "la partie facturation remonte de travers sur",
    ],
    "permission_review": [
        "la revue des acces reste en attente sur",
        "le circuit de validation des droits ne va pas au bout sur",
    ],
    "data_gap": [
        "il manque une partie des donnees dans",
        "les donnees ne remontent pas completement dans",
    ],
    "sync_delay": [
        "la synchro prend trop de retard dans",
        "les mises a jour arrivent avec decalage dans",
    ],
}

EMAIL_GOAL_TEXT = {
    "coordination": [
        "je vous fais un point pour recaler la coordination",
        "je vous partage un point court pour aligner la suite",
    ],
    "project_followup": [
        "je reviens sur le suivi du projet",
        "je fais un retour rapide sur l'avancement",
    ],
    "urgent_incident": [
        "je vous alerte sur un incident qui monte autour de",
        "je vous transfere le contexte sur un incident prioritaire cote",
    ],
    "context_share": [
        "je vous partage le contexte utile avant la suite",
        "je pose ici le contexte avant les prochaines actions",
    ],
    "handover": [
        "je centralise les infos pour la reprise du dossier",
        "je vous transmets le contexte de reprise",
    ],
    "vendor_followup": [
        "je fais un point avant relance fournisseur",
        "je consolide le contexte avant retour fournisseur",
    ],
    "capacity_alert": [
        "je vous alerte sur la charge qui monte autour du sujet",
        "je vous remonte une tension de capacite sur le sujet",
    ],
    "delivery_risk": [
        "je partage le point qui peut faire glisser la livraison",
        "je vous ecris sur un risque de delai",
    ],
}

EVENT_LABELS = {
    "migration_auth_q1": "la migration auth du T1",
    "post_merger_hire_2024": "les recrutements post-fusion de 2024",
    "release_support_v3": "la mise en production support V3",
    "audit_iso_q2": "l'audit ISO du T2",
    "freeze_fin_mois": "le freeze de fin de mois",
    "bascule_tenant_avril": "la bascule tenant d'avril",
    "revue_habilitations_mai": "la revue des habilitations de mai",
    "go_live_reporting_ete": "le go-live reporting de l'ete",
    "incident_2042_sso": "l'incident SSO 2042",
    "incident_3011_facturation": "l'incident facturation 3011",
    "incident_1178_mars": "l'incident de mars 1178",
    "incident_4102_escalade": "l'incident d'escalade 4102",
    "incident_2217_sync": "l'incident de synchro 2217",
    "incident_8872_reporting": "l'incident reporting 8872",
    "incident_5570_partner": "l'incident partenaire 5570",
    "incident_6621_capacity": "l'incident capacite 6621",
}

PROJECT_LABELS = {
    "portail_client": "le portail client",
    "connecteur_x": "le connecteur X",
    "moteur_routing": "le moteur de routage",
    "fusion_tenant": "la fusion de tenants",
    "migration_sso": "la migration SSO",
    "consolidation_facturation": "la consolidation facturation",
    "refonte_workflow": "la refonte workflow",
    "socle_reporting": "le socle reporting",
    "passerelle_partenaire": "la passerelle partenaire",
    "pilotage_habilitations": "le pilotage des habilitations",
}

RARE_RESPONSIBILITY_TEXT = {
    "only_phd_under_30_in_team": "je suis la seule personne de l'equipe a cumuler ce perimetre et un doctorat",
    "owner_incident_unique": "je garde seule le suivi de l'incident le plus sensible du perimetre",
    "only_certified_on_connector": "je suis la seule personne certifiee sur ce connecteur",
}

JARGON_TEXT = {
    "support_ops": "le runbook d'escalade support",
    "ai_ops": "la file d'observabilite modele",
}

AGE_TEXT = {
    "20-24": "j'ai entre 20 et 24 ans",
    "25-29": "j'ai entre 25 et 29 ans",
    "30-34": "j'ai entre 30 et 34 ans",
    "35-39": "j'ai entre 35 et 39 ans",
    "40-44": "j'ai entre 40 et 44 ans",
    "40-49": "j'ai entre 40 et 49 ans",
    "45-49": "j'ai entre 45 et 49 ans",
    "50-54": "j'ai entre 50 et 54 ans",
    "55-59": "j'ai entre 55 et 59 ans",
    "60-64": "j'ai entre 60 et 64 ans",
}

SENSITIVE_TEXT = {
    "je reviens d'un arret maladie": "je reviens tout juste d'un arret maladie",
    "mon contexte culturel complique certains horaires": "mon contexte culturel me force a amenager certains horaires cette semaine",
    "je dois aussi tenir compte d'une contrainte religieuse cette semaine": "je dois aussi organiser mes disponibilites autour d'une contrainte religieuse cette semaine",
    "j'utilise des amenagements d'accessibilite": "je travaille avec quelques amenagements d'accessibilite",
    "je gere aussi une urgence familiale": "je gere aussi une urgence familiale en parallele",
    "je prefere rester discrete sur ma vie personnelle": "je prefere rester discrete sur ma vie personnelle, meme si cela joue sur mes horaires",
    "je suis en echange avec l'equipe juridique interne": "je suis deja en echange avec l'equipe juridique interne sur le sujet",
    "je suis aussi en difficulte financiere temporaire": "je traverse aussi une difficulte financiere temporaire en ce moment",
}

PROJECT_GOALS = {
    "integration_issue",
    "project_followup",
    "coordination",
    "context_share",
    "handover",
    "vendor_followup",
    "delivery_risk",
    "data_gap",
    "sync_delay",
}
EVENT_GOALS = {"incident_report", "urgent_incident"}
SENSITIVE_LABELS = {"HEALTH", "ETHNICITY", "RELIGION", "DISABILITY", "FAMILY_STATUS", "SEXUAL_ORIENTATION", "LEGAL", "FINANCIAL"}
SUPPORT_GREETING_TEMPLATES = [
    "Bonjour {recipient},",
    "Bonjour,",
    "Hello {recipient},",
]
SUPPORT_CLOSING_TEMPLATES = [
    "Merci pour votre aide.",
    "Merci d'avance pour votre retour.",
    "Je reste dispo si besoin.",
    "Bonne journee.",
]
SUPPORT_SUBJECT_PREFIX = ["Objet", "Sujet", "Ticket"]
SUPPORT_SUMMARY_TEMPLATES = [
    "Resume rapide :",
    "En bref :",
    "Point rapide :",
]
SUPPORT_ATTEMPT_TEMPLATES = [
    "J'ai deja relance le flux et relogge, mais ca revient.",
    "On a tente une reinit de session, sans effet.",
    "On a rejoue la synchro et le blocage reste la.",
    "J'ai verifie les traces de base, sans signal clair.",
]
WHEN_TEMPLATES = [
    "depuis ce matin",
    "depuis hier",
    "depuis le debut de semaine",
    "depuis la derniere mise a jour",
]
EMAIL_GREETING_TEMPLATES = [
    "Bonjour {recipient},",
    "Bonjour,",
    "Hello {recipient},",
]


@lru_cache(maxsize=1)
def _register_markers() -> Dict[str, Dict[str, List[str]]]:
    from atlas_anno.generation.style_sampler import load_style_factors

    return load_style_factors().get("register_markers", {})


def _register_templates(register: str, key: str, fallback: Sequence[str]) -> List[str]:
    """Templates de salutation/clôture keyés par registre ; fallback = templates v1."""
    markers = _register_markers().get(register or "courant", {})
    templates = markers.get(key)
    return list(templates) if templates else list(fallback)


# Remplacements ordonnés (les plus longs d'abord) pour le tutoiement des
# templates contrôlés — pas un conjugueur générique.
_TUTOIEMENT_REPLACEMENTS = [
    ("Pouvez-vous", "Peux-tu"),
    ("pouvez-vous", "peux-tu"),
    ("dites-moi", "dis-moi"),
    ("Dites-moi", "Dis-moi"),
    ("vous me confirmez", "tu me confirmes"),
    ("vous recommandez", "tu recommandes"),
    ("vous preferez", "tu preferes"),
    ("vous reprenez", "tu reprends"),
    ("vous voyez", "tu vois"),
    ("vous voulez", "tu veux"),
    ("vous avez", "tu as"),
    ("de votre cote", "de ton cote"),
    ("votre retour", "ton retour"),
    ("votre aide", "ton aide"),
    ("vous laisser", "te laisser"),
    ("Je vous", "Je te"),
    ("je vous", "je te"),
    ("si vous", "si tu"),
]


def _apply_address_form(text: str, scenario: ScenarioSpec) -> str:
    if scenario.address_form != "tu":
        return text
    for formal, informal in _TUTOIEMENT_REPLACEMENTS:
        text = text.replace(formal, informal)
    return text


def _signal_phrase(label: str, author: CharacterProfile, world: World) -> str:
    if label == "PERSON_NAME":
        return author.full_name
    if label == "EMAIL":
        return author.email
    if label == "PHONE":
        return author.phone
    if label == "USERNAME":
        return author.username
    if label == "ACCOUNT_ID":
        return author.account_id
    if label == "ORG_NAME_STRONG":
        return world.organization_name
    if label == "PROJECT_NAME_STRONG":
        return world.projects[0]
    if label == "ROLE":
        return author.role
    if label == "DEGREE":
        return author.degrees[0]
    if label == "AGE_RANGE":
        return f"tranche d'age {author.age_range}"
    if label == "TENURE":
        return f"{author.tenure_years} ans d'anciennete"
    if label == "TEAM":
        return author.team
    if label == "DEPARTMENT":
        return author.department
    if label == "LOCATION":
        return author.location
    if label == "NATIONALITY":
        return author.nationality
    if label == "CERTIFICATION":
        return author.certifications[0]
    if label == "SKILL_RARE":
        return author.skills[0]
    if label == "EVENT_DATE":
        return author.events[0]
    if label == "PRODUCT_CONTEXT":
        return world.products[0]
    if label == "RARE_RESPONSIBILITY":
        return author.rare_traits[0] if author.rare_traits else "responsabilite unique"
    if label == "SIGNATURE_PATTERN":
        return author.style_profile.signature_pattern
    if label == "JARGON_PATTERN":
        return author.style_profile.jargon_pattern
    if label in {"HEALTH", "ETHNICITY", "RELIGION", "DISABILITY", "FAMILY_STATUS", "SEXUAL_ORIENTATION", "LEGAL", "FINANCIAL"}:
        return {
            "HEALTH": "je reviens d'un arret maladie",
            "ETHNICITY": "mon contexte culturel complique certains horaires",
            "RELIGION": "je dois aussi tenir compte d'une contrainte religieuse cette semaine",
            "DISABILITY": "j'utilise des amenagements d'accessibilite",
            "FAMILY_STATUS": "je gere aussi une urgence familiale",
            "SEXUAL_ORIENTATION": "je prefere rester discrete sur ma vie personnelle",
            "LEGAL": "je suis en echange avec l'equipe juridique interne",
            "FINANCIAL": "je suis aussi en difficulte financiere temporaire",
        }[label]
    return label.lower()


def _seed_value(scenario: ScenarioSpec, scope: str) -> int:
    seed = f"{scope}|{scenario.scenario_id}|{scenario.recipient_role}|{scenario.document_goal}|{scenario.urgency}|{scenario.noise_level}"
    return sum(ord(char) for char in seed)


def _variant(options: Sequence[str], scenario: ScenarioSpec, scope: str) -> str:
    index = _seed_value(scenario, scope) % len(options)
    return options[index]


def _pick_option(options: Sequence[str], scenario: ScenarioSpec, scope: str) -> str:
    return options[_seed_value(scenario, scope) % len(options)]


def _pick_list(options: Sequence[Sequence[str | MentionPart]], scenario: ScenarioSpec, scope: str) -> List[str | MentionPart]:
    return list(options[_seed_value(scenario, scope) % len(options)])


def _pick_bool(scenario: ScenarioSpec, scope: str, threshold: int, modulo: int = 100) -> bool:
    return _seed_value(scenario, scope) % modulo < threshold


def _uses_project_context(scenario: ScenarioSpec) -> bool:
    return scenario.noise_level in {"medium", "high"} or scenario.document_goal in PROJECT_GOALS


def _uses_event_context(scenario: ScenarioSpec) -> bool:
    return scenario.noise_level == "high" or scenario.document_goal in EVENT_GOALS


def build_signal_values(scenario: ScenarioSpec, author: CharacterProfile, world: World) -> Dict[str, List[str]]:
    values: Dict[str, List[str]] = {}
    labels = list(scenario.required_signals)
    labels.extend(scenario.implicit_signals)
    labels.extend(["ORG_NAME_STRONG", "PRODUCT_CONTEXT"])
    if _uses_project_context(scenario):
        labels.append("PROJECT_NAME_STRONG")
    if _uses_event_context(scenario):
        labels.append("EVENT_DATE")
    if scenario.include_direct_identifiers:
        labels.extend(["PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID"])
    if scenario.include_sensitive:
        labels.extend(author.sensitive_attributes)
    for label in labels:
        value = _signal_phrase(label, author, world)
        values.setdefault(label, [])
        if value and value not in values[label]:
            values[label].append(value)
    return values


class _DocumentBuilder:
    def __init__(
        self,
        mode_by_label: Dict[str, str] | None = None,
        overrides: Dict[str, SurfaceOverride] | None = None,
    ) -> None:
        self._parts: List[str] = []
        self._grounding: List[GroundedMention] = []
        self._occurrence_by_snippet: Dict[str, int] = {}
        self._mode_by_label = mode_by_label or {}
        self._overrides = overrides or {}

    def add(self, *parts: str | MentionPart) -> None:
        for part in parts:
            if not part:
                continue
            if isinstance(part, tuple):
                label, canonical_value, snippet = part
                difficulty_mode = self._mode_by_label.get(label, "explicit_easy")
                cue_type = ""
                if difficulty_mode != "explicit_easy":
                    override = self._overrides.get(label)
                    if override is not None:
                        snippet = override.snippet
                        cue_type = override.cue_type
                    else:
                        # Pas de forme alternative disponible : la mention reste easy.
                        difficulty_mode = "explicit_easy"
                self._parts.append(snippet)
                occurrence = self._occurrence_by_snippet.get(snippet, 0) + 1
                self._occurrence_by_snippet[snippet] = occurrence
                self._grounding.append(
                    GroundedMention(
                        label=label,
                        canonical_value=canonical_value,
                        snippet=snippet,
                        occurrence_hint=occurrence,
                        difficulty_mode=difficulty_mode,
                        cue_type=cue_type,
                    )
                )
            else:
                self._parts.append(str(part))

    def line(self, *parts: str | MentionPart) -> None:
        self.add(*parts)
        self._parts.append("\n")

    def blank_line(self) -> None:
        if not self._parts:
            return
        if not self._parts[-1].endswith("\n"):
            self._parts.append("\n")
        if len(self._parts) < 2 or not self._parts[-2].endswith("\n"):
            self._parts.append("\n")

    def bullet(self, *parts: str | MentionPart) -> None:
        if not parts:
            return
        self.add("- ")
        self.add(*parts)
        self._parts.append("\n")

    def paragraph(self, *sentences: Sequence[str | MentionPart]) -> None:
        first = True
        for sentence in sentences:
            if not sentence:
                continue
            if not first:
                self.add(" ")
            self.add(*sentence)
            first = False
        self._parts.append("\n")

    def draft(self, notes: Sequence[str] | None = None) -> GeneratedTextDraft:
        text = "".join(self._parts).strip()
        return GeneratedTextDraft(text=text, notes=list(notes or []), grounding=list(self._grounding))


def _mention(label: str, canonical_value: str, snippet: str) -> MentionPart:
    return (label, canonical_value, snippet)


def _make_builder(scenario: ScenarioSpec, author: CharacterProfile, world: World) -> _DocumentBuilder:
    mode_by_label = {entry.label: entry.difficulty_mode for entry in scenario.mention_plan}
    overrides = build_surface_overrides(scenario, author, world)
    return _DocumentBuilder(mode_by_label=mode_by_label, overrides=overrides)


def _canonical_value(signal_values: Dict[str, List[str]], label: str, default: str = "") -> str:
    values = signal_values.get(label, [])
    return values[0] if values else default


def _recipient_label(recipient_role: str) -> str:
    return RECIPIENT_LABELS.get(recipient_role, recipient_role.replace("_", " "))


def _slug_words(value: str) -> str:
    return value.replace("_", " ").replace("-", " ")


def _humanize_event(value: str) -> str:
    if value in EVENT_LABELS:
        return EVENT_LABELS[value]
    if value.startswith("incident_"):
        parts = value.split("_")
        if len(parts) >= 3:
            return f"l'incident {' '.join(parts[2:])} {parts[1]}".replace("  ", " ")
    return _slug_words(value)


def _humanize_project(value: str) -> str:
    return PROJECT_LABELS.get(value, _slug_words(value))


def _humanize_degree(value: str) -> str:
    if value == "PhD":
        return "un doctorat"
    article = "une" if value.lower().endswith("e") else "un"
    return f"{article} {value}"


def _humanize_certification(value: str) -> str:
    formatted: List[str] = []
    for token in value.replace("-", " ").split():
        lowered = token.lower()
        if lowered == "aws":
            formatted.append("AWS")
        elif lowered == "ai":
            formatted.append("AI")
        elif lowered == "itil":
            formatted.append("ITIL")
        elif lowered == "okta":
            formatted.append("Okta")
        else:
            formatted.append(token.capitalize())
    return " ".join(formatted)


def _humanize_age_range(value: str) -> str:
    return AGE_TEXT.get(value, f"je suis dans la tranche {value}")


def _humanize_tenure(years: int) -> str:
    if years == 0:
        return "je viens de prendre le poste"
    if years == 1:
        return "je suis en poste depuis 1 an"
    return f"je suis en poste depuis {years} ans"


def _humanize_rare_responsibility(value: str) -> str:
    return RARE_RESPONSIBILITY_TEXT.get(value, _slug_words(value))


def _humanize_jargon(value: str) -> str:
    return JARGON_TEXT.get(value, _slug_words(value))


def _humanize_sensitive(value: str) -> str:
    return SENSITIVE_TEXT.get(value, value)


def _impact_line_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    world: World,
) -> List[str | MentionPart]:
    team_value = _canonical_value(signal_values, "TEAM", author.team)
    if scenario.urgency == "high":
        return [
            "C'est bloquant pour l'activite de ",
            _mention("TEAM", team_value, author.team),
            " depuis ce matin.",
        ]
    if scenario.urgency == "medium":
        return ["Le point ralentit clairement le traitement en cours."]
    return ["Ce n'est pas critique, mais j'aimerais le resoudre sans trop attendre."]


def _support_opening_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    world: World,
) -> List[str | MentionPart]:
    product_value = _canonical_value(signal_values, "PRODUCT_CONTEXT", world.products[0])
    project_value = _canonical_value(signal_values, "PROJECT_NAME_STRONG", world.projects[0])
    event_value = _canonical_value(signal_values, "EVENT_DATE", author.events[0])
    if scenario.document_goal == "integration_issue":
        return [
            _variant(SUPPORT_GOAL_TEXT["integration_issue"], scenario, "support-opening"),
            " ",
            _mention("PRODUCT_CONTEXT", product_value, world.products[0]),
            " et ",
            _mention("PROJECT_NAME_STRONG", project_value, _humanize_project(world.projects[0])),
            " reste bloquee.",
        ]
    if scenario.document_goal == "incident_report" and "EVENT_DATE" in signal_values:
        return [
            _variant(SUPPORT_GOAL_TEXT["incident_report"], scenario, "support-opening"),
            " ",
            _mention("PRODUCT_CONTEXT", product_value, world.products[0]),
            " depuis ",
            _mention("EVENT_DATE", event_value, _humanize_event(author.events[0])),
            ".",
        ]
    template = _variant(SUPPORT_GOAL_TEXT.get(scenario.document_goal, ["je vous contacte a propos de"]), scenario, "support-opening")
    return [template, " ", _mention("PRODUCT_CONTEXT", product_value, world.products[0]), "."]


def _support_greeting_line(scenario: ScenarioSpec, author: CharacterProfile) -> str:
    if author.style_profile.formality == "high":
        threshold = 85
    else:
        threshold = 65
    if not _pick_bool(scenario, "support-greeting-enabled", threshold):
        return ""
    templates = _register_templates(scenario.register, "support_greetings", SUPPORT_GREETING_TEMPLATES)
    template = _pick_option(templates, scenario, "support-greeting-line")
    return template.format(recipient=_recipient_label(scenario.recipient_role))


def _support_subject_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    world: World,
) -> List[str | MentionPart]:
    prefix = _pick_option(SUPPORT_SUBJECT_PREFIX, scenario, "support-subject-prefix")
    product_value = _canonical_value(signal_values, "PRODUCT_CONTEXT", world.products[0])
    project_value = _canonical_value(signal_values, "PROJECT_NAME_STRONG", world.projects[0])
    if _uses_project_context(scenario):
        return [f"{prefix}: ", _mention("PROJECT_NAME_STRONG", project_value, _humanize_project(project_value))]
    return [f"{prefix}: ", _mention("PRODUCT_CONTEXT", product_value, world.products[0])]


def _support_summary_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    world: World,
) -> List[str | MentionPart]:
    prefix = _pick_option(SUPPORT_SUMMARY_TEMPLATES, scenario, "support-summary-prefix")
    product_value = _canonical_value(signal_values, "PRODUCT_CONTEXT", world.products[0])
    return [prefix, " ", _mention("PRODUCT_CONTEXT", product_value, world.products[0]), "."]


def _support_attempt_parts(scenario: ScenarioSpec) -> List[str]:
    return [_pick_option(SUPPORT_ATTEMPT_TEMPLATES, scenario, "support-attempt")]


def _support_when_phrase(scenario: ScenarioSpec) -> str:
    return _pick_option(WHEN_TEMPLATES, scenario, "support-when")


def _support_status_parts(scenario: ScenarioSpec) -> List[str]:
    return [f"Le souci est visible {_support_when_phrase(scenario)}."]


def _support_identity_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    world: World,
) -> List[str | MentionPart]:
    role = _canonical_value(signal_values, "ROLE", author.role)
    team = _canonical_value(signal_values, "TEAM", author.team)
    organization = _canonical_value(signal_values, "ORG_NAME_STRONG", world.organization_name)
    templates: List[List[str | MentionPart]] = [
        [
            "Je suis ",
            _mention("ROLE", role, author.role),
            " dans l'equipe ",
            _mention("TEAM", team, author.team),
            " chez ",
            _mention("ORG_NAME_STRONG", organization, world.organization_name),
            ".",
        ],
        [
            "Cote ",
            _mention("TEAM", team, author.team),
            ", je suis ",
            _mention("ROLE", role, author.role),
            " chez ",
            _mention("ORG_NAME_STRONG", organization, world.organization_name),
            ".",
        ],
        [
            "Je travaille chez ",
            _mention("ORG_NAME_STRONG", organization, world.organization_name),
            " comme ",
            _mention("ROLE", role, author.role),
            " dans l'equipe ",
            _mention("TEAM", team, author.team),
            ".",
        ],
    ]
    return _pick_list(templates, scenario, "support-identity-line")


def _support_closing_line(scenario: ScenarioSpec, author: CharacterProfile) -> str:
    register_choices = _register_templates(scenario.register, "support_closings", [])
    if register_choices:
        choices = register_choices
    elif author.style_profile.formality == "high":
        choices = SUPPORT_CLOSING_TEMPLATES[:2] + SUPPORT_CLOSING_TEMPLATES[3:]
    else:
        choices = SUPPORT_CLOSING_TEMPLATES
    return _apply_address_form(_pick_option(choices, scenario, "support-closing"), scenario)


def _email_greeting_line(scenario: ScenarioSpec) -> str:
    templates = _register_templates(scenario.register, "email_greetings", EMAIL_GREETING_TEMPLATES)
    template = _pick_option(templates, scenario, "email-greeting-line")
    return template.format(recipient=_recipient_label(scenario.recipient_role))


def _email_opening_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    world: World,
) -> List[str | MentionPart]:
    product_value = _canonical_value(signal_values, "PRODUCT_CONTEXT", world.products[0])
    project_value = _canonical_value(signal_values, "PROJECT_NAME_STRONG", world.projects[0])
    event_value = _canonical_value(signal_values, "EVENT_DATE", author.events[0])
    if scenario.document_goal in PROJECT_GOALS and "PROJECT_NAME_STRONG" in signal_values:
        return [
            _variant(EMAIL_GOAL_TEXT.get(scenario.document_goal, ["je vous ecris a propos de"]), scenario, "email-opening"),
            " sur ",
            _mention("PROJECT_NAME_STRONG", project_value, _humanize_project(world.projects[0])),
            " cote ",
            _mention("PRODUCT_CONTEXT", product_value, world.products[0]),
            ".",
        ]
    if scenario.document_goal == "urgent_incident" and "EVENT_DATE" in signal_values:
        return [
            _variant(EMAIL_GOAL_TEXT["urgent_incident"], scenario, "email-opening"),
            " ",
            _mention("PRODUCT_CONTEXT", product_value, world.products[0]),
            " depuis ",
            _mention("EVENT_DATE", event_value, _humanize_event(author.events[0])),
            ".",
        ]
    template = _variant(EMAIL_GOAL_TEXT.get(scenario.document_goal, ["je vous ecris a propos de"]), scenario, "email-opening")
    return [template, " ", _mention("PRODUCT_CONTEXT", product_value, world.products[0]), "."]


def _goal_detail_line_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    world: World,
) -> List[str | MentionPart]:
    project_value = _canonical_value(signal_values, "PROJECT_NAME_STRONG", world.projects[0])
    team_value = _canonical_value(signal_values, "TEAM", author.team)
    if scenario.document_goal == "billing_mismatch":
        return ["Le volume et le suivi ne tombent plus juste sur la partie facturation."]
    if scenario.document_goal == "permission_review":
        return [
            "Le circuit de validation des acces reste bloque pour ",
            _mention("TEAM", team_value, author.team),
            ".",
        ]
    if scenario.document_goal == "data_gap":
        return ["Une partie des lignes manque encore dans le reporting attendu."]
    if scenario.document_goal == "sync_delay":
        return ["La synchro repart puis reprend du retard apres quelques minutes."]
    if scenario.document_goal == "handover":
        return ["Je rassemble ici les derniers points utiles avant reprise du dossier."]
    if scenario.document_goal == "vendor_followup":
        return ["Il reste surtout une dependance externe qui bloque la suite du sujet."]
    if scenario.document_goal == "capacity_alert":
        return ["La file d'attente monte plus vite que notre capacite de traitement."]
    if scenario.document_goal == "delivery_risk":
        return [
            "Si on ne tranche pas vite, ",
            _mention("PROJECT_NAME_STRONG", project_value, _humanize_project(world.projects[0])),
            " risque de glisser.",
        ]
    return []


def _noise_line_parts(
    scenario: ScenarioSpec,
    signal_values: Dict[str, List[str]],
    world: World,
    author: CharacterProfile,
) -> List[str | MentionPart]:
    project_value = _canonical_value(signal_values, "PROJECT_NAME_STRONG", world.projects[0])
    event_value = _canonical_value(signal_values, "EVENT_DATE", author.events[0])
    if scenario.noise_level == "high":
        return [
            "J'ai deja refait un passage sur ",
            _mention("PROJECT_NAME_STRONG", project_value, _humanize_project(world.projects[0])),
            " et compare avec ",
            _mention("EVENT_DATE", event_value, _humanize_event(author.events[0])),
            ", sans voir de difference nette.",
        ]
    if scenario.noise_level == "medium":
        return [
            "Le sujet semble surtout se jouer sur ",
            _mention("PROJECT_NAME_STRONG", project_value, _humanize_project(world.projects[0])),
            ".",
        ]
    return []


def _support_request_line(scenario: ScenarioSpec, recipient_role: str) -> str:
    if scenario.document_goal == "permission_review":
        line = "Pouvez-vous me dire quelle validation manque encore et qui doit reprendre la main ?"
    elif scenario.document_goal == "billing_mismatch":
        line = "Pouvez-vous verifier la piste facturation et me dire quelle reference reprendre ?"
    elif scenario.document_goal == "sync_delay":
        line = "Pouvez-vous confirmer ou la synchro prend du retard et quel contournement reste viable ?"
    elif recipient_role in {"customer_support", "billing_support", "partner_support"}:
        if scenario.urgency == "high":
            line = "Pouvez-vous me dire rapidement si vous voyez le blocage de votre cote et quelle action lancer en priorite ?"
        else:
            line = "Pouvez-vous me confirmer la marche a suivre ou la verif la plus utile de votre cote ?"
    elif scenario.urgency == "high":
        line = "Pouvez-vous verifier les traces prioritaires ou me dire quel contournement vous recommandez ?"
    else:
        line = "Pouvez-vous verifier le dossier et me dire ce qu'il vous manque pour avancer ?"
    return _apply_address_form(line, scenario)


def _email_next_step_line(scenario: ScenarioSpec, recipient_role: str, connector: str) -> str:
    if scenario.document_goal == "delivery_risk":
        line = f"{connector}, dites-moi s'il faut securiser le perimetre maintenant ou deplacer la prochaine jalon."
    elif scenario.document_goal == "handover":
        line = f"{connector}, je peux vous laisser un recap plus detaille si vous reprenez le sujet aujourd'hui."
    elif scenario.document_goal == "vendor_followup":
        line = f"{connector}, je peux preparer la relance fournisseur si vous me confirmez le bon angle."
    elif recipient_role in {"client_success", "vendor_contact"}:
        line = f"{connector}, dites-moi si vous preferez un point rapide aujourd'hui ou une reponse par retour de mail."
    elif scenario.urgency == "high":
        line = f"{connector}, je prends tout de suite le prochain point si vous avez besoin d'un complement."
    else:
        line = f"{connector}, je peux completer le contexte si vous voulez que je creuse un point en plus."
    return _apply_address_form(line, scenario)


def _signature_snippet(author: CharacterProfile, include_direct_identifiers: bool) -> Tuple[str, List[str | MentionPart]]:
    if author.style_profile.signature_pattern == "full_signature":
        snippet = "Bien cordialement,"
        if include_direct_identifiers:
            lines: List[str | MentionPart] = [
                "\n",
                _mention("PERSON_NAME", author.full_name, author.full_name),
                "\n",
                _mention("ROLE", author.role, author.role),
                "\n",
                _mention("EMAIL", author.email, author.email),
            ]
        else:
            lines = [snippet]
        return snippet, lines

    snippet = "Merci d'avance,"
    if include_direct_identifiers:
        lines = ["\n", _mention("PERSON_NAME", author.full_name, author.full_name)]
    else:
        lines = [snippet]
    return snippet, lines


def _profile_line(builder: _DocumentBuilder, signal_values: Dict[str, List[str]], author: CharacterProfile, world: World) -> None:
    role = _canonical_value(signal_values, "ROLE", author.role)
    team = _canonical_value(signal_values, "TEAM", author.team)
    organization = _canonical_value(signal_values, "ORG_NAME_STRONG", world.organization_name)
    builder.line(
        "Je travaille comme ",
        _mention("ROLE", role, author.role),
        " dans l'equipe ",
        _mention("TEAM", team, author.team),
        " chez ",
        _mention("ORG_NAME_STRONG", organization, world.organization_name),
        ".",
    )


def _experience_line(builder: _DocumentBuilder, signal_values: Dict[str, List[str]], author: CharacterProfile) -> None:
    parts: List[str | MentionPart] = []
    age_value = _canonical_value(signal_values, "AGE_RANGE")
    tenure_value = _canonical_value(signal_values, "TENURE")
    if age_value:
        parts.extend(
            [
                _mention("AGE_RANGE", age_value, _humanize_age_range(author.age_range)),
                "",
            ]
        )
    if age_value and tenure_value:
        parts.append(", et ")
    if tenure_value:
        parts.append(_mention("TENURE", tenure_value, _humanize_tenure(author.tenure_years)))
    if parts:
        builder.line("Pour situer le contexte, ", *parts, ".")


def _expertise_line(builder: _DocumentBuilder, signal_values: Dict[str, List[str]], author: CharacterProfile) -> None:
    degree_value = _canonical_value(signal_values, "DEGREE")
    certification_value = _canonical_value(signal_values, "CERTIFICATION")
    rare_value = _canonical_value(signal_values, "RARE_RESPONSIBILITY")
    event_value = _canonical_value(signal_values, "EVENT_DATE")
    if degree_value:
        builder.line(
            "Je reste la personne qui porte le sujet avec ",
            _mention("DEGREE", degree_value, _humanize_degree(degree_value)),
            ".",
        )
    if certification_value:
        builder.line(
            "Je suis aussi la personne reference cote certification ",
            _mention("CERTIFICATION", certification_value, _humanize_certification(certification_value)),
            ".",
        )
    if rare_value:
        if event_value:
            builder.line(
                "Depuis ",
                _mention("EVENT_DATE", event_value, _humanize_event(event_value)),
                ", ",
                _mention("RARE_RESPONSIBILITY", rare_value, _humanize_rare_responsibility(rare_value)),
                ".",
            )
        else:
            builder.line(
                "Depuis ",
                _humanize_event(author.events[0]),
                ", ",
                _mention("RARE_RESPONSIBILITY", rare_value, _humanize_rare_responsibility(rare_value)),
                ".",
            )


def _style_line(builder: _DocumentBuilder, signal_values: Dict[str, List[str]], connector: str) -> None:
    jargon_value = _canonical_value(signal_values, "JARGON_PATTERN")
    if jargon_value:
        builder.line(
            connector.capitalize(),
            ", dans notre jargon on parle plutot de ce cas comme ",
            _mention("JARGON_PATTERN", jargon_value, _humanize_jargon(jargon_value)),
            " pour ce cas-la.",
        )


def _contact_line(
    builder: _DocumentBuilder,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    scenario: ScenarioSpec,
) -> None:
    if not all(label in signal_values for label in ("PERSON_NAME", "EMAIL", "PHONE", "USERNAME", "ACCOUNT_ID")):
        return
    variants: List[List[str | MentionPart]] = [
        [
            "Je suis ",
            _mention("PERSON_NAME", author.full_name, author.full_name),
            ". Vous pouvez me joindre sur ",
            _mention("EMAIL", author.email, author.email),
            " ou au ",
            _mention("PHONE", author.phone, author.phone),
            ". Mon login est ",
            _mention("USERNAME", author.username, f"login {author.username}"),
            " et le compte concerne est ",
            _mention("ACCOUNT_ID", author.account_id, author.account_id),
            ".",
        ],
        [
            "Contact: ",
            _mention("PERSON_NAME", author.full_name, author.full_name),
            ", ",
            _mention("EMAIL", author.email, author.email),
            ", ",
            _mention("PHONE", author.phone, author.phone),
            ". Login ",
            _mention("USERNAME", author.username, f"login {author.username}"),
            ", compte ",
            _mention("ACCOUNT_ID", author.account_id, author.account_id),
            ".",
        ],
    ]
    builder.line(*_pick_list(variants, scenario, "contact-line"))


def _sensitive_line(builder: _DocumentBuilder, signal_values: Dict[str, List[str]], author: CharacterProfile) -> None:
    if not author.sensitive_attributes:
        return
    sensitive_label = author.sensitive_attributes[0]
    if sensitive_label not in signal_values:
        return
    value = _canonical_value(signal_values, sensitive_label)
    builder.line(
        "Je prefere aussi signaler que ",
        _mention(sensitive_label, value, _humanize_sensitive(value)),
        ".",
    )


def _surface_snippets_for_label(label: str, canonical_value: str, author: CharacterProfile) -> List[str]:
    if label in {"PERSON_NAME", "EMAIL", "PHONE", "ACCOUNT_ID", "ROLE", "TEAM", "ORG_NAME_STRONG", "PRODUCT_CONTEXT"}:
        return [canonical_value]
    if label == "USERNAME":
        return [f"login {canonical_value}", canonical_value]
    if label == "PROJECT_NAME_STRONG":
        return [canonical_value, _humanize_project(canonical_value)]
    if label == "EVENT_DATE":
        return [canonical_value, _humanize_event(canonical_value)]
    if label == "AGE_RANGE":
        return [_humanize_age_range(author.age_range)]
    if label == "TENURE":
        return [_humanize_tenure(author.tenure_years)]
    if label == "DEGREE":
        return [_humanize_degree(canonical_value)]
    if label == "CERTIFICATION":
        return [_humanize_certification(canonical_value)]
    if label == "RARE_RESPONSIBILITY":
        return [_humanize_rare_responsibility(canonical_value)]
    if label == "JARGON_PATTERN":
        return [_humanize_jargon(canonical_value)]
    if label == "SIGNATURE_PATTERN":
        return ["Bien cordialement," if canonical_value == "full_signature" else "Merci d'avance,"]
    if label in SENSITIVE_LABELS:
        return [_humanize_sensitive(canonical_value)]
    return [canonical_value]


def _complete_grounding(
    draft: GeneratedTextDraft,
    signal_values: Dict[str, List[str]],
    author: CharacterProfile,
    scenario: ScenarioSpec | None = None,
    world: World | None = None,
) -> GeneratedTextDraft:
    grounding = list(draft.grounding)
    seen = {(mention.label, mention.snippet, mention.occurrence_hint) for mention in grounding}
    mode_by_label = {entry.label: entry.difficulty_mode for entry in scenario.mention_plan} if scenario else {}
    overrides = build_surface_overrides(scenario, author, world) if scenario and world else {}
    for label, values in signal_values.items():
        for canonical_value in values:
            for snippet in dict.fromkeys(_surface_snippets_for_label(label, canonical_value, author)):
                if not snippet:
                    continue
                for occurrence_hint, _ in enumerate(find_occurrences(draft.text, snippet), start=1):
                    key = (label, snippet, occurrence_hint)
                    if key in seen:
                        continue
                    difficulty_mode = "explicit_easy"
                    cue_type = ""
                    override = overrides.get(label)
                    if override is not None and snippet == override.snippet:
                        difficulty_mode = mode_by_label.get(label, "explicit_easy")
                        cue_type = override.cue_type
                    grounding.append(
                        GroundedMention(
                            label=label,
                            canonical_value=canonical_value,
                            snippet=snippet,
                            occurrence_hint=occurrence_hint,
                            difficulty_mode=difficulty_mode,
                            cue_type=cue_type,
                        )
                    )
                    seen.add(key)
    return GeneratedTextDraft(text=draft.text, notes=list(draft.notes), grounding=grounding)


def _support_draft(
    scenario: ScenarioSpec,
    author: CharacterProfile,
    world: World,
    signal_values: Dict[str, List[str]],
) -> GeneratedTextDraft:
    builder = _make_builder(scenario, author, world)
    connector = author.style_profile.favorite_connectors[0] if author.style_profile.favorite_connectors else "a ce stade"
    layout = _pick_option(
        ["compact", "paragraph", "bullets", "context_first", "threadlike"],
        scenario,
        "support-layout",
    )
    greeting = _support_greeting_line(scenario, author)
    use_blank_lines = _pick_bool(scenario, "support-blank-lines", 35 if author.style_profile.formality == "high" else 20)
    include_subject = _pick_bool(scenario, "support-subject", 25)
    include_attempts = _pick_bool(scenario, "support-attempts", 35 if scenario.noise_level == "low" else 55)
    include_status = _pick_bool(scenario, "support-status", 45)

    opening = _support_opening_parts(scenario, signal_values, author, world)
    detail = _goal_detail_line_parts(scenario, signal_values, author, world)
    impact = _impact_line_parts(scenario, signal_values, author, world)
    noise = _noise_line_parts(scenario, signal_values, world, author)
    identity = _support_identity_parts(scenario, signal_values, author, world)
    summary = _support_summary_parts(scenario, signal_values, world)
    status = _support_status_parts(scenario) if include_status else []

    if greeting:
        builder.line(greeting)
    if include_subject:
        builder.line(*_support_subject_parts(scenario, signal_values, world))
    if use_blank_lines:
        builder.blank_line()

    if layout == "compact":
        builder.paragraph(opening, detail, identity, status or None, impact)
        if noise:
            builder.line(*noise)
        if include_attempts:
            builder.line(*_support_attempt_parts(scenario))
    elif layout == "paragraph":
        builder.paragraph(opening, identity, detail, impact, noise or None, status or None)
        if include_attempts and not noise:
            builder.line(*_support_attempt_parts(scenario))
    elif layout == "bullets":
        builder.line(*opening)
        builder.line(_pick_option(["Points rapides :", "Concretement :", "Recap rapide :"], scenario, "support-bullet-intro"))
        if detail:
            builder.bullet(*detail)
        else:
            builder.bullet(*summary)
        builder.bullet(*impact)
        if status:
            builder.bullet(*status)
        if noise:
            builder.bullet(*noise)
        if include_attempts:
            builder.bullet(*_support_attempt_parts(scenario))
        builder.line(*identity)
    elif layout == "context_first":
        builder.line(*identity)
        _experience_line(builder, signal_values, author)
        builder.line(*opening)
        if detail:
            builder.line(*detail)
        builder.line(*impact)
        if status:
            builder.line(*status)
        if noise:
            builder.line(*noise)
    else:
        builder.line(*opening)
        if detail:
            builder.line(*detail)
        if use_blank_lines:
            builder.blank_line()
        builder.line("Rebonjour,")
        builder.paragraph(identity, impact, status or None)
        if noise:
            builder.line(*noise)
        if include_attempts:
            builder.line(*_support_attempt_parts(scenario))

    # Les lignes ci-dessous s'auto-gatent sur signal_values (labels requis) :
    # elles doivent toujours être tentées, sinon l'audit missing_surface_label échoue.
    if layout != "context_first":
        _experience_line(builder, signal_values, author)
    _expertise_line(builder, signal_values, author)
    _style_line(builder, signal_values, connector)
    _sensitive_line(builder, signal_values, author)
    builder.line(_support_request_line(scenario, scenario.recipient_role))
    if scenario.include_direct_identifiers:
        _contact_line(builder, signal_values, author, scenario)
    closing = _support_closing_line(scenario, author)
    if "PERSON_NAME" in signal_values and _pick_bool(scenario, "support-sign-name", 55):
        builder.line(closing, " ", _mention("PERSON_NAME", author.full_name, author.full_name))
    else:
        builder.line(closing)
    return builder.draft(notes=[f"recipient:{scenario.recipient_role}", f"unit:{scenario.unit_type}"])


def _email_single_message_draft(
    scenario: ScenarioSpec,
    author: CharacterProfile,
    world: World,
    signal_values: Dict[str, List[str]],
) -> GeneratedTextDraft:
    builder = _make_builder(scenario, author, world)
    connector = author.style_profile.favorite_connectors[0] if author.style_profile.favorite_connectors else "a ce stade"
    layout = _pick_option(["standard", "context_first", "compact"], scenario, "email-layout")
    use_blank_lines = _pick_bool(scenario, "email-blank-lines", 45)
    builder.line(_email_greeting_line(scenario))
    if use_blank_lines:
        builder.blank_line()
    opening = _email_opening_parts(scenario, signal_values, author, world)
    goal_detail = _goal_detail_line_parts(scenario, signal_values, author, world)
    impact = _impact_line_parts(scenario, signal_values, author, world)
    noise_parts = _noise_line_parts(scenario, signal_values, world, author)

    if layout == "context_first":
        _profile_line(builder, signal_values, author, world)
        _experience_line(builder, signal_values, author)
        builder.line(*opening)
    elif layout == "compact":
        builder.paragraph(opening, goal_detail or None, impact)
        _profile_line(builder, signal_values, author, world)
    else:
        builder.line(*opening)
        if goal_detail:
            builder.line(*goal_detail)
        _profile_line(builder, signal_values, author, world)

    # Lignes auto-gatées sur signal_values : toujours tentées pour garantir la
    # couverture des labels requis, quel que soit le layout.
    if layout != "context_first":
        _experience_line(builder, signal_values, author)
    _expertise_line(builder, signal_values, author)
    if layout != "compact" and noise_parts:
        builder.line(*noise_parts)
    _sensitive_line(builder, signal_values, author)
    if layout != "compact":
        builder.line(*impact)
    builder.line(_email_next_step_line(scenario, scenario.recipient_role, connector))
    if scenario.include_direct_identifiers:
        _contact_line(builder, signal_values, author, scenario)
    if scenario.include_signature:
        builder.blank_line()
        signature_value = _canonical_value(signal_values, "SIGNATURE_PATTERN", author.style_profile.signature_pattern)
        snippet, signature_lines = _signature_snippet(author, scenario.include_direct_identifiers)
        builder.add(_mention("SIGNATURE_PATTERN", signature_value, snippet))
        if signature_lines[0] != snippet:
            builder.add(*signature_lines)
        else:
            for line in signature_lines[1:]:
                builder.add(line)
        builder.line()
    return builder.draft(notes=[f"recipient:{scenario.recipient_role}", f"unit:{scenario.unit_type}"])


def _email_thread_short_draft(
    scenario: ScenarioSpec,
    author: CharacterProfile,
    world: World,
    signal_values: Dict[str, List[str]],
) -> GeneratedTextDraft:
    builder = _make_builder(scenario, author, world)
    connector = author.style_profile.favorite_connectors[0] if author.style_profile.favorite_connectors else "a ce stade"
    use_blank_lines = _pick_bool(scenario, "email-thread-blank-lines", 60)
    builder.line(_email_greeting_line(scenario))
    if use_blank_lines:
        builder.blank_line()
    builder.line(*_email_opening_parts(scenario, signal_values, author, world))
    goal_detail = _goal_detail_line_parts(scenario, signal_values, author, world)
    if goal_detail:
        builder.line(*goal_detail)
    _profile_line(builder, signal_values, author, world)
    builder.line(*_impact_line_parts(scenario, signal_values, author, world))
    if use_blank_lines:
        builder.blank_line()
    builder.line(_pick_option(["Rebonjour,", "Rebonjour a tous,", "Petit ajout :"], scenario, "email-thread-followup"))
    if use_blank_lines:
        builder.blank_line()
    _experience_line(builder, signal_values, author)
    _expertise_line(builder, signal_values, author)
    _style_line(builder, signal_values, connector)
    _sensitive_line(builder, signal_values, author)
    noise_parts = _noise_line_parts(scenario, signal_values, world, author)
    if noise_parts:
        builder.line(*noise_parts)
    if scenario.include_direct_identifiers:
        _contact_line(builder, signal_values, author, scenario)
    builder.line(_email_next_step_line(scenario, scenario.recipient_role, connector))
    if scenario.include_signature:
        builder.blank_line()
        signature_value = _canonical_value(signal_values, "SIGNATURE_PATTERN", author.style_profile.signature_pattern)
        snippet, signature_lines = _signature_snippet(author, scenario.include_direct_identifiers)
        builder.add(_mention("SIGNATURE_PATTERN", signature_value, snippet))
        if signature_lines[0] != snippet:
            builder.add(*signature_lines)
        else:
            for line in signature_lines[1:]:
                builder.add(line)
        builder.line()
    return builder.draft(notes=[f"recipient:{scenario.recipient_role}", f"unit:{scenario.unit_type}"])


def compose_document_draft(
    scenario: ScenarioSpec,
    author: CharacterProfile,
    world: World,
    signal_values: Dict[str, List[str]],
) -> GeneratedTextDraft:
    if scenario.domain == "support_ticket":
        return _support_draft(scenario, author, world, signal_values)
    if scenario.unit_type == "thread_short":
        return _email_thread_short_draft(scenario, author, world, signal_values)
    return _email_single_message_draft(scenario, author, world, signal_values)


def build_documents(
    worlds: List[World],
    characters: List[CharacterProfile],
    scenarios: List[ScenarioSpec],
    candidate_pools: Dict[str, CandidatePools],
) -> List[DocumentRecord]:
    world_by_org = {world.organization_id: world for world in worlds}
    character_by_id = {character.person_id: character for character in characters}
    documents: List[DocumentRecord] = []
    for index, scenario in enumerate(scenarios):
        author = character_by_id[scenario.author_id]
        world = world_by_org[author.organization_id]
        signal_values = build_signal_values(scenario, author, world)
        draft = _complete_grounding(
            compose_document_draft(scenario, author, world, signal_values),
            signal_values,
            author,
            scenario,
            world,
        )
        documents.append(
            DocumentRecord(
                doc_id=f"doc_{index + 1:06d}",
                domain=scenario.domain,
                unit_type=scenario.unit_type,
                language=scenario.language,
                author_id=author.person_id,
                target_person_ids=[author.person_id],
                world_id=world.world_id,
                split=scenario.split,
                text=draft.text,
                scenario=scenario,
                candidate_pools=candidate_pools[author.person_id],
                annotations=AnnotationBundle(),
                metadata={
                    "difficulty": scenario.difficulty,
                    "register": scenario.register,
                    "address_form": scenario.address_form,
                    "signal_values": signal_values,
                    "surface_grounding": [mention.__dict__ for mention in draft.grounding],
                    "text_notes": list(draft.notes),
                    "world_name": world.organization_name,
                    "mention_difficulty": dict(
                        Counter(mention.difficulty_mode for mention in draft.grounding)
                    ),
                },
            )
        )
    return documents
