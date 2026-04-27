from __future__ import annotations

import random
from typing import List

from atlas_anno.schemas import World


COMPANIES = [
    "Atlas Services",
    "Meridian Cloud",
    "Nordic Signals",
    "Hexa Support",
    "Blue Orbit Systems",
    "NorthBridge Data",
    "Signal Harbor",
    "Vertex Care",
    "Asteria Connect",
    "Lumen Grid",
]
DEPARTMENTS = [
    "AI Solutions",
    "Customer Care",
    "IT Operations",
    "HR Services",
    "Platform Engineering",
    "Billing Operations",
    "Identity Programs",
    "Analytics Delivery",
    "Partner Success",
]
TEAMS = [
    "LLM Ops",
    "Escalade Support",
    "Identity Access",
    "People Ops",
    "Data Platform",
    "Billing Control",
    "Partner Care",
    "Release Coordination",
    "Trust Operations",
    "Reporting Hub",
    "Tenant Migration",
    "Workflow Automation",
]
PROJECTS = [
    "connecteur_x",
    "portail_client",
    "moteur_routing",
    "fusion_tenant",
    "migration_sso",
    "consolidation_facturation",
    "refonte_workflow",
    "socle_reporting",
    "passerelle_partenaire",
    "pilotage_habilitations",
]
PRODUCTS = [
    "AtlasDesk",
    "SignalHub",
    "MergeFlow",
    "OpsConsole",
    "PulseBoard",
    "AccessBridge",
    "LedgerSync",
    "CasePilot",
    "TenantView",
    "RouteMonitor",
]
INCIDENTS = [
    "incident_1178_mars",
    "incident_2042_sso",
    "incident_3011_facturation",
    "incident_4102_escalade",
    "incident_2217_sync",
    "incident_8872_reporting",
    "incident_5570_partner",
    "incident_6621_capacity",
]
EVENTS = [
    "post_merger_hire_2024",
    "audit_iso_q2",
    "migration_auth_q1",
    "release_support_v3",
    "freeze_fin_mois",
    "bascule_tenant_avril",
    "revue_habilitations_mai",
    "go_live_reporting_ete",
]


def build_worlds(count: int, seed: int = 13, language: str = "fr") -> List[World]:
    rng = random.Random(seed)
    worlds: List[World] = []
    for index in range(count):
        organization_id = f"org_{index + 1:02d}"
        worlds.append(
            World(
                world_id=f"world_{index + 1:02d}",
                language=language,
                organization_id=organization_id,
                organization_name=COMPANIES[index % len(COMPANIES)],
                departments=rng.sample(DEPARTMENTS, k=min(4, len(DEPARTMENTS))),
                teams=rng.sample(TEAMS, k=min(4, len(TEAMS))),
                projects=rng.sample(PROJECTS, k=min(4, len(PROJECTS))),
                products=rng.sample(PRODUCTS, k=min(3, len(PRODUCTS))),
                incidents=rng.sample(INCIDENTS, k=min(3, len(INCIDENTS))),
                calendar_events=rng.sample(EVENTS, k=min(3, len(EVENTS))),
            )
        )
    return worlds
