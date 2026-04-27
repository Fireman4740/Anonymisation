from __future__ import annotations

import unittest

from atlas_anno.schemas import CharacterProfile, StyleProfile


class SchemasTest(unittest.TestCase):
    def test_character_profile_validation(self) -> None:
        with self.assertRaises(ValueError):
            CharacterProfile(
                person_id="",
                full_name="Name",
                email="a@example.test",
                phone="+33 6 00 00 00",
                username="name",
                account_id="ACC-01-0001",
                language="fr",
                country="France",
                location="Paris",
                age_range="25-29",
                gender="female",
                nationality="France",
                organization_id="org_01",
                department="AI Solutions",
                team="LLM Ops",
                role="AI Support Engineer",
                seniority="mid",
                tenure_years=1,
                degrees=["Master"],
                skills=["python"],
                certifications=["azure-ai-architect"],
                rare_traits=[],
                events=["incident_1178"],
                sensitive_attributes=[],
                style_profile=StyleProfile(
                    formality="medium",
                    signature_pattern="thanks_name",
                    verbosity="short",
                    emoji_usage="none",
                    favorite_connectors=["donc"],
                    jargon_pattern="support_ops",
                ),
            )

