from __future__ import annotations

import unittest
from collections import Counter

from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.scenario_planner import (
    EXTERNAL_RECIPIENTS,
    build_scenarios,
    effective_register,
)
from atlas_anno.generation.style_sampler import StyleFactorSampler, load_style_factors
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.records import style_profile_from_dict
from atlas_anno.schemas import StyleProfile


class StyleFactorSamplerTest(unittest.TestCase):
    def test_sampling_is_deterministic_per_counter(self) -> None:
        sampler = StyleFactorSampler(seed=27)
        first = sampler.sample(5, "mid", "France")
        sampler.sample(3, "lead", "Belgique")
        again = sampler.sample(5, "mid", "France")
        self.assertEqual(first, again)

    def test_distribution_follows_weights(self) -> None:
        sampler = StyleFactorSampler(seed=7)
        factors = load_style_factors()
        counts = Counter(
            sampler.sample(counter, "mid", "France").register for counter in range(2000)
        )
        for register, weight in factors["registers"].items():
            observed = counts.get(register, 0) / 2000
            self.assertLess(abs(observed - weight), 0.05, f"{register}: {observed} vs {weight}")

    def test_soutenu_never_tutoie(self) -> None:
        sampler = StyleFactorSampler(seed=11)
        for counter in range(500):
            profile = sampler.sample(counter, "senior", "France")
            if profile.register == "soutenu":
                self.assertEqual(profile.address_form, "vous")

    def test_variety_derived_from_nationality(self) -> None:
        sampler = StyleFactorSampler(seed=13)
        self.assertEqual(sampler.sample(1, "mid", "Canada").francophone_variety, "quebec")
        self.assertEqual(sampler.sample(2, "mid", "Suisse").francophone_variety, "suisse")
        self.assertEqual(sampler.sample(3, "mid", "Inconnu").francophone_variety, "metropole")

    def test_connectors_respect_register_taboos(self) -> None:
        sampler = StyleFactorSampler(seed=17)
        taboos = load_style_factors()["register_taboos"]
        for counter in range(500):
            profile = sampler.sample(counter, "mid", "France")
            for taboo in taboos.get(profile.register, []):
                for connector in profile.favorite_connectors:
                    self.assertNotIn(taboo.lower(), connector.lower())


class EffectiveRegisterTest(unittest.TestCase):
    def _author_with(self, register: str, address_form: str):
        worlds = build_worlds(count=1)
        author = build_characters(worlds, per_world=1)[0]
        author.style_profile.register = register
        author.style_profile.address_form = address_form
        return author

    def test_familier_demoted_for_external_recipient(self) -> None:
        author = self._author_with("familier", "tu")
        external = next(iter(EXTERNAL_RECIPIENTS))
        register, address_form = effective_register(author, external)
        self.assertEqual((register, address_form), ("courant", "vous"))

    def test_familier_kept_for_internal_recipient(self) -> None:
        author = self._author_with("familier", "tu")
        register, address_form = effective_register(author, "service_desk")
        self.assertEqual((register, address_form), ("familier", "tu"))

    def test_soutenu_forces_vous(self) -> None:
        author = self._author_with("soutenu", "tu")
        register, address_form = effective_register(author, "service_desk")
        self.assertEqual((register, address_form), ("soutenu", "vous"))

    def test_scenarios_carry_effective_values(self) -> None:
        worlds = build_worlds(count=2)
        characters = build_characters(worlds, per_world=6)
        scenarios = build_scenarios(characters, documents=24)
        for scenario in scenarios:
            self.assertIn(scenario.register, {"familier", "courant", "soutenu"})
            self.assertIn(scenario.address_form, {"tu", "vous"})
            if scenario.register == "soutenu":
                self.assertEqual(scenario.address_form, "vous")


class StyleProfileCompatTest(unittest.TestCase):
    def test_v2_payload_loads_with_defaults(self) -> None:
        payload = {
            "formality": "medium",
            "signature_pattern": "thanks_name",
            "verbosity": "short",
            "emoji_usage": "none",
            "favorite_connectors": ["donc"],
            "jargon_pattern": "support_ops",
        }
        profile = style_profile_from_dict(payload)
        self.assertEqual(profile.register, "courant")
        self.assertEqual(profile.address_form, "vous")

    def test_future_payload_with_unknown_fields_loads(self) -> None:
        payload = {
            "formality": "medium",
            "signature_pattern": "thanks_name",
            "verbosity": "short",
            "emoji_usage": "none",
            "register": "familier",
            "future_field": "x",
        }
        profile = style_profile_from_dict(payload)
        self.assertEqual(profile.register, "familier")

    def test_dataclass_defaults(self) -> None:
        profile = StyleProfile(
            formality="medium",
            signature_pattern="thanks_name",
            verbosity="short",
            emoji_usage="none",
        )
        self.assertEqual(profile.typo_propensity, "none")


if __name__ == "__main__":
    unittest.main()
