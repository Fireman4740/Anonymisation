from __future__ import annotations

import random
import re
import unittest
from collections import Counter

from atlas_anno.generation.attribute_sampler import (
    IdentitySampler,
    MarginalSampler,
    load_insee_marginals,
)
from atlas_anno.generation.character_builder import (
    UNIQUENESS_FIELDS,
    build_characters,
)
from atlas_anno.generation.world_builder import build_worlds

PHONE_PATTERN = re.compile(r"^0[1-7]( \d{2}){4}$")
NIR_PATTERN = re.compile(r"^[12] \d{2} \d{2} \d{2} \d{3} \d{3} \d{2}$")


def _shared(left, right) -> int:
    return sum(1 for field in UNIQUENESS_FIELDS if getattr(left, field) == getattr(right, field))


class AttributeSamplingTest(unittest.TestCase):
    def test_build_characters_is_deterministic(self) -> None:
        worlds = build_worlds(count=3)
        first = build_characters(worlds, per_world=12, seed=27)
        second = build_characters(worlds, per_world=12, seed=27)
        self.assertEqual(
            [(c.full_name, c.email, c.phone, c.location, c.age_range) for c in first],
            [(c.full_name, c.email, c.phone, c.location, c.age_range) for c in second],
        )

    def test_direct_identifier_formats(self) -> None:
        worlds = build_worlds(count=3)
        characters = build_characters(worlds, per_world=12)
        for character in characters:
            self.assertRegex(character.phone, PHONE_PATTERN)
            self.assertRegex(character.nir_like, NIR_PATTERN)
            self.assertTrue(character.email.endswith(".fr"), character.email)
            self.assertIn(character.username, character.email)
            self.assertTrue(character.address)
            self.assertGreater(character.birth_year, 1950)

    def test_world_uniqueness_constraint(self) -> None:
        worlds = build_worlds(count=3)
        characters = build_characters(worlds, per_world=12)
        by_world = {}
        for character in characters:
            by_world.setdefault(character.organization_id, []).append(character)
        for profiles in by_world.values():
            for index, profile in enumerate(profiles):
                for other in profiles[:index]:
                    self.assertLessEqual(
                        _shared(profile, other),
                        1,
                        f"{profile.person_id} et {other.person_id} partagent trop de quasi-identifiants",
                    )

    def test_name_diversity(self) -> None:
        worlds = build_worlds(count=3)
        characters = build_characters(worlds, per_world=12)
        names = [character.full_name for character in characters]
        self.assertGreaterEqual(len(set(names)), int(len(names) * 0.9))

    def test_age_distribution_follows_marginals(self) -> None:
        marginals = load_insee_marginals()
        sampler = MarginalSampler(random.Random(7), marginals)
        counts = Counter(sampler.sample_age_range("female") for _ in range(500))
        expected = marginals["age_by_sex"]["female"]
        for bucket, weight in expected.items():
            observed = counts.get(bucket, 0) / 500
            self.assertLess(abs(observed - weight), 0.08, f"{bucket}: {observed} vs {weight}")

    def test_identity_sampler_insertion_stable(self) -> None:
        sampler = IdentitySampler(seed=27)
        identity_5 = sampler.sample_identity(5, "female", 1990, "Bretagne")
        sampler.sample_identity(3, "male", 1985, "Normandie")
        identity_5_again = sampler.sample_identity(5, "female", 1990, "Bretagne")
        self.assertEqual(identity_5, identity_5_again)


if __name__ == "__main__":
    unittest.main()
