from __future__ import annotations

import unittest

from atlas_anno.generation.character_builder import build_characters
from atlas_anno.generation.world_builder import build_worlds
from atlas_anno.records import character_from_dict
from atlas_anno.io import serialize


class ContextualCuesTest(unittest.TestCase):
    def setUp(self) -> None:
        worlds = build_worlds(count=3)
        self.characters = build_characters(worlds, per_world=12)

    def test_every_character_has_cues(self) -> None:
        for character in self.characters:
            self.assertGreaterEqual(len(character.contextual_cues), 1, character.person_id)
            self.assertLessEqual(len(character.contextual_cues), 4)

    def test_cues_reveal_actual_attributes(self) -> None:
        for character in self.characters:
            for cue in character.contextual_cues:
                if cue.reveals_label == "LOCATION":
                    self.assertEqual(cue.reveals_value, character.region)
                elif cue.reveals_label == "ROLE":
                    self.assertEqual(cue.reveals_value, character.role)
                elif cue.reveals_label == "AGE_RANGE":
                    self.assertEqual(cue.reveals_value, character.age_range)
                elif cue.reveals_label == "NATIONALITY":
                    self.assertEqual(cue.reveals_value, character.nationality)
                elif cue.reveals_label == "FAMILY_STATUS":
                    self.assertIn("FAMILY_STATUS", character.sensitive_attributes)
                else:
                    self.fail(f"label inattendu: {cue.reveals_label}")

    def test_education_era_cue_matches_birth_year(self) -> None:
        found = False
        for character in self.characters:
            for cue in character.contextual_cues:
                if cue.cue_type == "education_era":
                    found = True
                    self.assertTrue(
                        str(character.birth_year + 18) in cue.cue_text
                        or str(character.birth_year + 23) in cue.cue_text,
                        cue.cue_text,
                    )
        self.assertTrue(found, "aucun indice education_era tiré sur 36 profils")

    def test_cues_are_deterministic(self) -> None:
        worlds = build_worlds(count=3)
        again = build_characters(worlds, per_world=12)
        for first, second in zip(self.characters, again):
            self.assertEqual(first.contextual_cues, second.contextual_cues)

    def test_cues_round_trip_serialization(self) -> None:
        character = self.characters[0]
        payload = serialize(character)
        loaded = character_from_dict(payload)
        self.assertEqual(loaded.contextual_cues, character.contextual_cues)


if __name__ == "__main__":
    unittest.main()
