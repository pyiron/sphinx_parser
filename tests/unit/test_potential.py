import os
import unittest
from pathlib import Path

from sphinx_parser.potential import (
    _is_jth_potential,
    _is_vasp_potential,
    _remove_hash_tag,
    get_paw_from_chemical_symbols,
    get_potential_path,
)


class TestPotential(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the path to the folder of this file
        d = os.path.dirname(os.path.realpath(__file__))
        cls.file_path = os.path.join(d, "..", "static")

    def test_path_exists(self):
        self.assertTrue(os.path.exists(get_potential_path("Ag")))

    # Repeated species blocks are stored as "species", "species___0", ...
    @staticmethod
    def _all_species(paw_dict):
        return [
            v
            for k, v in paw_dict.items()
            if k == "species" or k.startswith("species___")
        ]

    def test_injected_potential_used(self):
        custom_path = Path("/custom/pots/Ag_custom.atomicdata")
        result = get_paw_from_chemical_symbols(
            ["Ag"],
            potentials={"Ag": {"potential": custom_path, "potType": "VaspPAW"}},
        )
        species = self._all_species(result)
        self.assertEqual(len(species), 1)
        self.assertEqual(species[0]["potential"], f'"{custom_path}"')
        self.assertEqual(species[0]["potType"], '"VaspPAW"')

    def test_injected_potType_respected(self):
        custom_path = Path("/pots/Fe_GGA.atomicdata")
        result = get_paw_from_chemical_symbols(
            ["Fe"],
            potentials={"Fe": {"potential": custom_path, "potType": "AtomPAW"}},
        )
        self.assertEqual(self._all_species(result)[0]["potType"], '"AtomPAW"')

    def test_fallback_when_element_not_in_potentials(self):
        # Ag is not in the injected dict, so it must fall back to the env path
        result = get_paw_from_chemical_symbols(
            ["Ag"],
            potentials={
                "Fe": {"potential": Path("/pots/Fe.atomicdata"), "potType": "AtomPAW"}
            },
        )
        species = self._all_species(result)
        self.assertEqual(len(species), 1)
        self.assertIn(get_potential_path("Ag"), species[0]["potential"])
        self.assertEqual(species[0]["potType"], '"AtomPAW"')

    def test_fallback_when_potentials_none(self):
        result = get_paw_from_chemical_symbols(["Ag"], potentials=None)
        self.assertIn(
            get_potential_path("Ag"), self._all_species(result)[0]["potential"]
        )

    def test_mixed_injected_and_fallback(self):
        custom_path = Path("/pots/Fe_custom.atomicdata")
        result = get_paw_from_chemical_symbols(
            ["Ag", "Fe"],
            potentials={"Fe": {"potential": custom_path, "potType": "VaspPAW"}},
        )
        by_element = {s["element"].strip('"'): s for s in self._all_species(result)}
        self.assertIn(str(custom_path), by_element["Fe"]["potential"])
        self.assertEqual(by_element["Fe"]["potType"], '"VaspPAW"')
        self.assertIn(get_potential_path("Ag"), by_element["Ag"]["potential"])
        self.assertEqual(by_element["Ag"]["potType"], '"AtomPAW"')

    def test_is_xyz_potential(self):
        with open(os.path.join(self.file_path, "potentials", "Ag_POTCAR"), "r") as f:
            file_content = f.read()
            file_content = _remove_hash_tag(file_content)
        self.assertTrue(_is_vasp_potential(file_content))
        self.assertFalse(_is_jth_potential(file_content))
        with open(
            os.path.join(self.file_path, "potentials", "Ag_GGA.atomicdata"), "r"
        ) as f:
            file_content = f.read()
            file_content = _remove_hash_tag(file_content)
        self.assertFalse(_is_vasp_potential(file_content))
        self.assertTrue(_is_jth_potential(file_content))


if __name__ == "__main__":
    unittest.main()
