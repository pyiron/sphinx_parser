import os
import unittest

from sphinx_parser.potential import (
    _is_jth_potential,
    _is_vasp_potential,
    _remove_hash_tag,
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
