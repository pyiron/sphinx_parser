import unittest
from ase.build import bulk
from stinx.ase import get_structure_group
from stinx.toolkit import to_sphinx
import re


class TestStinx(unittest.TestCase):
    def test_Ni_Al_bulk(self):
        structure = bulk("Al", cubic=True)
        structure[0].symbol = "Ni"
        input_dict = get_structure_group(structure, use_symmetry=False)
        for key in ["cell", "species", "species___0", "symmetry"]:
            self.assertTrue(key in input_dict)
        text = to_sphinx(input_dict)
        self.assertEqual(len(re.findall(r'\bspecies\b', text, re.IGNORECASE)), 2)
        self.assertEqual(len(re.findall(r'\batom\b', text, re.IGNORECASE)), 4)


if __name__ == "__main__":
    unittest.main()
