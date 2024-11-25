import unittest
from ase.build import bulk
from stinx.ase import get_structure_group, id_ase_to_spx, id_spx_to_ase
from stinx.toolkit import to_sphinx
import re
from ase.constraints import FixedPlane


class TestStinx(unittest.TestCase):
    def test_Ni_Al_bulk(self):
        structure = bulk("Al", cubic=True)
        structure[0].symbol = "Ni"
        input_dict = get_structure_group(structure, use_symmetry=False)
        for key in ["cell", "species", "species___0", "symmetry"]:
            self.assertTrue(key in input_dict)
        text = to_sphinx(input_dict)
        self.assertEqual(
            len(re.findall(r"\bspecies\b", text, re.IGNORECASE)),
            2,
            msg="There must be exactly two species (Al and Ni) in the structure",
        )
        self.assertEqual(
            len(re.findall(r"\batom\b", text, re.IGNORECASE)),
            4,
            msg="There must be exactly four atoms in cubic fcc",
        )

    def test_constraint_bulk(self):
        structure = bulk("Al", cubic=True)
        c = FixedPlane([0], [1, 0, 0])
        structure.set_constraint(c)
        struct_group = get_structure_group(structure)
        self.assertFalse(
            "movableX" in struct_group["species"]["atom"],
            msg="Not allowed to move along X",
        )
        for term in ["movableY", "movableZ"]:
            self.assertTrue(
                term in struct_group["species"]["atom"],
                msg="Must be allowed to move along Y and Z",
            )
        for term in ["movableX", "movableY", "movableZ"]:
            self.assertFalse(
                term in struct_group["species"]["atom___0"],
                msg="Must be using the global movable",
            )
        self.assertTrue(
            "movable" in struct_group["species"]["atom___0"],
            msg="Must be globally movable",
        )

    def test_id_conversion(self):
        structure = bulk("Al", cubic=True)
        structure[0].symbol = "Ni"
        self.assertEqual(id_ase_to_spx(structure).tolist(), [1, 2, 3, 0])
        self.assertEqual(
            id_spx_to_ase(structure)[id_ase_to_spx(structure)].tolist(),
            [0, 1, 2, 3]
        )


if __name__ == "__main__":
    unittest.main()
