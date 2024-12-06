import unittest
from ase.build import bulk
from sphinx_parser.ase import get_structure_group, id_ase_to_spx, id_spx_to_ase
from sphinx_parser.toolkit import to_sphinx
import re
from ase.constraints import FixedPlane


class TestASE(unittest.TestCase):
    def test_Ni_Al_bulk(self):
        structure = bulk("Al", cubic=True)
        structure[0].symbol = "Ni"
        input_dict = get_structure_group(structure, use_symmetry=False)[0]
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

    def test_magmom(self):
        structure = bulk("Fe", cubic=True)
        struct_group = get_structure_group(structure)[0]
        self.assertEqual(struct_group["species"]["atom"]["label"][1:-1], "spin_2.3")

    def test_constraint_bulk(self):
        structure = bulk("Al", cubic=True)
        c = FixedPlane([0], [1, 0, 0])
        structure.set_constraint(c)
        struct_group = get_structure_group(structure)[0]
        self.assertFalse(
            struct_group["species"]["atom"]["movableX"],
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

    def test_magmom(self):
        structure = bulk("Al", cubic=True)
        input_dict, spin_lst = get_structure_group(structure, use_symmetry=False)
        self.assertIsNone(spin_lst, msg="No spin_lst for non-magnetic system")
        structure = bulk("Fe", cubic=True)
        input_dict, spin_lst = get_structure_group(structure, use_symmetry=False)
        self.assertEqual(
            len(spin_lst), 1, msg="Only one value for ferrimagnetic system"
        )
        structure = bulk("Fe", cubic=True)
        structure.arrays["initial_magmoms"] = [2.0, -2.0]
        input_dict, spin_lst = get_structure_group(structure, use_symmetry=False)
        self.assertEqual(
            len(spin_lst), 2, msg="Two values for antiferromagnetic system"
        )


if __name__ == "__main__":
    unittest.main()
