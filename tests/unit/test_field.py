# coding: utf-8
# Copyright (c) - Max-Planck-Institut für Eisenforschung GmbH Computational Materials Design (CM) Department, MPIE.
# Distributed under the terms of "New BSD License", see the LICENSE file.

import unittest

import numpy as np
from ase.build import bulk, fcc111
from ase.constraints import FixAtoms, FixedLine, FixedPlane

from sphinx_parser.lib.field import (
    _apply_constraint,
    _get_total_charge,
    create_sphinx_input,
    angstrom_to_bohr,
)
from sphinx_parser.toolkit import to_sphinx


def _make_slab():
    """Create a simple slab structure for testing."""
    structure = fcc111("Al", size=(2, 2, 3), vacuum=10.0)
    return structure


class TestApplyConstraint(unittest.TestCase):
    def setUp(self):
        self.structure = _make_slab()

    def test_returns_copy(self):
        original_constraints = self.structure.constraints
        result = _apply_constraint(self.structure, index=0, z_height=3.0, PES_xy=False)
        self.assertIsNot(result, self.structure)
        self.assertEqual(self.structure.constraints, original_constraints)

    def test_fixed_layers_applied(self):
        z_height = 3.0
        result = _apply_constraint(
            self.structure, index=0, z_height=z_height, PES_xy=False
        )
        fixed_indices = np.where(self.structure.positions[:, 2] < z_height)[0]
        fix_atoms_constraint = next(
            (c for c in result.constraints if isinstance(c, FixAtoms)), None
        )
        self.assertIsNotNone(
            fix_atoms_constraint, "FixAtoms constraint should be present"
        )
        np.testing.assert_array_equal(
            np.sort(fix_atoms_constraint.index), np.sort(fixed_indices)
        )

    def test_pes_xy_false_uses_fixed_line(self):
        result = _apply_constraint(self.structure, index=0, z_height=0.0, PES_xy=False)
        fixed_line = next(
            (c for c in result.constraints if isinstance(c, FixedLine)), None
        )
        self.assertIsNotNone(
            fixed_line, "FixedLine constraint should be present when PES_xy=False"
        )

    def test_pes_xy_true_uses_fixed_plane(self):
        result = _apply_constraint(self.structure, index=0, z_height=0.0, PES_xy=True)
        fixed_plane = next(
            (c for c in result.constraints if isinstance(c, FixedPlane)), None
        )
        self.assertIsNotNone(
            fixed_plane, "FixedPlane constraint should be present when PES_xy=True"
        )

    def test_no_fixed_layer_when_z_height_is_zero(self):
        result = _apply_constraint(self.structure, index=0, z_height=0.0, PES_xy=False)
        fix_atoms_constraint = next(
            (c for c in result.constraints if isinstance(c, FixAtoms)), None
        )
        self.assertIsNotNone(fix_atoms_constraint)
        self.assertEqual(len(fix_atoms_constraint.index), 0)

    def test_constraint_count(self):
        result = _apply_constraint(self.structure, index=0, z_height=3.0, PES_xy=False)
        self.assertEqual(len(result.constraints), 2)


class TestGetTotalCharge(unittest.TestCase):
    def test_rectangular_cell(self):
        """Test with a rectangular cell where cell[0] and cell[1] are orthogonal."""
        cell = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
        e_field = 1.0
        expected_area = 3.0 * 4.0
        expected_charge = (e_field * expected_area) / (4 * np.pi)
        result = _get_total_charge(e_field, cell)
        self.assertAlmostEqual(result, expected_charge)

    def test_zero_field(self):
        cell = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
        self.assertAlmostEqual(_get_total_charge(0.0, cell), 0.0)

    def test_negative_field(self):
        cell = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
        e_field = -0.5
        expected_area = 3.0 * 4.0
        expected_charge = (e_field * expected_area) / (4 * np.pi)
        result = _get_total_charge(e_field, cell)
        self.assertAlmostEqual(result, expected_charge)

    def test_non_orthogonal_cell(self):
        """Test with a hexagonal-like cell."""
        a = 3.0
        cell = np.array(
            [[a, 0.0, 0.0], [a * 0.5, a * np.sqrt(3) / 2, 0.0], [0.0, 0.0, 10.0]]
        )
        e_field = 0.01
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        expected_charge = (e_field * area) / (4 * np.pi)
        result = _get_total_charge(e_field, cell)
        self.assertAlmostEqual(result, expected_charge)

    def test_scales_with_area(self):
        """Doubling one cell vector should double the charge."""
        cell1 = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
        cell2 = np.array([[6.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 10.0]])
        e_field = 0.01
        charge1 = _get_total_charge(e_field, cell1)
        charge2 = _get_total_charge(e_field, cell2)
        self.assertAlmostEqual(charge2, 2 * charge1)


class TestCreateSphinxInput(unittest.TestCase):
    def setUp(self):
        self.structure = _make_slab()
        self.e_field = 0.001
        self.en_cut = 25.0
        self.k_cut = [2, 2, 1]
        self.index = 5

    def test_returns_dict(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
        )
        self.assertIsInstance(result, dict)

    def test_required_top_level_keys(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
        )
        for key in ["structure", "main", "PAWHamiltonian", "initialGuess", "basis"]:
            self.assertIn(key, result, f"Key '{key}' should be present in output")

    def test_without_ts(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            TS=False,
        )
        main = result["main"]
        self.assertIn("ricQN", main, "ricQN should be in main when TS=False")
        self.assertNotIn("ricTS", main, "ricTS should not be in main when TS=False")

    def test_with_ts(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            TS=True,
        )
        main = result["main"]
        self.assertIn("ricTS", main, "ricTS should be in main when TS=True")
        self.assertNotIn("ricQN", main, "ricQN should not be in main when TS=True")

    def test_with_ts_trans_path_atom_id(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            TS=True,
        )
        trans_path = result["main"]["ricTS"]["transPath"]
        self.assertEqual(
            trans_path["atomId"],
            self.index + 1,
            "atomId should be 1-indexed (SPHInX convention)",
        )

    def test_with_vdw(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            vdw=True,
        )
        paw = result["PAWHamiltonian"]
        self.assertIn(
            "vdwCorrection", paw, "vdwCorrection should be present when vdw=True"
        )

    def test_without_vdw(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            vdw=False,
        )
        paw = result["PAWHamiltonian"]
        self.assertNotIn(
            "vdwCorrection", paw, "vdwCorrection should not be present when vdw=False"
        )

    def test_n_excess_electrons_from_charge(self):
        """nExcessElectrons should equal the negated total charge."""
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
        )
        cell_bohr = np.array(self.structure.cell) * angstrom_to_bohr
        total_charge = _get_total_charge(self.e_field, cell_bohr)
        paw = result["PAWHamiltonian"]
        self.assertAlmostEqual(paw["nExcessElectrons"], -total_charge)

    def test_dipole_correction_enabled(self):
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
        )
        self.assertTrue(result["PAWHamiltonian"]["dipoleCorrection"])

    def test_pes_xy_false_uses_fixed_line(self):
        """With PES_xy=False the constrained atom can only move along z (FixedLine)."""
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            PES_xy=False,
        )
        # FixedLine along z means movableX=False, movableY=False, movableZ=True for
        # the constrained atom. The sphinx structure text should contain movableX
        # or movableY entries equal to false.
        structure_text = to_sphinx(result["structure"])
        self.assertIn(
            "movableX = false",
            structure_text,
            "FixedLine constraint should restrict X movement",
        )
        self.assertIn(
            "movableY = false",
            structure_text,
            "FixedLine constraint should restrict Y movement",
        )

    def test_pes_xy_true(self):
        """With PES_xy=True the constrained atom can only move in the xy plane (FixedPlane)."""
        result = create_sphinx_input(
            structure=self.structure,
            e_field=self.e_field,
            en_cut=self.en_cut,
            k_cut=self.k_cut,
            index=self.index,
            PES_xy=True,
        )
        # FixedPlane with normal [0,0,1] means movableZ=False for the constrained atom.
        structure_text = to_sphinx(result["structure"])
        self.assertIn(
            "movableZ = false",
            structure_text,
            "FixedPlane constraint should restrict Z movement",
        )


if __name__ == "__main__":
    unittest.main()
