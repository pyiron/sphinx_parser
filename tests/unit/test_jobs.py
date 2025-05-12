import unittest

from ase.build import bulk

from sphinx_parser.jobs import apply_minimization, set_base_parameters
from sphinx_parser.toolkit import to_sphinx


class TestJobs(unittest.TestCase):
    def test_magnetic_bulk(self):
        structure = bulk("Fe", cubic=True)
        self.assertTrue("atomicSpin" in to_sphinx(set_base_parameters(structure)))
        structure = bulk("Al", cubic=True)
        self.assertFalse("atomicSpin" in to_sphinx(set_base_parameters(structure)))

    def test_calc_minimize(self):
        structure = bulk("Fe", cubic=True)
        input_sx = set_base_parameters(structure)
        default_input_sx = to_sphinx(apply_minimization(input_sx))
        self.assertTrue("linQN" in default_input_sx)
        for term in ["ricQN", "QN", "ricTS"]:
            self.assertEqual(
                to_sphinx(apply_minimization(input_sx, mode=term)),
                default_input_sx.replace("linQN", term),
            )
        with self.assertRaises(ValueError):
            apply_minimization(input_sx, mode="not_a_valid_mode")
        input_sx.pop("main")
        with self.assertRaises(ValueError):
            apply_minimization(input_sx)


if __name__ == "__main__":
    unittest.main()
