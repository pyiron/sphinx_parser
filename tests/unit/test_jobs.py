import unittest
from ase.build import bulk
from sphinx_parser.jobs import calc_static
from sphinx_parser.toolkit import to_sphinx


class TestJobs(unittest.TestCase):
    def test_magnetic_bulk(self):
        structure = bulk("Fe", cubic=True)
        self.assertTrue("atomicSpin" in to_sphinx(calc_static(structure)))
        structure = bulk("Al", cubic=True)
        self.assertFalse("atomicSpin" in to_sphinx(calc_static(structure)))


if __name__ == "__main__":
    unittest.main()
