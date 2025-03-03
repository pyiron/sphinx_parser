import unittest
import os
from sphinx_parser import output
import numpy as np


class TestOutput(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Get the path to the folder of this file
        d = os.path.dirname(os.path.realpath(__file__))
        cls.file_path = os.path.join(d, "..", "static")

    def _find_file(self, file_name):
        for root, dirs, files in os.walk(self.file_path):
            if file_name in files:
                yield os.path.join(root, file_name)

    def test_energy_dat(self):
        counter = 0
        for file in self._find_file("energy.dat"):
            energy = output.collect_energy_dat(file)
            self.assertTrue("scf_energy_free" in energy)
            self.assertIsInstance(energy["scf_energy_free"], list)
            self.assertLess(
                energy["scf_energy_free"][-1][-1], energy["scf_energy_zero"][-1][-1]
            )
            self.assertLess(
                energy["scf_energy_zero"][-1][-1], energy["scf_energy_int"][-1][-1]
            )
            counter += 1
        self.assertGreater(counter, 0)

    def test_residue(self):
        counter = 0
        for file in self._find_file("residue.dat"):
            residue = output.collect_residue_dat(file)
            counter += 1
            self.assertGreater(len(residue), 0)
            self.assertGreater(len(residue["scf_residue"]), 0)
            self.assertGreater(len(residue["scf_residue"][-1]), 0)
            self.assertGreater(np.min(residue["scf_residue"][-1]), 0)
        self.assertGreater(counter, 0)

    def test_spx_log_parser(self):
        for file in self._find_file("sphinx.log"):
            spx_output = output.SphinxLogParser.load_from_path(file)
            self.assertIsInstance(spx_output.results, dict)

    def test_collect_eps_dat(self):
        counter = 0
        for file in self._find_file("eps.dat"):
            eps = output.collect_eps_dat(file)
            counter += 1
            self.assertEqual(len(eps["bands_eigen_values"].shape), 4)
        self.assertGreater(counter, 0)

    def test_energy_struct(self):
        counter = 0
        for file in self._find_file("energy-structOpt.dat"):
            energy = output.collect_energy_struct(file)
            self.assertTrue("energy_free" in energy)
            self.assertIsInstance(energy["energy_free"], np.ndarray)
            counter += 1
        self.assertGreater(counter, 0)

    def test_spins(self):
        counter = 0
        for file in self._find_file("spins.dat"):
            spins = output.collect_spins_dat(file)
            self.assertTrue("atom_scf_spins" in spins)
            self.assertIsInstance(spins["atom_scf_spins"], list)
            counter += 1
        self.assertGreater(counter, 0)


if __name__ == "__main__":
    unittest.main()
