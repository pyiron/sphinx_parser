import unittest
import os
from stinx import output


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

    def test_output(self):
        for file in self._find_file("energy.dat"):
            energy = output.collect_energy_dat(file)
            self.assertTrue("scf_energy_free" in energy)
            self.assertIsInstance(energy["scf_energy_free"], list)


if __name__ == '__main__':
    unittest.main()
