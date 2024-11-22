import unittest
from stinx.potential import get_potential_path
import os


class TestStinx(unittest.TestCase):

    def test_path_exists(self):
        self.assertTrue(os.path.exists(get_potential_path("Ag")))


if __name__ == "__main__":
    unittest.main()
