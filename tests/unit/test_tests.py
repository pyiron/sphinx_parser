import unittest
import stinx


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = stinx.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
