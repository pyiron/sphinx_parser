import unittest
import sphinx


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = sphinx.__version__
        print(version)
        self.assertTrue(version.startswith('0'))
