import unittest

import sphinx_parser


class TestVersion(unittest.TestCase):
    def test_version(self):
        version = sphinx_parser.__version__
        print(version)
        self.assertTrue(version.startswith("0"))
