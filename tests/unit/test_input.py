import unittest
from stinx.input import sphinx


class TestStinx(unittest.TestCase):

    def test_wrap_string(self):
        with_wrap_string = sphinx.pawPot.species.create(
            potential="my_potential_path",
            potType="AtomPAW",
        )
        self.assertEqual(with_wrap_string, {'potential': '"my_potential_path"', 'potType': '"AtomPAW"'})
        without_wrap_string = sphinx.pawPot.species.create(
            potential="my_potential_path",
            potType="AtomPAW",
            wrap_string=False,
        )
        self.assertEqual(without_wrap_string, {'potential': 'my_potential_path', 'potType': 'AtomPAW'})


if __name__ == "__main__":
    unittest.main()
