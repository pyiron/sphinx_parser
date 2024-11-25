import unittest
from sphinx_parser.input import sphinx
from sphinx_parser.toolkit import to_sphinx


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

    def test_include_format(self):
        paw_group = sphinx.PAWHamiltonian.create(xc=1, spinPolarized=False, ekt=0.2)
        input_sx = sphinx.create(PAWHamiltonian=paw_group)
        self.assertTrue("format paw;" in to_sphinx(input_sx))
        self.assertFalse("format paw;" in to_sphinx(input_sx, include_format=False))
        basis_group = sphinx.basis.create(eCut=25, kPoint=sphinx.basis.kPoint.create(coords=3 * [0.5]))
        input_sx = sphinx.create(basis=basis_group)
        self.assertFalse("format paw;" in to_sphinx(input_sx))
        pw = sphinx.PWHamiltonian.create(xc="PBE")
        pseudo = sphinx.pseudoPot.create()
        input_sx = sphinx.create(PWHamiltonian=pw, pseudoPot=pseudo)
        self.assertTrue("format sphinx;" in to_sphinx(input_sx))


if __name__ == "__main__":
    unittest.main()
