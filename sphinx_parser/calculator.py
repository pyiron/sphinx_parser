from pathlib import Path

from ase.calculators.calculator import FileIOCalculator, FileIORules, StandardProfile

from sphinx_parser.ase import get_structure_group
from sphinx_parser.input import sphinx
from sphinx_parser.output import SphinxLogParser
from sphinx_parser.potential import get_paw_from_structure
from sphinx_parser.toolkit import to_sphinx


class SphinxDft(FileIOCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, profile=StandardProfile(command="sphinx"))
        self.fileio_rules = FileIORules(stdout_name="log.sx")

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)

        struct_group = get_structure_group(atoms)[0]
        main_group = sphinx.main(
            scfDiag=sphinx.main.scfDiag(maxSteps=10, blockCCG={}),
            evalForces=sphinx.main.evalForces("forces.txt"),
        )
        pawPot_group = get_paw_from_structure(atoms)
        basis_group = sphinx.basis(
            eCut=25, kPoint=sphinx.basis.kPoint(coords=3 * [0.5])
        )
        paw_group = sphinx.PAWHamiltonian(xc=1, spinPolarized=False, ekt=0.2)
        initial_guess_group = sphinx.initialGuess(
            waves=sphinx.initialGuess.waves(lcao=sphinx.initialGuess.waves.lcao()),
            rho=sphinx.initialGuess.rho(atomicOrbitals=True),
        )

        input_sx = sphinx(
            pawPot=pawPot_group,
            structure=struct_group,
            main=main_group,
            basis=basis_group,
            PAWHamiltonian=paw_group,
            initialGuess=initial_guess_group,
        )

        cwd = self.directory
        with open(Path(cwd) / "input.sx", "w") as f:
            f.write(to_sphinx(input_sx))

    def read_results(self):
        parser = SphinxLogParser.load_from_path(Path(self.directory) / "log.sx")
        self.results["energy"] = parser.get_energy_free()[-1][-1]
        self.results["forces"] = parser.get_forces()[-1]
