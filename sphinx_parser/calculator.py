from pathlib import Path

from ase.calculators.calculator import FileIOCalculator, FileIORules, StandardProfile
from ase.units import Hartree, Bohr

from sphinx_parser.jobs import set_base_parameters
from sphinx_parser.output import SphinxLogParser
from sphinx_parser.toolkit import to_sphinx


class SphinxDft(FileIOCalculator):
    implemented_properties = ["energy", "forces"]

    def __init__(self, *args, command: str = "sphinx", potentials=None, **kwargs):
        super().__init__(*args, **kwargs, profile=StandardProfile(command=command))
        self.fileio_rules = FileIORules(stdout_name="sphinx.log")
        self.potentials = potentials

    def write_input(self, atoms, properties=None, system_changes=None):
        super().write_input(atoms, properties, system_changes)

        # Reuse the shared helper to build the Sphinx input structure,
        # avoiding divergence from the defaults defined in jobs.set_base_parameters.
        input_sx = set_base_parameters(atoms, potentials=self.potentials)

        cwd = self.directory
        with open(Path(cwd) / "input.sx", "w") as f:
            f.write(to_sphinx(input_sx))

    def read_results(self):
        parser = SphinxLogParser.load_from_path(Path(self.directory) / "sphinx.log")
        self.results["energy"] = parser.get_energy_free()[-1][-1] * Hartree
        forces_au = parser.get_forces()[-1]
        self.results["forces"] = forces_au * Hartree / Bohr
