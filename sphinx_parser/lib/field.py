# coding: utf-8
# Copyright (c) - Max-Planck-Institut für Eisenforschung GmbH Computational Materials Design (CM) Department, MPIE.
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Any, Dict

import ase
import numpy as np
from semantikon.converter import units
from typing_extensions import Annotated

from sphinx_parser.ase import get_structure_group
from sphinx_parser.input import sphinx

angstrom_to_bohr = 1.8897259886


def _apply_constraint(
    structure: ase.Atoms, index: int, z_height: float, PES_xy: bool
) -> ase.Atoms:
    """
    Apply selective dynamics constraints to the structure.

    Args:
        structure (ase.Atoms): ASE structure object.
        index (int): Index of the evaporating atom.
        z_height (float): Height of layers to be fixed in Å.
        PES_xy (bool): Whether to calculate PES along xy.

    Returns:
        ase.Atoms: Structure with applied constraints.
    """
    fixed_layers = np.where(structure.positions.T[2] < z_height)[0]
    fix_atoms = ase.constraints.FixAtoms(indices=fixed_layers)
    if PES_xy:
        fix_index = ase.constraints.FixedPlane(indices=index, direction=[0, 0, 1])
    else:
        fix_index = ase.constraints.FixedLine(indices=index, direction=[0, 0, 1])
    structure_copy = structure.copy()
    structure_copy.set_constraint([fix_atoms, fix_index])
    return structure_copy


def _get_total_charge(e_field: float, cell: np.ndarray) -> float:
    """
    Calculate total charge based on the electric field and cell area.

    Args:
        e_field (float): Electric field in hartree/bohr.
        cell (np.ndarray): Cell vectors in bohr.

    Returns:
        float: Total charge.
    """
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    return (e_field * area) / (4 * np.pi)


@units
def create_sphinx_input(
    structure: ase.Atoms,
    e_field: Annotated[float, {"units": "hartree/bohr"}],
    en_cut: Annotated[float, {"units": "hartree"}],
    k_cut: list[float],
    index: int,
    z_height: Annotated[float, {"units": "angstrom"}] = 2.0,
    vdw: bool = False,
    PES_xy: bool = False,
    TS: bool = False,
    preconditioner: str = "ELLIPTIC",
    rhomixing: float = 0.7,
    preconscaling: float = 0.3,
    ekt: Annotated[float, {"units": "eV"}] = 0.1,
    e_energy: Annotated[float, {"units": "hartree"}] = 1e-8,
    i_energy: Annotated[float, {"units": "hartree"}] = 1e-4,
) -> Dict[str, Any]:
    """
    Create a dictionary for SPHInX input using sphinx_parser.input.

    Args:
        structure (ase.Atoms): ASE structure object.
        e_field (float): Electric field in hartree/bohr.
        en_cut (float): Energy cutoff in hartree.
        k_cut (list[float]): K-point mesh.
        z_height (float): Height of layers to be fixed in Å.
        index (Optional[int]): Index of the evaporating atom.
        vdw (bool): Whether to include van der Waals corrections.
        PES_xy (bool): Whether to calculate PES along xy.
        TS (bool): Whether to perform transition state optimization.
        preconditioner (str): Preconditioner type.
        rhomixing (float): Rho mixing value.
        preconscaling (float): Preconditioner scaling value.
        ekt (float): Electronic temperature in eV.
        e_energy (float): Electronic energy convergence criterion in hartree.
        i_energy (float): Ionic energy convergence criterion in hartree.

    Returns:
        Dict[str, Any]: SPHInX input dictionary.
    """
    # Calculate total charge based on the electric field and cell area
    total_charge = _get_total_charge(e_field, structure.cell * angstrom_to_bohr)

    structure = _apply_constraint(structure, index, z_height, PES_xy)

    # Create structure group
    struct_group, spin_lst = get_structure_group(structure)

    # Create main group
    bornOppenheimer = sphinx.main.ricQN.bornOppenheimer(
        scfDiag=sphinx.main.ricQN.bornOppenheimer.scfDiag(
            rhoMixing=rhomixing,
            preconditioner=sphinx.main.ricQN.bornOppenheimer.scfDiag.preconditioner(
                type_=preconditioner,
                scaling=preconscaling,
            ),
            dEnergy=e_energy,
        )
    )

    # Add transition state optimization if TS is True
    if TS:
        main_group = sphinx.main(
            ricTS=sphinx.main.ricTS(
                bornOppenheimer=bornOppenheimer,
                transPath=sphinx.main.ricTS.transPath(
                    atomId=index + 1,  # Python to SPHInX index adjustment
                    dir_=[0, 0, 1],
                ),
                dEnergy=i_energy,
            )
        )
    else:
        main_group = sphinx.main(
            ricQN=sphinx.main.ricQN(bornOppenheimer=bornOppenheimer, dEnergy=i_energy)
        )

    # Create PAWHamiltonian group
    paw_group = sphinx.PAWHamiltonian(
        xc="PBE",
        spinPolarized=spin_lst is not None,
        ekt=ekt,
        nExcessElectrons=-total_charge,
        dipoleCorrection=True,
        zField=0.0,  # Left field is 0.0 in this case
        MethfesselPaxton=1,
    )
    if vdw:
        paw_group["vdwCorrection"] = sphinx.PAWHamiltonian.vdwCorrection(method="D2")

    # Create initial guess group
    initial_guess_group = sphinx.initialGuess(
        waves=sphinx.initialGuess.waves(
            lcao=sphinx.initialGuess.waves.lcao(),
        ),
        rho=sphinx.initialGuess.rho(
            charged=sphinx.initialGuess.rho.charged(
                charge=total_charge,
                z=np.sort(structure.positions[:, -1])[-2] * angstrom_to_bohr,
            ),
            atomicOrbitals=True,
            atomicSpin=spin_lst,
        ),
    )

    # Create basis group
    basis_group = sphinx.basis(
        eCut=en_cut,
        kPoint=sphinx.basis.kPoint(coords=[0.5, 0.5, 0.25], weight=1, relative=True),
        folding=k_cut,
    )

    # Combine all groups into the SPHInX input dictionary
    return sphinx(
        structure=struct_group,
        main=main_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
        basis=basis_group,
    )
