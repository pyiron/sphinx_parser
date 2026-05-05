# coding: utf-8
# Copyright (c) - Max-Planck-Institut für Eisenforschung GmbH Computational Materials Design (CM) Department, MPIE.
# Distributed under the terms of "New BSD License", see the LICENSE file.

from typing import Optional, Dict, Any
from typing_extensions import Annotated
from ase import Atoms
from sphinx_parser.input import sphinx
import numpy as np


def create_sphinx_input(
    structure: Atoms,
    e_field: Annotated[float, {"units": "V/Å"}],
    en_cut: Annotated[float, {"units": "eV"}],
    k_cut: list[float],
    z_height: Annotated[float, {"units": "Å"}] = 2.0,
    index: Optional[int] = None,
    vdw: bool = False,
    PES_xy: bool = False,
    TS: bool = False,
    preconditioner: str = "ELLIPTIC",
    rhomixing: float = 0.7,
    preconscaling: float = 0.3,
    ekt: Annotated[float, {"units": "eV"}] = 0.1,
    e_energy: Annotated[float, {"units": "hartree"}] = 1e-7,
    i_energy: Annotated[float, {"units": "hartree"}] = 1e-3,
) -> Dict[str, Any]:
    """
    Create a dictionary for SPHInX input using sphinx_parser.input.

    Args:
        structure (Atoms): ASE structure object.
        e_field (float): Electric field in V/Å.
        en_cut (float): Energy cutoff in eV.
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
    cell = structure.cell
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    total_charge = (e_field * area) / (4 * np.pi)

    # Sort positions and determine fixed layers
    positions = [p[2] for p in structure.positions]
    sort_positions = np.sort(positions)
    fixed_layers = np.where(np.asarray(positions) < z_height)[0]

    # Create structure group
    struct_group = sphinx.structure(
        cell=structure.cell.tolist(),
        species={},
        wrap_string=True,
    )

    # Add selective dynamics
    selective_dynamics = np.full((len(structure), 3), True)
    selective_dynamics[fixed_layers] = [False, False, False]
    if PES_xy:
        selective_dynamics[index] = [False, False, True]
    selective_dynamics[index] = [True, True, False]

    # Create main group
    main_group = sphinx.main(
        ricQN=sphinx.main.ricQN(
            bornOppenheimer=sphinx.main.ricQN.bornOppenheimer(
                scfDiag=sphinx.main.ricQN.bornOppenheimer.scfDiag(
                    rhoMixing=rhomixing,
                    preconditioner=sphinx.main.ricQN.bornOppenheimer.scfDiag.preconditioner(
                        type_=preconditioner,
                        scaling=preconscaling,
                    ),
                )
            )
        )
    )

    # Add transition state optimization if TS is True
    if TS:
        main_group["ricTS"] = main_group.pop("ricQN")
        main_group["ricTS"].set_group("transPath")
        tp = main_group["ricTS"]["transPath"]
        tp["atomId"] = index + 1  # Python to SPHInX index adjustment
        tp["dir"] = [0, 0, 1]

    # Create PAWHamiltonian group
    paw_group = sphinx.PAWHamiltonian(
        xc="PBE",
        spinPolarized=False,
        ekt=ekt,
        nExcessElectrons=-total_charge,
        dipoleCorrection=True,
        zField=0.0,  # Left field is 0.0 in this case
    )
    if vdw:
        paw_group["vdwCorrection"] = sphinx.PAWHamiltonian.vdwCorrection(
            method="D2"
        )

    # Create initial guess group
    initial_guess_group = sphinx.initialGuess(
        rho=sphinx.initialGuess.rho(
            charged=sphinx.initialGuess.rho.charged(
                charge=total_charge,
                z=sort_positions[-2],
            ),
            atomicOrbitals=True,
        )
    )

    # Create basis group
    basis_group = sphinx.basis(
        eCut=en_cut,
        kPoint=sphinx.basis.kPoint(coords=k_cut),
    )

    # Combine all groups into the SPHInX input dictionary
    input_dict = sphinx(
        structure=struct_group,
        main=main_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
        basis=basis_group,
    )

    return input_dict
