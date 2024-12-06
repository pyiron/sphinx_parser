import numpy as np
import scipy.constants as sc
from sphinx_parser.input import sphinx
from ase.io.vasp import _handle_ase_constraints


def _to_angstrom(cell, positions):
    bohr_to_angstrom = sc.physical_constants["Bohr radius"][0] / sc.angstrom
    cell = np.array(cell) / bohr_to_angstrom
    positions = np.array(positions) / bohr_to_angstrom
    return cell, positions


def _get_movable(selective):
    if all(selective):
        return {"movable": True}
    elif any(selective):
        return dict(zip(["movableX", "movableY", "movableZ"], selective))
    else:
        return {"movable": False}


def _get_atom_list(positions, spins, movable, elm_list):
    atom_list = []
    for elm_pos, elm_magmom, selective in zip(
        positions[elm_list],
        spins[elm_list],
        movable[elm_list],
    ):
        atom_group = {
            "coords": np.array(elm_pos),
            "label": f"spin_{elm_magmom}",
        }
        atom_group.update(_get_movable(selective))
        atom_list.append(sphinx.structure.species.atom.create(**atom_group))
    return atom_list


def _get_species_list(elements, spins, movable):
    species = []
    for elm_species in np.unique(elements):
        elm_list = elements == elm_species
        atom_list = _get_atom_list(positions, spins, movable, elm_list)
        species.append(
            sphinx.structure.species.create(element=elm_species, atom=atom_list)
        )
    return species


def get_structure_group(structure, use_symmetry=True):
    """
    create a SPHInX Group object based on structure

    Args:
        structure (Atoms): ASE structure object
        use_symmetry (bool): Whether or not consider internal symmetry

    Returns:
        (Group): structure group
    """
    cell, positions = _to_angstrom(structure.cell, structure.positions)
    movable = ~_handle_ase_constraints(structure)
    spins = structure.get_initial_magnetic_moments()
    elements = np.array(structure.get_chemical_symbols())
    species = _get_species_list(elements, spins, movable)
    symmetry = None
    if not use_symmetry:
        symmetry = sphinx.structure.symmetry.create(
            operator=sphinx.structure.symmetry.operator.create(S=np.eye(3).tolist())
        )
    structure_group = sphinx.structure.create(
        cell=np.array(cell), species=species, symmetry=symmetry
    )
    return structure_group


def id_ase_to_spx(structure):
    """
    Translate the ASE ordering of the atoms to the SPHInX ordering

    Args:
        structure (Atoms): ASE structure object

    Returns:
        (np.ndarray): SPHInX ordering of the atoms
    """
    # Translate the chemical symbols into indices
    indices = np.unique(structure.get_chemical_symbols(), return_inverse=True)[1]
    # Add small ramp to ensure order uniqueness
    indices = indices + np.arange(len(indices)) / len(indices)
    return np.argsort(indices)


def id_spx_to_ase(structure):
    """
    Translate the SPHInX ordering of the atoms to the ASE ordering

    Args:
        structure (Atoms): ASE structure object

    Returns:
        (np.ndarray): ASE ordering of the atoms
    """
    return np.argsort(id_ase_to_spx(structure))
