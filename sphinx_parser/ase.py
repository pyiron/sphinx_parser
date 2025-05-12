import numpy as np
from ase.io.vasp import _handle_ase_constraints
from pint import UnitRegistry

from sphinx_parser.input import sphinx


def _get_spin_label(spin):
    return f"spin_{spin}"


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
            "coords": elm_pos,
            "label": _get_spin_label(elm_magmom),
        }
        atom_group.update(_get_movable(selective))
        atom_list.append(sphinx.structure.species.atom.create(**atom_group))
    return atom_list


def _get_species_list(positions, elements, spins, movable):
    species = []
    for elm_species in np.unique(elements):
        elm_list = elements == elm_species
        atom_list = _get_atom_list(positions, spins, movable, elm_list)
        species.append(
            sphinx.structure.species.create(element=elm_species, atom=atom_list)
        )
    return species


def _get_spin_list(spins):
    return [
        sphinx.initialGuess.rho.atomicSpin.create(
            label=_get_spin_label(spin),
            spin=spin,
        )
        for spin in np.unique(spins)
    ]


def get_structure_group(structure, use_symmetry=True):
    """
    create a SPHInX Group object based on structure

    Args:
        structure (Atoms): ASE structure object
        use_symmetry (bool): Whether or not consider internal symmetry

    Returns:
        (Group): structure group
    """
    ureg = UnitRegistry()
    movable = ~_handle_ase_constraints(structure)
    # When the user changes the magnetic moments, they are sometimes no longer
    # numpy array afterwards.
    spins = np.array(structure.get_initial_magnetic_moments())
    elements = np.array(structure.get_chemical_symbols())
    species = _get_species_list(
        structure.positions * ureg.angstrom, elements, spins, movable
    )
    symmetry = None
    if not use_symmetry:
        symmetry = sphinx.structure.symmetry.create(
            operator=sphinx.structure.symmetry.operator.create(S=np.eye(3).tolist())
        )
    structure_group = sphinx.structure.create(
        cell=np.array(structure.cell) * ureg.angstrom,
        species=species,
        symmetry=symmetry,
    )
    if "initial_magmoms" in structure.arrays:
        spin_list = _get_spin_list(spins)
    else:
        spin_list = None
    return structure_group, spin_list


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
