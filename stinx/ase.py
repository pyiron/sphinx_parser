import numpy as np
import scipy.constants as sc
from stinx.input import sphinx
from ase.io.vasp import _handle_ase_constraints


def get_constraints(atoms):
    if atoms.constraints:
        return _handle_ase_constraints(atoms)
    else:
        return np.full(shape=atoms.positions.shape, fill_value=True)


def _to_angstrom(cell, positions):
    bohr_to_angstrom = sc.physical_constants["Bohr radius"][0] / sc.angstrom
    cell = np.array(cell) / bohr_to_angstrom
    positions = np.array(positions) / bohr_to_angstrom
    return cell, positions


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
    movable = get_constraints(structure)
    labels = structure.get_initial_magnetic_moments()
    elements = np.array(structure.get_chemical_symbols())
    species = []
    for elm_species in np.unique(elements):
        elm_list = elements == elm_species
        atom_list = []
        for elm_pos, elm_magmom, selective in zip(
            positions[elm_list],
            labels[elm_list],
            movable[elm_list],
        ):
            atom_group = {
                "coords": np.array(elm_pos),
                "label": f'"spin_{elm_magmom}"',
            }
            if all(selective):
                atom_group["movable"] = True
            elif any(selective):
                for xx in np.array(["X", "Y", "Z"])[selective]:
                    atom_group["movable" + xx] = True
            atom_list.append(sphinx.structure.species.atom.create(**atom_group))
        species.append(
            sphinx.structure.species.create(element=f'"{elm_species}"', atom=atom_list)
        )
    symmetry = None
    if not use_symmetry:
        symmetry = sphinx.structure.symmetry.create(
            operator=sphinx.structure.symmetry.operator.create(S=np.eye(3).tolist())
        )
    structure_group = sphinx.structure.create(
        cell=np.array(cell), species=species, symmetry=symmetry
    )
    return structure_group
