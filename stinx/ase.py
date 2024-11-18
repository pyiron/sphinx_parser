import numpy as np
import scipy.constants as sc
from stinx.input import sphinx
from ase.io.vasp import _handle_ase_constraints


def get_constraints(atoms):
    if atoms.constraints:
        return _handle_ase_constraints(atoms)
    else:
        return np.full(shape=atoms.positions.shape, fill_value=True)


def get_structure_group(structure, use_symmetry=True):
    """
    create a SPHInX Group object based on structure

    Args:
        structure (Atoms): ASE structure object
        use_symmetry (bool): Whether or not consider internal symmetry

    Returns:
        (Group): structure group
    """
    bohr_to_angstrom = sc.physical_constants["Bohr radius"][0] / sc.angstrom
    positions = np.array(structure.positions) / bohr_to_angstrom
    cell = np.array(structure.cell) / bohr_to_angstrom
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
            atom_group = {"coords": np.array(elm_pos)}
            if elm_magmom is not None:
                atom_group["label"] = '"spin_' + str(elm_magmom) + '"'
            if all(selective):
                atom_group["movable"] = True
            elif any(selective):
                for xx in np.array(["X", "Y", "Z"])[selective]:
                    atom_group["movable" + xx] = True
            atom_list.append(sphinx.structure.species.atom.create(**atom_group))
        species.append(
            sphinx.structure.species.create(element=f'"{elm_species}"', atom=atom_list)
        )
    structure_group = sphinx.structure.create(
        cell=np.array(cell), species=species
    )
    return structure_group
