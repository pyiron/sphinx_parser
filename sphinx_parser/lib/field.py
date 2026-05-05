# coding: utf-8
# Copyright (c) - Max-Planck-Institut für Eisenforschung GmbH Computational Materials Design (CM) Department, MPIE.
# Distributed under the terms of "New BSD License", see the LICENSE file.

import numpy as np
from pyiron_atomistics.sphinx.base import Group


class HighFieldJob:
    num_of_jobs = 0
    HARTREE_TO_EV = 27.2114
    ANGSTROM_TO_BOHR = 1.8897
    preconditioner = "ELLIPTIC"
    rhomixing = str(0.7)
    preconscaling = 0.3
    ekt = 0.1
    ekt_scheme = "Fermi"
    e_energy = 1e-7
    i_energy = 1e-3

    def __init__(self, pr, e_field, en_cut, k_cut):
        """HighFieldJob instance which has pr a pyiron project attribute, structure attribute, job_name attribute,
        eField as the electric field (V/A) to be applied attribute, en_cut as the energy cutoff attribute in eV and
        k_cut as the k_point mesh."""
        self.pr = pr
        self.e_field = e_field
        self.en_cut = en_cut
        self.k_cut = k_cut
        HighFieldJob.num_of_jobs += 1

    def gdc_evaporation(
        self, structure, job_name, index, z_height=2, vdw=False, PES_xy=False, TS=False
    ):
        """Function to set up charged slab calculations with eField in Volts/Angstrom, and fixing layers below the
        specified z_height (Angstroms). The function take HighFieldJob instance as input with additional arguments
        of index for field evaporating atom. Set PES_xy to True to calculate PES along xy.

        :param TS: Set to True for transition state optimization calculation
        :param PES_xy: Set to True for potential energy surface calculation along xy
        :param vdw: set to True to include van derWalls corrections of D2 type
        :param z_height: height of layers to be fixed
        :param index: index of the evaporating (interested) atom
        :param job_name: name of pyiron job
        :param structure: pyiron structure object

        :return: a pyiron job
        """
        job = self.pr.create_job(job_type=self.pr.job_type.Sphinx, job_name=job_name)
        job.set_occupancy_smearing(self.ekt_scheme, width=self.ekt)
        job.structure = structure
        job.set_encut(self.en_cut)  # in eV
        job.set_kpoints(self.k_cut, center_shift=[0.5, 0.5, 0.25])
        job.set_convergence_precision(
            electronic_energy=self.e_energy, ionic_energy_tolerance=self.i_energy
        )
        positions = [p[2] for p in job.structure.positions]
        job.structure.add_tag(selective_dynamics=(True, True, True))
        job.structure.selective_dynamics[
            np.where(np.asarray(positions) < z_height)[0]
        ] = (False, False, False)
        if PES_xy:
            job.structure.selective_dynamics[index] = (False, False, True)
        job.structure.selective_dynamics[index] = (True, True, False)
        job.calc_minimize(ionic_steps=100, electronic_steps=100)
        right_field = self.e_field / 51.4  # atomic units (1 E_h/ea_0 ~= 51.4 V/Å)
        left_field = 0.0
        cell = job.structure.cell * self.ANGSTROM_TO_BOHR
        area = np.linalg.norm(np.cross(cell[0], cell[1]))
        total_charge = (right_field - left_field) * area / (4 * np.pi)
        sort_positions = np.sort(positions)
        job.input.sphinx.initialGuess.rho.charged = Group({})
        job.input.sphinx.initialGuess.rho.charged.charge = total_charge
        job.input.sphinx.initialGuess.rho.charged.z = (
            sort_positions[-2] * self.ANGSTROM_TO_BOHR
        )
        if vdw:
            job.input.sphinx.PAWHamiltonian.vdwCorrection = Group({})
            job.input.sphinx.PAWHamiltonian.vdwCorrection.method = '"D2"'
        job.input.sphinx.PAWHamiltonian.nExcessElectrons = -total_charge
        job.input.sphinx.PAWHamiltonian.dipoleCorrection = True
        job.input.sphinx.PAWHamiltonian.zField = left_field * self.HARTREE_TO_EV
        job.input["sphinx"]["main"]["ricQN"]["bornOppenheimer"]["scfDiag"][
            "rhoMixing"
        ] = self.rhomixing  # using conservative mixing can help with convergence.
        job.input["sphinx"]["main"]["ricQN"]["bornOppenheimer"]["scfDiag"][
            "preconditioner"
        ]["type"] = self.preconditioner
        job.input["sphinx"]["main"]["ricQN"]["bornOppenheimer"]["scfDiag"][
            "preconditioner"
        ]["scaling"] = self.preconscaling
        if TS:
            mainGroup = job.input.sphinx["main"]
            # rename ricQN group inside mainGroup to ricTS group
            mainGroup["ricTS"] = mainGroup.pop("ricQN")
            # now add the transition path group inside ricTS
            # atomId must be the atom index of the evaporating atom
            mainGroup["ricTS"].set_group("transPath")
            tp = mainGroup["ricTS"]["transPath"]
            tp["atomId"] = index + 1  # python to sphinx change in indices
            tp["dir"] = [0, 0, 1]
            job.input["sphinx"]["main"]["ricTS"]["bornOppenheimer"]["scfDiag"][
                "rhoMixing"
            ] = self.rhomixing  # using conservative mixing can help with convergence.
            job.input["sphinx"]["main"]["ricTS"]["bornOppenheimer"]["scfDiag"][
                "preconditioner"
            ]["type"] = self.preconditioner
            job.input["sphinx"]["main"]["ricTS"]["bornOppenheimer"]["scfDiag"][
                "preconditioner"
            ]["scaling"] = self.preconscaling
        return job
