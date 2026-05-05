import re
import warnings
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray


def _splitter(
    arr: Union[NDArray, list], counter: Union[NDArray, list]
) -> list:
    if len(arr) == 0 or len(counter) == 0:
        return []
    arr_new = []
    spl_loc = list(np.where(np.array(counter) == min(counter))[0])
    spl_loc.append(None)
    for ii, ll in enumerate(spl_loc[:-1]):
        arr_new.append(np.array(arr[ll : spl_loc[ii + 1]]).tolist())
    return arr_new


def collect_energy_dat(file_name: Union[str, Path] = "energy.dat") -> dict:
    """

    Args:
        file_name (str): file name

    Returns:
        (dict): results

    """
    energies = np.loadtxt(str(file_name), ndmin=2)

    return {
        "scf_computation_time": _splitter(energies[:, 1], energies[:, 0]),
        "scf_energy_int": _splitter(energies[:, 2], energies[:, 0]),
        "scf_energy_free": _splitter(energies[:, 3], energies[:, 0]),
        "scf_energy_zero": _splitter(energies[:, 4], energies[:, 0]),
        "scf_energy_band": _splitter(energies[:, 5], energies[:, 0]),
        "scf_electronic_entropy": _splitter(energies[:, 6], energies[:, 0]),
    }


def collect_residue_dat(file_name: Union[str, Path] = "residue.dat") -> dict:
    """

    Args:
        file_name (str): file name

    Returns:
        (dict): results

    """
    residue = np.loadtxt(file_name, ndmin=2)
    assert len(residue.shape) == 2
    return {"scf_residue": _splitter(residue[:, 1:].squeeze(), residue[:, 0])}


def _collect_eps_dat(file_name: Union[str, Path] = "eps.dat") -> NDArray:
    """

    Args:
        file_name:

    Returns:

    """
    return np.loadtxt(str(file_name), ndmin=2)[..., 1:]


def collect_eps_dat(
    file_name: Optional[Union[str, Path]] = None,
    cwd: Optional[Union[str, Path]] = None,
    spins: bool = True,
) -> dict:
    if file_name is not None:
        values = [_collect_eps_dat(file_name=file_name)]
    elif cwd is None:
        raise ValueError("cwd or file_name must be defined")
    elif spins:
        values = []
        for i in range(2):
            path = Path(cwd) / f"eps.{i}.dat"
            values.append(_collect_eps_dat(file_name=path))
    else:
        path = Path(cwd) / "eps.dat"
        values = [_collect_eps_dat(file_name=path)]
    values = np.stack(values, axis=0)
    return {"bands_eigen_values": values.reshape((-1,) + values.shape)}


def collect_energy_struct(
    file_name: Union[str, Path] = "energy-structOpt.dat",
) -> dict:
    """

    Args:
        file_name (str): file name

    Returns:
        (dict): results

    """
    return {"energy_free": np.loadtxt(str(file_name), ndmin=2).reshape(-1, 2)[:, 1]}


def _check_permutation(index_permutation: Optional[NDArray]) -> None:
    if index_permutation is None:
        return
    unique_indices = np.unique(index_permutation)
    assert len(unique_indices) == len(index_permutation)
    assert np.min(index_permutation) == 0
    assert np.max(index_permutation) == len(index_permutation) - 1


def collect_spins_dat(
    file_name: Union[str, Path] = "spins.dat",
    index_permutation: Optional[NDArray] = None,
) -> dict:
    """

    Args:
        file_name (str): file name
        index_permutation (numpy.ndarray): Indices for the permutation

    Returns:
        (dict): results

    """
    _check_permutation(index_permutation)
    spins = np.loadtxt(str(file_name), ndmin=2)
    if index_permutation is not None:
        s = np.array([ss[index_permutation] for ss in spins[:, 1:]])
    else:
        s = spins[:, 1:]
    return {"atom_scf_spins": _splitter(s, spins[:, 0])}


def collect_eval_forces(
    file_name: Union[str, Path],
    index_permutation: Optional[NDArray] = None,
) -> dict:
    """
    Args:
        file_name (str): file name
        index_permutation (numpy.ndarray): Indices for the permutation

    Returns:
        (dict): results

    # TODO: parse movable, elements, species etc.
    """
    _check_permutation(index_permutation)
    with open(str(file_name), "r") as f:
        file_content = "".join(f.readlines())
    n_steps = max(len(re.findall(r"// --- step \d", file_content, re.MULTILINE)), 1)
    f_v = ",".join(3 * [r"\s*([\d.-]+)"])

    def get_value(term: str, f: str = file_content, n: int = n_steps, p: Optional[NDArray] = index_permutation) -> NDArray:
        value = (
            np.array(re.findall(term, f, re.MULTILINE)).astype(float).reshape(n, -1, 3)
        )
        if p is not None:
            value = np.array([ff[p] for ff in value])
        return value

    cell = re.findall(
        r"cell = \[\[" + r"\],\n\s*\[".join(3 * [f_v]) + r"\]\];",
        file_content,
        re.MULTILINE,
    )
    cell = np.array(cell).astype(float).reshape(n_steps, 3, 3)
    return {
        "positions": get_value(r"atom {coords = \[" + f_v + r"\];"),
        "forces": get_value(r"force  = \[" + f_v + r"\]; }"),
        "cell": cell,
    }


class SphinxLogParser:
    def __init__(
        self,
        file_content: str,
        index_permutation: Optional[NDArray] = None,
    ) -> None:
        """
        Args:
            file_name (str): file name
            index_permutation (numpy.ndarray): Indices for the permutation

        """
        self.log_file: str = file_content
        self._n_atoms: Optional[int] = None
        _check_permutation(index_permutation)
        self._index_permutation = index_permutation
        self.generic_dict: dict = {
            "volume": self.get_volume,
            "forces": self.get_forces,
            "job_finished": self.job_finished,
        }
        self.dft_dict: dict = {
            "n_valence": self.get_n_valence,
            "bands_k_weights": self.get_bands_k_weights,
            "kpoints_cartesian": self.get_kpoints_cartesian,
            "bands_e_fermi": self.get_fermi,
            "bands_occ": self.get_occupancy,
            "bands_eigen_values": self.get_band_energy,
            "scf_convergence": self.get_convergence,
            "scf_energy_int": self.get_energy_int,
            "scf_energy_free": self.get_energy_free,
            "scf_magnetic_forces": self.get_magnetic_forces,
        }

    @classmethod
    def load_from_path(
        cls,
        path: Union[str, Path],
        index_permutation: Optional[NDArray] = None,
    ) -> "SphinxLogParser":
        """
        Args:
            path (str): file name
            index_permutation (numpy.ndarray): Indices for the permutation

        Returns:
            (SphinxLogParser): instance

        """
        with open(str(path), "r") as f:
            file_content = f.read()
        return cls(file_content, index_permutation)

    @property
    def index_permutation(self) -> Optional[NDArray]:
        return self._index_permutation

    @property
    def spin_enabled(self) -> bool:
        return len(re.findall("Spin moment:", self.log_file)) > 0

    @cached_property
    def log_main(self) -> Optional[str]:
        term = "Enter Main Loop"
        matches = re.finditer(rf"\b{re.escape(term)}\b", self.log_file)
        positions = [(match.start(), match.end()) for match in matches]
        if len(positions) > 1:
            warnings.warn(
                "Something is wrong with the log file; maybe stacked together?"
            )
        if len(positions) == 0:
            warnings.warn("Log file created but first scf loop not reached")
            return None
        log_main = positions[-1][-1] + 1
        return self.log_file[log_main:]

    def job_finished(self) -> bool:
        if (
            len(re.findall("Program exited normally.", self.log_file, re.MULTILINE))
            == 0
        ):
            warnings.warn("scf loops did not converge")
            return False
        return True

    def get_n_valence(self) -> dict:
        log = self.log_file.split("\n")
        return {
            log[ii - 1].split()[1]: int(ll.split("=")[-1])
            for ii, ll in enumerate(log)
            if ll.startswith("| Z=")
        }

    @property
    def _log_k_points(self) -> list:
        start_match = re.search(
            r"-ik-     -x-      -y-       -z-    \|  -weight-    -nG-    -label-",
            self.log_file,
        )
        assert start_match is not None
        log_part = self.log_file[start_match.end() + 1 :]
        end_match = re.search("^\n", log_part, re.MULTILINE)
        assert end_match is not None
        log_part = log_part[: end_match.start()]
        return log_part.split("\n")[:-2]

    def get_bands_k_weights(self) -> NDArray:
        return np.array([float(kk.split()[6]) for kk in self._log_k_points])

    @property
    def _rec_cell(self) -> NDArray:
        log_extract = re.findall("b[1-3]:.*$", self.log_file, re.MULTILINE)
        return (np.array([ll.split()[1:4] for ll in log_extract]).astype(float))[:3]

    def get_kpoints_cartesian(self) -> NDArray:
        return np.einsum("ni,ij->nj", self.k_points, self._rec_cell)

    @property
    def k_points(self) -> NDArray:
        return np.array(
            [[float(kk.split()[i]) for i in range(2, 5)] for kk in self._log_k_points]
        )

    def get_volume(self) -> NDArray:
        volume_matches = re.findall("Omega:.*$", self.log_file, re.MULTILINE)
        volume: float
        if len(volume_matches) > 0:
            volume = float(volume_matches[0].split()[1])
        else:
            volume = 0.0
        return np.array(self.n_steps * [volume])

    @property
    def counter(self) -> list:
        log_main = self.log_main
        assert log_main is not None
        return [
            int(re.sub("[^0-9]", "", line.split("=")[0]))
            for line in re.findall(r"F\(.*$", log_main, re.MULTILINE)
        ]

    def _get_energy(self, pattern: str) -> list:
        log_main = self.log_main
        assert log_main is not None
        c, F = np.array(re.findall(pattern, log_main, re.MULTILINE)).T
        return _splitter(F.astype(float), c.astype(int))

    def get_energy_free(self) -> list:
        return self._get_energy(pattern=r"F\((\d+)\)=(-?\d+\.\d+)")

    def get_energy_int(self) -> list:
        return self._get_energy(pattern=r"eTot\((\d+)\)=(-?\d+\.\d+)")

    @property
    def n_atoms(self) -> int:
        if self._n_atoms is None:
            log_main = self.log_main
            assert log_main is not None
            self._n_atoms = len(
                np.unique(re.findall(r"^Species.*\{", log_main, re.MULTILINE))
            )
        return self._n_atoms

    def get_forces(self) -> Union[NDArray, list]:
        """
        Returns:
            (numpy.ndarray): Forces of the shape (n_steps, n_atoms, 3)
        """
        str_fl = r"([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)"
        pattern = r"Atom: (\d+)\t{" + ",".join(3 * [str_fl]) + r"\}"
        arr = np.array(re.findall(pattern, self.log_file))
        if len(arr) == 0:
            return []
        forces = arr[:, 1:].astype(float).reshape(-1, self.n_atoms, 3)
        if self.index_permutation is not None:
            for ii, ff in enumerate(forces):
                forces[ii] = ff[self.index_permutation]
        return forces

    def get_magnetic_forces(self) -> list:
        """
        Returns:
            (numpy.ndarray): Magnetic forces of the shape (n_steps, n_atoms)
        """
        log_main = self.log_main
        assert log_main is not None
        magnetic_forces: Union[list, NDArray] = [
            float(line.split()[-1])
            for line in re.findall(r"^nu\(.*$", log_main, re.MULTILINE)
        ]
        if len(magnetic_forces) != 0:
            magnetic_forces = np.array(magnetic_forces).reshape(-1, self.n_atoms)
            if self.index_permutation is not None:
                for ii, mm in enumerate(magnetic_forces):
                    magnetic_forces[ii] = mm[self.index_permutation]
        return _splitter(magnetic_forces, self.counter)

    @property
    def n_steps(self) -> int:
        return len(re.findall(r"\| SCF calculation", self.log_file, re.MULTILINE))

    def _parse_band(self, term: str) -> Union[NDArray, list]:
        log_main = self.log_main
        assert log_main is not None
        content = re.findall(term, log_main, re.MULTILINE)
        if len(content) == 0:
            return []
        arr = np.loadtxt(content, ndmin=2)
        n_k = len(self.k_points)
        n_bands = arr.shape[-1]
        shape: tuple[int, ...]
        if self.spin_enabled:
            shape = (-1, 2, n_k, n_bands)
        else:
            shape = (-1, n_k, n_bands)
        return arr.reshape(shape)

    def get_band_energy(self) -> Union[NDArray, list]:
        return self._parse_band(r"final eig \[eV\]:(.*)$")

    def get_occupancy(self) -> Union[NDArray, list]:
        return self._parse_band("final focc:(.*)$")

    def get_convergence(self) -> list:
        log_main = self.log_main
        assert log_main is not None
        conv_dict = {
            "WARNING: Maximum number of steps exceeded": False,
            "Convergence reached.": True,
        }
        key = "|".join(list(conv_dict.keys())).replace(".", r"\.")
        items = re.findall(key, log_main, re.MULTILINE)
        convergence = [conv_dict[k] for k in items]
        diff = self.n_steps - len(convergence)
        for _ in range(diff):
            convergence.append(False)
        return convergence

    def get_fermi(self) -> NDArray:
        log_main = self.log_main
        assert log_main is not None
        pattern = r"Fermi energy:\s+(-?\d+\.\d+)\s+eV"
        return np.array(re.findall(pattern, log_main)).astype(float)

    @property
    def results(self) -> dict:
        if self.log_main is None:
            return {}
        results: dict = {"generic": {}, "dft": {}}
        for key, func in self.generic_dict.items():
            value = func()
            if key == "job_finished" or len(value) > 0:
                results["generic"][key] = value
        for key, func in self.dft_dict.items():
            value = func()
            if len(value) > 0:
                results["dft"][key] = value
        return results
