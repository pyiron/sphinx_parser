"""
Small tests run, relatively fast tests for checking individual bits of the code base.
"""
from pathlib import Path  
import numpy as np        

def test_collect_occ_dat(file_name="hubbardOcc.dat", cwd=None):
    """
    Collect orbital occupancy data from the Hubbard DFT+U calculation.

    Args:
        file_name (str): Name of the file containing Hubbard occupancy data (default: "hubbardOcc.dat").
        cwd (str): Directory path where the file is located (default: None, meaning the current directory).

    Returns:
        dict: A dictionary with keys for each atom with applied Hubbard U (e.g., "Uatom_1", "Uatom_2", ...)
              and their sorted occupancy data.
    """
    path = Path(file_name)
    if cwd is not None:
        path = Path(cwd) / path

    try:
        occupancy_data = np.loadtxt(path)
    except OSError:
        raise FileNotFoundError(f"File '{file_name}' not found in '{cwd or Path.cwd()}'.")

    num_lines_per_group = 10
    grouped_data = [
        sorted(occupancy_data[i:i + num_lines_per_group, 0])
        for i in range(0, len(occupancy_data), num_lines_per_group)
    ]

    # Create a dictionary with atom labels as keys
    Uatoms = {f"Uatom_{i+1}": group for i, group in enumerate(grouped_data)}

    return Uatoms
