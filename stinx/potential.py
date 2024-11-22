import os


def get_potential_path(element: str):
    """
    Get the path to the potential file for the given element.

    Args:
        element (str): The element symbol.

    Returns:
        str: The path to the potential file.
    """
    path = os.path.join(os.getenv("CONDA_PREFIX"), "share", "sphinxdft", "jth-gga-pbe")
    return os.path.join(path,  f"{element}_GGA.atomicdata")
