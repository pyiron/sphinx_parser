from sphinx_parser.ase import get_structure_group
from sphinx_parser.input import sphinx
from sphinx_parser.potential import get_paw_from_structure


def set_base_parameters(
    structure,
    eCut: float = 25,
    xc: int = 1,
    maxSteps: int = 30,
    ekt: float = 0.2,
    k_point_coords: list = [0.5, 0.5, 0.5],
):
    """
    Set the base parameters for the sphinx input file

    Args:
        structure (ase.Atoms): ASE Atoms object
        eCut (float, optional): Energy cutoff. Defaults to 25.
        xc (int, optional): Exchange-correlation functional. Defaults to 1.
        maxSteps (int, optional): Maximum number of steps. Defaults to 30.
        ekt (float, optional): Temperature. Defaults to 0.2.
        k_point_coords (list, optional): K-point coordinates. Defaults to [0.5, 0.5, 0.5].

    Returns:
        dict: Sphinx input dictionary
    """
    struct_group, spin_lst = get_structure_group(structure)
    spinPolarized = spin_lst is not None
    main_group = sphinx.main(
        scfDiag=sphinx.main.scfDiag(
            maxSteps=maxSteps, blockCCG=sphinx.main.scfDiag.blockCCG()
        )
    )
    pawPot_group = get_paw_from_structure(structure)
    basis_group = sphinx.basis(
        eCut=eCut, kPoint=sphinx.basis.kPoint(coords=k_point_coords)
    )
    paw_group = sphinx.PAWHamiltonian(xc=xc, spinPolarized=spinPolarized, ekt=ekt)
    initial_guess_group = sphinx.initialGuess(
        waves=sphinx.initialGuess.waves(lcao=sphinx.initialGuess.waves.lcao()),
        rho=sphinx.initialGuess.rho(atomicOrbitals=True, atomicSpin=spin_lst),
    )
    input_sx = sphinx(
        pawPot=pawPot_group,
        structure=struct_group,
        main=main_group,
        basis=basis_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
    )
    return input_sx


def apply_minimization(sphinx_input, mode="linQN", dEnergy=1.0e-6, maxSteps=50):
    """
    Apply minimization to the sphinx input file

    Args:
        sphinx_input (dict): Sphinx input dictionary
        mode (str, optional): Minimization mode. Defaults to "linQN".
        dEnergy (float, optional): Energy tolerance. Defaults to 1.0e-6.
        maxSteps (int, optional): Maximum number of steps. Defaults to 50.

    Returns:
        dict: Sphinx input dictionary
    """
    input_sx = sphinx_input.copy()
    if "main" not in input_sx or "scfDiag" not in input_sx["main"]:
        raise ValueError("main group not found - run set_base_parameters first")
    if mode == "linQN":
        input_sx["main"] = sphinx.main(
            linQN=sphinx.main.linQN(
                dEnergy=dEnergy,
                maxSteps=maxSteps,
                bornOppenheimer=sphinx.main.linQN.bornOppenheimer(
                    scfDiag=input_sx["main"]["scfDiag"]
                ),
            )
        )
    elif mode == "QN":
        input_sx["main"] = sphinx.main(
            QN=sphinx.main.QN(
                dEnergy=dEnergy,
                maxSteps=maxSteps,
                bornOppenheimer=sphinx.main.QN.bornOppenheimer(
                    scfDiag=input_sx["main"]["scfDiag"]
                ),
            )
        )
    elif mode == "ricQN":
        input_sx["main"] = sphinx.main(
            ricQN=sphinx.main.ricQN(
                dEnergy=dEnergy,
                maxSteps=maxSteps,
                bornOppenheimer=sphinx.main.ricQN.bornOppenheimer(
                    scfDiag=input_sx["main"]["scfDiag"]
                ),
            )
        )
    elif mode == "ricTS":
        input_sx["main"] = sphinx.main(
            ricTS=sphinx.main.ricTS(
                dEnergy=dEnergy,
                maxSteps=maxSteps,
                bornOppenheimer=sphinx.main.ricTS.bornOppenheimer(
                    scfDiag=input_sx["main"]["scfDiag"]
                ),
            )
        )
    else:
        raise ValueError("mode not recognized")
    return input_sx
