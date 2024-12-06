from sphinx_parser.input import sphinx
from sphinx_parser.ase import get_structure_group
from sphinx_parser.potential import get_paw_from_structure


def calc_static(
    structure,
    eCut=25,
    xc=1,
    spinPolarized=False,
    maxSteps=30,
    ekt=0.2,
    k_point_coords=[0.5, 0.5, 0.5],
):
    struct_group, spin_lst = get_structure_group(structure)
    main_group = sphinx.main.create(
        scfDiag=sphinx.main.scfDiag.create(
            maxSteps=maxSteps, blockCCG=sphinx.main.scfDiag.blockCCG.create()
        )
    )
    pawPot_group = get_paw_from_structure(structure)
    basis_group = sphinx.basis.create(
        eCut=eCut, kPoint=sphinx.basis.kPoint.create(coords=k_point_coords)
    )
    paw_group = sphinx.PAWHamiltonian.create(
        xc=xc, spinPolarized=spinPolarized, ekt=ekt
    )
    initial_guess_group = sphinx.initialGuess.create(
        waves=sphinx.initialGuess.waves.create(
            lcao=sphinx.initialGuess.waves.lcao.create()
        ),
        rho=sphinx.initialGuess.rho.create(
            atomicOrbitals=True, atomicSpin=spin_lst
        ),
    )
    input_sx = sphinx.create(
        pawPot=pawPot_group,
        structure=struct_group,
        main=main_group,
        basis=basis_group,
        PAWHamiltonian=paw_group,
        initialGuess=initial_guess_group,
    )
    return input_sx
