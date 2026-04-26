# sphinx_parser

Python interface for the [Sphinx DFT code](https://sxrepo.mpie.de) — a high-performance density functional theory (DFT) package developed at the Max Planck Institute.

## What this project does

- Provides a Python API to construct Sphinx input files programmatically
- Integrates with ASE (Atomic Simulation Environment) as a standard `FileIOCalculator`
- Auto-generates input classes from a YAML specification (`sphinx_parser/src/input_data.yml`)
- Parses Sphinx output files (logs, energies, forces)
- Handles unit conversions between Hartree/Bohr (Sphinx) and eV/Å (ASE)

## Data flow

```
input_data.yml → generator.py → input.py (generated, do not hand-edit)
                                    ↓
                              jobs.py (convenience wrappers)
                                    ↓
                            toolkit.py (serialisation to .sx format)
                                    ↓
                        calculator.py or manual file write
                                    ↓
                              input.sx → [Sphinx binary]
                                    ↓
                   sphinx.log / energy.dat / forces.sx
                                    ↓
                              output.py (SphinxLogParser)
```

## Key files

| File | Role |
|------|------|
| `sphinx_parser/input.py` | Auto-generated input classes — **do not edit by hand** |
| `sphinx_parser/src/input_data.yml` | Source-of-truth YAML spec for all Sphinx input parameters |
| `sphinx_parser/src/generator.py` | Reads the YAML and regenerates `input.py` |
| `sphinx_parser/calculator.py` | ASE `FileIOCalculator` subclass (`SphinxDft`) |
| `sphinx_parser/jobs.py` | High-level helpers (`set_base_parameters`, `apply_minimization`) |
| `sphinx_parser/toolkit.py` | Low-level formatting (`to_sphinx`, `format_value`, `fill_values`) |
| `sphinx_parser/ase.py` | ASE ↔ Sphinx structure conversion |
| `sphinx_parser/output.py` | `SphinxLogParser` — parses log files into result dicts |
| `sphinx_parser/potential.py` | PAW potential lookup (VASP and JTH formats) |

## Regenerating input.py

```bash
python sphinx_parser/src/generator.py
```

Run this after editing `input_data.yml`.

## Tests

```bash
python -m pytest tests/
```

- `tests/unit/` — unit tests for jobs, output parsing, generator
- `tests/integration/` — integration tests including README examples
- `tests/static/` — static reference data

CI runs on Python 3.11–3.14 via GitHub Actions.

## Code style

```bash
black sphinx_parser/
ruff check sphinx_parser/
```

Both are enforced in CI.

## Dependencies

- **numpy**, **ase** — core scientific stack
- **semantikon**, **pint** — unit handling
- **h5py** — HDF5 I/O
- **pyyaml** — YAML parsing

## Active branch: `calculator`

This branch adds `calculator.py` — the new ASE `FileIOCalculator` interface. It is not yet on `main`.

## Notes

- Units inside Sphinx are Hartree / Bohr; the calculator layer converts to eV / Å for ASE.
- Atom ordering in Sphinx input must group species together; `sphinx_parser/ase.py` handles this reordering.
- Magnetic moments and constraints are also handled in `ase.py`.
