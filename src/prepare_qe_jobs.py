"""Prepare validated hypothetical structures for Quantum ESPRESSO.

Creates pw.x vc-relax and scf starting inputs.

This script does not:
- run DFT,
- download pseudopotentials,
- generate pseudopotentials.

Pseudopotentials must already exist in the configured pseudo directory.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

VALIDATION_FILE = (
    ROOT
    / "results"
    / "hypothetical_structures"
    / "structure_validation.csv"
)

DEFAULT_OUTPUT = ROOT / "results" / "qe_jobs"

DEFAULT_PSEUDO_ROOT = ROOT / "results" / "qe_pseudopotentials"


RELAX = """&CONTROL
  calculation = 'vc-relax'
  prefix = '{prefix}'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tstress = .true.
  tprnfor = .true.
/

&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = 60
  ecutrho = 480
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/

&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.3
/

&IONS
  ion_dynamics = 'bfgs'
/

&CELL
  cell_dynamics = 'bfgs'
  press = 0.0
  cell_dofree = 'all'
/

ATOMIC_SPECIES
{species}

CELL_PARAMETERS angstrom
{cell}

ATOMIC_POSITIONS crystal
{positions}

K_POINTS automatic
{kpoints}
"""


SCF = """&CONTROL
  calculation = 'scf'
  prefix = '{prefix}'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tstress = .true.
  tprnfor = .true.
/

&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = 60
  ecutrho = 480
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/

&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
/

ATOMIC_SPECIES
{species}

CELL_PARAMETERS angstrom
{cell}

ATOMIC_POSITIONS crystal
{positions}

K_POINTS automatic
{kpoints}
"""


def args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Quantum ESPRESSO starting inputs from "
            "validated hypothetical structures."
        )
    )

    parser.add_argument(
        "--validation-file",
        type=Path,
        default=VALIDATION_FILE,
        help="Path to structure_validation.csv",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for QE jobs",
    )

    parser.add_argument(
        "--status",
        choices=["all", "valid", "valid_with_warnings"],
        default="all",
        help="Which validation statuses to include",
    )

    parser.add_argument(
        "--pseudo-root",
        type=Path,
        default=DEFAULT_PSEUDO_ROOT,
        help="Directory containing .UPF/.upf pseudopotentials",
    )

    parser.add_argument(
        "--strict-pseudo",
        action="store_true",
        help=(
            "Fail if a required element pseudopotential is missing "
            "or ambiguous."
        ),
    )

    return parser.parse_args()


def root(path):
    """Convert a relative project path into an absolute path."""
    path = Path(path)

    if path.is_absolute():
        return path

    return ROOT / path


def read_poscar(path):
    """Read a VASP POSCAR and return lattice and fractional coordinates."""

    lines = path.read_text(encoding="utf-8").splitlines()

    if len(lines) < 8:
        raise ValueError(f"Invalid POSCAR: {path}")

    scale = float(lines[1])

    if abs(scale - 1.0) > 1e-10:
        raise ValueError(
            "Expected POSCAR scale factor 1.0."
        )

    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]

    if len(elements) != len(counts):
        raise ValueError(
            f"Element/count mismatch in POSCAR: {path}"
        )

    i = 7

    # Optional Selective Dynamics line
    if lines[i].strip().lower().startswith("s"):
        i += 1

    mode = lines[i].strip().lower()
    i += 1

    lattice = [
        [float(x) for x in lines[2 + j].split()[:3]]
        for j in range(3)
    ]

    n_atoms = sum(counts)

    raw = [
        [float(x) for x in lines[i + j].split()[:3]]
        for j in range(n_atoms)
    ]

    if mode.startswith("d"):
        frac = raw

    elif mode.startswith(("c", "k")):
        import numpy as np

        lattice_array = np.array(lattice, dtype=float)
        raw_array = np.array(raw, dtype=float)

        frac = (
            raw_array
            @ np.linalg.inv(lattice_array.T)
        ).tolist()

    else:
        raise ValueError(
            f"Unsupported POSCAR coordinate mode: {mode}"
        )

    positions = []

    n = 0

    for element, count in zip(elements, counts):
        for j in range(count):
            positions.append(
                (element, frac[n + j])
            )

        n += count

    return elements, counts, lattice, positions


def mesh(row):
    """Generate a simple starting k-point mesh from lattice lengths."""

    values = []

    for key in ("a_A", "b_A", "c_A"):
        try:
            value = max(float(row[key]), 1.0)
        except (TypeError, ValueError):
            value = 10.0

        k = round(25.0 / value)

        k = max(1, min(8, k))

        values.append(k)

    return tuple(values)


def canonical_element(symbol):
    """Convert an element symbol to canonical form.

    Examples:
        'al' -> 'Al'
        'AL' -> 'Al'
        'rb' -> 'Rb'
    """

    symbol = str(symbol).strip()

    if not symbol:
        return symbol

    return symbol[0].upper() + symbol[1:].lower()


def find_pseudopotentials(elements, pseudo_root):
    """Find exactly one pseudopotential for each requested element.

    Matching is based on the actual element symbol at the beginning
    of the filename.

    This avoids incorrect prefix matching such as:

        I  -> I... and In...
        N  -> N... and Na/Nb/Ni/...
        K  -> K... and Kr...

    Both .UPF and .upf extensions are supported.

    The implementation is also safe on Windows, where filesystem
    matching is case-insensitive and glob('*.UPF') and glob('*.upf')
    can return the same files.
    """

    pseudo_root = root(pseudo_root)

    if not pseudo_root.exists():
        raise FileNotFoundError(
            f"Pseudopotential directory not found: {pseudo_root}"
        )

    if not pseudo_root.is_dir():
        raise NotADirectoryError(
            f"Pseudopotential path is not a directory: {pseudo_root}"
        )

    # Collect every .UPF/.upf file exactly once.
    #
    # Using iterdir() instead of:
    #
    #   glob("*.UPF") + glob("*.upf")
    #
    # avoids duplicate files on Windows.
    files = [
        path
        for path in pseudo_root.iterdir()
        if path.is_file()
        and path.suffix.lower() == ".upf"
    ]

    wanted = {
        canonical_element(element)
        for element in elements
    }

    candidates = {
        element: []
        for element in wanted
    }

    for path in files:
        name = path.name

        # Match exactly one or two letters at the beginning,
        # followed by a filename delimiter.
        #
        # Examples:
        #
        # I.pbe...       -> I
        # In.pbe...      -> In
        # N.pbe...       -> N
        # br_pbe...      -> Br
        # Rb_ONCV...     -> Rb
        #
        # The delimiter is important because it prevents
        # "I" from matching "In".
        match = re.match(
            r"^([A-Za-z]{1,2})(?=[._-])",
            name,
        )

        if not match:
            continue

        symbol = canonical_element(match.group(1))

        if symbol in candidates:
            candidates[symbol].append(path)

    pseudo_map = {}
    missing_pseudo = []
    ambiguous_pseudo = []

    for element in elements:
        element = canonical_element(element)

        matches = candidates.get(element, [])

        if len(matches) == 0:
            missing_pseudo.append(element)

        elif len(matches) > 1:
            ambiguous_pseudo.append(element)

        else:
            pseudo_map[element] = matches[0]

    return (
        pseudo_map,
        missing_pseudo,
        ambiguous_pseudo,
    )


def make_input(
    row,
    out,
    pseudo_root,
    strict_pseudo,
):
    """Create one Quantum ESPRESSO job package."""

    formula = str(row["formula"])

    idx = int(row["prototype_index"])

    source = root(Path(str(row["poscar"])))

    if not source.exists():
        raise FileNotFoundError(
            f"POSCAR not found: {source}"
        )

    elements, counts, lattice, positions = read_poscar(
        source
    )

    elements = [
        canonical_element(element)
        for element in elements
    ]

    job = (
        out
        / formula
        / f"prototype_{idx}"
    )

    job.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        job / "POSCAR",
    )

    cif = source.with_name("structure.cif")

    if cif.exists():
        shutil.copy2(
            cif,
            job / "structure.cif",
        )

    ka, kb, kc = mesh(row)

    (
        pseudo_map,
        missing_pseudo,
        ambiguous_pseudo,
    ) = find_pseudopotentials(
        elements,
        pseudo_root,
    )

    if strict_pseudo and (
        missing_pseudo
        or ambiguous_pseudo
    ):
        raise RuntimeError(
            f"{formula} prototype {idx}: "
            "pseudopotential issue; "
            f"missing={missing_pseudo}, "
            f"ambiguous={ambiguous_pseudo}"
        )

    job_pseudo = job / "pseudo"

    job_pseudo.mkdir(
        exist_ok=True
    )

    pseudo_names = {}

    for element in elements:

        if element in pseudo_map:

            source_pseudo = pseudo_map[element]

            destination = (
                job_pseudo
                / source_pseudo.name
            )

            shutil.copy2(
                source_pseudo,
                destination,
            )

            pseudo_names[element] = (
                source_pseudo.name
            )

        else:

            # Placeholder only for non-strict mode.
            pseudo_names[element] = (
                f"{element}.UPF"
            )

    species = "\n".join(
        f"{element} 1.0 {pseudo_names[element]}"
        for element in elements
    )

    cell = "\n".join(
        " ".join(
            f"{value:.10f}"
            for value in vector
        )
        for vector in lattice
    )

    pos = "\n".join(
        f"{element} "
        f"{coordinates[0]:.10f} "
        f"{coordinates[1]:.10f} "
        f"{coordinates[2]:.10f}"
        for element, coordinates in positions
    )

    data = {
        "prefix": f"{formula}_p{idx}",
        "nat": sum(counts),
        "ntyp": len(elements),
        "species": species,
        "cell": cell,
        "positions": pos,
        "kpoints": f"{ka} {kb} {kc} 0 0 0",
    }

    (
        job / "pw_relax.in"
    ).write_text(
        RELAX.format(**data),
        encoding="utf-8",
    )

    (
        job / "pw_scf.in"
    ).write_text(
        SCF.format(**data),
        encoding="utf-8",
    )

    warnings = (
        str(row.get("warnings", "")).strip()
        or "none"
    )

    metadata = {
        "formula": formula,
        "prototype_index": idx,
        "prototype_source_jid": str(
            row["prototype_source_jid"]
        ),
        "prototype_source_formula": str(
            row["prototype_source_formula"]
        ),
        "prototype": str(
            row["prototype"]
        ),
        "lattice_scale": float(
            row["lattice_scale"]
        ),
        "validation_status": str(
            row["status"]
        ),
        "validation_warnings": warnings,
        "kpoint_mesh": [
            ka,
            kb,
            kc,
        ],
        "ecutwfc_Ry_start": 60,
        "ecutrho_Ry_start": 480,
        "pseudopotentials": pseudo_names,
        "pseudopotential_missing_elements": (
            missing_pseudo
        ),
        "pseudopotential_ambiguous_elements": (
            ambiguous_pseudo
        ),
        "scientific_status": (
            "prototype-transferred initial "
            "structure; not relaxed or "
            "stability-validated"
        ),
    }

    (
        job / "metadata.json"
    ).write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    readme = f"""Quantum ESPRESSO starting package

Formula: {formula}

Prototype: {idx}

Source: {row["prototype_source_jid"]} ({row["prototype_source_formula"]})

Validation: {row["status"]}

Warnings: {warnings}

Files:

- pw_relax.in: vc-relax starting input
- pw_scf.in: SCF starting input
- POSCAR: source structure
- structure.cif: source CIF when available
- pseudo/: selected .UPF/.upf files
- metadata.json: provenance

IMPORTANT:

Pseudopotentials are selected from
results/qe_pseudopotentials when exactly one
matching pseudopotential exists for each element.

Pseudopotentials are not downloaded or generated
by this script.

Use one internally consistent pseudopotential family
and exchange-correlation functional for production
calculations.

The cutoff (60/480 Ry), k-point mesh, smearing and
other settings are starting values only.

Perform convergence tests before production DFT.

The structure is hypothetical and prototype-transferred.
Stability, synthesizability and experimental
realizability are not established.

This script prepares Quantum ESPRESSO inputs only.
It does not run pw.x.
"""

    (
        job / "README.txt"
    ).write_text(
        readme,
        encoding="utf-8",
    )

    return {
        "formula": formula,
        "prototype_index": idx,
        "validation_status": str(
            row["status"]
        ),
        "prototype_source_jid": str(
            row["prototype_source_jid"]
        ),
        "prototype_source_formula": str(
            row["prototype_source_formula"]
        ),
        "lattice_scale": float(
            row["lattice_scale"]
        ),
        "kpoints": (
            f"{ka} {kb} {kc}"
        ),
        "job_directory": str(
            job.relative_to(ROOT)
        ),
        "pw_relax": str(
            (
                job / "pw_relax.in"
            ).relative_to(ROOT)
        ),
        "pw_scf": str(
            (
                job / "pw_scf.in"
            ).relative_to(ROOT)
        ),
        "pseudo_directory": str(
            (
                job / "pseudo"
            ).relative_to(ROOT)
        ),
        "warnings": warnings,
    }


def main():
    """Main program."""

    a = args()

    validation_file = root(
        a.validation_file
    )

    output_root = root(
        a.output_root
    )

    pseudo_root = root(
        a.pseudo_root
    )

    if not validation_file.exists():
        raise FileNotFoundError(
            f"Validation file not found: "
            f"{validation_file}"
        )

    df = pd.read_csv(
        validation_file
    )

    # Only structurally validated structures
    # are allowed into QE preparation.
    df = df[
        df["status"].isin(
            [
                "valid",
                "valid_with_warnings",
            ]
        )
    ].copy()

    if a.status != "all":
        df = df[
            df["status"] == a.status
        ].copy()

    if df.empty:
        raise RuntimeError(
            "No structures match the "
            "requested validation status."
        )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Structures to prepare: {len(df)}"
    )

    rows = []

    for _, row in df.iterrows():

        formula = str(
            row["formula"]
        )

        prototype_index = int(
            row["prototype_index"]
        )

        print(
            f"Preparing {formula} / "
            f"prototype {prototype_index}..."
        )

        result = make_input(
            row,
            output_root,
            pseudo_root,
            a.strict_pseudo,
        )

        rows.append(result)

        print(
            "  [OK] Quantum ESPRESSO "
            "starting package prepared"
        )

    manifest = pd.DataFrame(
        rows
    )

    manifest_file = (
        output_root
        / "qe_job_manifest.csv"
    )

    summary_file = (
        output_root
        / "qe_job_summary.json"
    )

    manifest.to_csv(
        manifest_file,
        index=False,
    )

    summary = {
        "source_validation_file": str(
            validation_file.relative_to(ROOT)
        ),
        "output_root": str(
            output_root.relative_to(ROOT)
        ),
        "prepared_structures": len(
            manifest
        ),
        "valid": int(
            (
                manifest[
                    "validation_status"
                ]
                == "valid"
            ).sum()
        ),
        "valid_with_warnings": int(
            (
                manifest[
                    "validation_status"
                ]
                == "valid_with_warnings"
            ).sum()
        ),
        "pseudopotentials_generated": False,
        "interpretation": (
            "Quantum ESPRESSO input preparation "
            "only; convergence and pseudopotential "
            "checks are required."
        ),
        "pseudopotential_root": str(
            pseudo_root.relative_to(ROOT)
        ),
        "strict_pseudopotential_mode": bool(
            a.strict_pseudo
        ),
        "manifest_file": str(
            manifest_file.relative_to(ROOT)
        ),
    }

    summary_file.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        "\nQuantum ESPRESSO "
        "job preparation complete."
    )

    print(
        f"Prepared: "
        f"{summary['prepared_structures']}"
    )

    print(
        f"Manifest: {manifest_file}"
    )

    print(
        f"Summary: {summary_file}"
    )


if __name__ == "__main__":
    main()