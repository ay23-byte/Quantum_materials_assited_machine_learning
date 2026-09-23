"""Validate prototype-transferred hypothetical crystal structures.

This is a pre-DFT structural sanity-check stage. It validates generated
POSCAR files against the intended composition and checks basic geometry:
cell validity, finite coordinates, periodic minimum distances, and optional
large lattice scaling flags from the generation manifest.

It does NOT establish thermodynamic stability, dynamical stability,
synthesizability, or a ground-state structure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from itertools import product

import numpy as np
import pandas as pd

try:
    from src.descriptors import parse_formula
except ModuleNotFoundError:
    from descriptors import parse_formula

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = ROOT / "results" / "hypothetical_structures" / "structure_generation_manifest.csv"
OUTPUT_ROOT = ROOT / "results" / "hypothetical_structures"
VALIDATION_FILE = OUTPUT_ROOT / "structure_validation.csv"
SUMMARY_FILE = OUTPUT_ROOT / "structure_validation_summary.json"

# Conservative warning thresholds for initial structures.
MIN_DISTANCE_WARN_ANGSTROM = 1.20
MAX_LATTICE_SCALE_WARN = 1.25


def reduce_counts(raw: dict[str, int]) -> dict[str, int]:
    g = 0
    for value in raw.values():
        g = math.gcd(g, int(value))
    if g <= 0:
        raise ValueError("Could not reduce composition counts.")
    return {element: int(value) // g for element, value in raw.items()}


def expected_counts(formula: str) -> dict[str, int]:
    raw = {str(k): int(round(float(v))) for k, v in parse_formula(formula).items()}
    if not raw or any(v <= 0 for v in raw.values()):
        raise ValueError(f"Invalid composition: {formula}")
    return reduce_counts(raw)


def parse_poscar(path: Path) -> dict:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 8:
        raise ValueError("POSCAR has too few non-empty lines")

    scale = float(lines[1])
    if not math.isfinite(scale) or scale == 0:
        raise ValueError("Invalid POSCAR scale factor")

    lattice = np.array(
        [[float(x) for x in lines[i].split()[:3]] for i in range(2, 5)],
        dtype=float,
    )
    lattice *= scale

    if not np.all(np.isfinite(lattice)):
        raise ValueError("Non-finite lattice vectors")

    volume = abs(float(np.linalg.det(lattice)))
    if volume <= 1e-6:
        raise ValueError(f"Non-positive/degenerate cell volume: {volume}")

    species = lines[5].split()
    try:
        counts = [int(x) for x in lines[6].split()]
    except ValueError as exc:
        raise ValueError("Invalid POSCAR atom counts") from exc

    if len(species) != len(counts):
        raise ValueError("Species/count length mismatch")

    idx = 7
    if lines[idx].lower().startswith("selective"):
        idx += 1

    if idx >= len(lines):
        raise ValueError("Missing coordinate mode")
    mode = lines[idx].lower()
    idx += 1

    if mode.startswith("d"):
        direct = True
    elif mode.startswith("c") or mode.startswith("k"):
        direct = False
    else:
        raise ValueError(f"Unknown coordinate mode: {lines[idx - 1]}")

    n_atoms = sum(counts)
    if len(lines) < idx + n_atoms:
        raise ValueError("POSCAR contains fewer coordinates than expected")

    coords = np.array(
        [[float(x) for x in lines[idx + i].split()[:3]] for i in range(n_atoms)],
        dtype=float,
    )
    if not np.all(np.isfinite(coords)):
        raise ValueError("Non-finite atomic coordinates")

    if direct:
        frac = coords
    else:
        # Cartesian coordinates are related to row-vector lattice convention:
        # cart = frac @ lattice.
        frac = coords @ np.linalg.inv(lattice)

    if not np.all(np.isfinite(frac)):
        raise ValueError("Could not convert coordinates to finite fractional values")

    elements = []
    for element, count in zip(species, counts):
        elements.extend([element] * count)

    return {
        "lattice": lattice,
        "volume": volume,
        "species": species,
        "counts": counts,
        "elements": elements,
        "frac": frac,
    }


def cell_lengths(lattice: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(v)) for v in lattice)


def cell_angles(lattice: np.ndarray) -> tuple[float, float, float]:
    a, b, c = lattice

    def angle(v1, v2) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        cosine = float(np.dot(v1, v2) / denom)
        cosine = max(-1.0, min(1.0, cosine))
        return math.degrees(math.acos(cosine))

    return angle(b, c), angle(a, c), angle(a, b)


def minimum_periodic_distance(frac: np.ndarray, lattice: np.ndarray) -> float:
    """Return the minimum pair distance including periodic images."""
    min_distance = float("inf")
    translations = list(product((-1, 0, 1), repeat=3))

    for i in range(len(frac)):
        for j in range(i, len(frac)):
            for t in translations:
                if i == j and t == (0, 0, 0):
                    continue
                delta = frac[j] + np.asarray(t, dtype=float) - frac[i]
                cart = delta @ lattice
                distance = float(np.linalg.norm(cart))
                if distance > 1e-8:
                    min_distance = min(min_distance, distance)

    return min_distance


def validate_row(row: pd.Series) -> dict:
    formula = str(row["formula"])
    poscar_path = ROOT / str(row["poscar"])
    cif_path = ROOT / str(row["cif"])

    result = {
        "formula": formula,
        "shortlist_rank": int(row["shortlist_rank"]),
        "prototype_index": int(row["prototype_index"]),
        "prototype_source_jid": str(row["prototype_source_jid"]),
        "prototype_source_formula": str(row["prototype_source_formula"]),
        "prototype": str(row["prototype"]),
        "lattice_scale": float(row["lattice_scale"]),
        "poscar": str(row["poscar"]),
        "cif": str(row["cif"]),
        "status": "valid",
        "issues": "",
        "warnings": "",
        "atom_count": None,
        "expected_atom_count": None,
        "cell_volume_A3": None,
        "a_A": None,
        "b_A": None,
        "c_A": None,
        "alpha_deg": None,
        "beta_deg": None,
        "gamma_deg": None,
        "minimum_periodic_distance_A": None,
        "composition_match": False,
        "poscar_readable": False,
        "cif_nonempty": False,
    }

    issues = []
    warnings = []

    try:
        expected = expected_counts(formula)
        result["expected_atom_count"] = sum(expected.values())
    except Exception as exc:
        issues.append(f"invalid target composition: {exc}")
        expected = {}

    if not poscar_path.exists():
        issues.append("POSCAR file missing")
        result["status"] = "invalid"
        result["issues"] = "; ".join(issues)
        return result

    if not cif_path.exists():
        issues.append("CIF file missing")

    if cif_path.exists() and cif_path.stat().st_size > 0:
        result["cif_nonempty"] = True
    elif cif_path.exists():
        issues.append("CIF file is empty")

    try:
        data = parse_poscar(poscar_path)
        result["poscar_readable"] = True
        result["atom_count"] = len(data["elements"])

        actual = {}
        for element, count in zip(data["species"], data["counts"]):
            actual[element] = actual.get(element, 0) + count
        actual_reduced = reduce_counts(actual)

        if actual_reduced != expected:
            issues.append(
                f"composition mismatch: POSCAR reduced={actual_reduced}, "
                f"expected reduced={expected}; raw POSCAR={actual}"
            )
        else:
            result["composition_match"] = True

        result["cell_volume_A3"] = data["volume"]
        a, b, c = cell_lengths(data["lattice"])
        alpha, beta, gamma = cell_angles(data["lattice"])
        result["a_A"], result["b_A"], result["c_A"] = a, b, c
        result["alpha_deg"], result["beta_deg"], result["gamma_deg"] = alpha, beta, gamma

        if min(a, b, c) <= 0:
            issues.append("non-positive lattice length")
        if not all(1.0 < angle < 179.0 for angle in (alpha, beta, gamma)):
            issues.append("degenerate lattice angle")

        min_dist = minimum_periodic_distance(data["frac"], data["lattice"])
        result["minimum_periodic_distance_A"] = min_dist

        if min_dist < MIN_DISTANCE_WARN_ANGSTROM:
            warnings.append(
                f"very short periodic interatomic distance: {min_dist:.3f} A"
            )

    except Exception as exc:
        issues.append(f"POSCAR validation failed: {exc}")

    scale = float(row["lattice_scale"])
    if scale > MAX_LATTICE_SCALE_WARN or scale < 1.0 / MAX_LATTICE_SCALE_WARN:
        warnings.append(
            f"large lattice scaling factor: {scale:.3f}"
        )

    if issues:
        result["status"] = "invalid"
    elif warnings:
        result["status"] = "valid_with_warnings"

    result["issues"] = "; ".join(issues)
    result["warnings"] = "; ".join(warnings)
    return result


def main() -> None:
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}. Run structure generation first."
        )

    manifest = pd.read_csv(MANIFEST_FILE)
    manifest = manifest[
        manifest["status"].astype(str) == "prototype_transferred_initial_structure"
    ].copy()

    print(f"Structures to validate: {len(manifest)}")

    rows = []
    for _, row in manifest.iterrows():
        label = (
            f"{row['formula']} / prototype {int(row['prototype_index'])}"
            f" / {row['prototype_source_jid']}"
        )
        print(f"Validating {label}...")
        result = validate_row(row)
        rows.append(result)

        if result["status"] == "valid":
            print("  [OK] valid")
        elif result["status"] == "valid_with_warnings":
            print(f"  [WARN] {result['warnings']}")
        else:
            print(f"  [FAILED] {result['issues']}")

    validation = pd.DataFrame(rows)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    validation.to_csv(VALIDATION_FILE, index=False)

    summary = {
        "input_manifest": str(MANIFEST_FILE.relative_to(ROOT)),
        "validated_structures": int(len(validation)),
        "valid": int((validation["status"] == "valid").sum()),
        "valid_with_warnings": int((validation["status"] == "valid_with_warnings").sum()),
        "invalid": int((validation["status"] == "invalid").sum()),
        "composition_matches": int(validation["composition_match"].sum()),
        "poscar_readable": int(validation["poscar_readable"].sum()),
        "cif_nonempty": int(validation["cif_nonempty"].sum()),
        "minimum_distance_warning_threshold_A": MIN_DISTANCE_WARN_ANGSTROM,
        "large_lattice_scale_warning_threshold": MAX_LATTICE_SCALE_WARN,
        "interpretation": (
            "Structural sanity checks only. Passing validation does not establish "
            "thermodynamic stability, dynamical stability, synthesizability, or a "
            "ground-state structure."
        ),
        "validation_file": str(VALIDATION_FILE.relative_to(ROOT)),
    }
    SUMMARY_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("
Structural validation complete.")
    print(f"Valid: {summary['valid']}")
    print(f"Valid with warnings: {summary['valid_with_warnings']}")
    print(f"Invalid: {summary['invalid']}")
    print(f"Validation: {VALIDATION_FILE}")
    print(f"Summary: {SUMMARY_FILE}")


if __name__ == "__main__":
    main()
