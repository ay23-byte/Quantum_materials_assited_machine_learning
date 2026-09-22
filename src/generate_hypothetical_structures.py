"""Generate prototype-transferred initial crystal structures for hypothetical candidates.

This is a structure-generation stage, not a stability calculation.

Method:
1. Read the ML-generated hypothetical shortlist.
2. Download/load the JARVIS-DFT 3D reference structures.
3. Find known structures with the same reduced stoichiometric prototype
   (e.g. A3BC, ABC, A2BC, etc.).
4. Transfer the reference fractional coordinates to the hypothetical
   composition, matching species by stoichiometric multiplicity and
   electronegativity ordering.
5. Uniformly scale the reference lattice using the ratio of composition-
   averaged JARVIS atomic radii.
6. Write CIF and POSCAR files plus a manifest describing the provenance.

The resulting structures are prototype-transferred starting geometries.
They are NOT predicted ground-state structures and are not evidence of
stability or synthesizability. DFT/force-field relaxation is required.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from jarvis.core.atoms import Atoms
from jarvis.core.specie import Specie
from jarvis.db.figshare import data as jarvis_data
from jarvis.io.vasp.inputs import Poscar

from src.descriptors import parse_formula

ROOT = Path(__file__).resolve().parents[1]
SHORTLIST_FILE = ROOT / "results" / "hypothetical_shortlist.csv"
OUTPUT_ROOT = ROOT / "results" / "hypothetical_structures"
MANIFEST_FILE = OUTPUT_ROOT / "structure_generation_manifest.csv"

# Same broad element families used by the shortlist stage. They are used only
# as a tie-breaker when several elements have the same stoichiometric count.
ELEMENT_FAMILY = {
    "Li": "alkali", "Na": "alkali", "K": "alkali", "Rb": "alkali", "Cs": "alkali",
    "Mg": "alkaline_earth", "Ca": "alkaline_earth", "Sr": "alkaline_earth", "Ba": "alkaline_earth",
    "Al": "post_transition", "Ga": "post_transition", "In": "post_transition",
    "Si": "metalloid", "Ge": "metalloid", "Sn": "post_transition",
    "P": "pnictogen", "As": "pnictogen", "Sb": "pnictogen", "Bi": "pnictogen",
    "N": "pnictogen", "B": "metalloid", "C": "nonmetal",
    "O": "chalcogen", "S": "chalcogen", "Se": "chalcogen", "Te": "chalcogen",
    "F": "halogen", "Cl": "halogen", "Br": "halogen", "I": "halogen",
}


def valid_float(value: object) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value) or value <= -9990:
        return None
    return value


def reduced_counts(formula: str) -> dict[str, int]:
    comp = parse_formula(formula)
    raw = {el: int(round(float(v))) for el, v in comp.items()}
    if not raw or any(v <= 0 for v in raw.values()):
        raise ValueError(f"Invalid integer composition: {formula}")
    g = 0
    for value in raw.values():
        g = math.gcd(g, value)
    return {el: value // g for el, value in raw.items()}


def prototype_from_counts(counts: dict[str, int]) -> str:
    """Return a count-only prototype such as A3BC or ABC2."""
    values = sorted(counts.values(), reverse=True)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    parts = []
    for i, count in enumerate(values):
        parts.append(letters[i] if count == 1 else f"{letters[i]}{count}")
    return "".join(parts)


def structure_reduced_counts(atoms: Atoms) -> dict[str, int]:
    counts = Counter(atoms.elements)
    raw = {str(k): int(v) for k, v in counts.items()}
    g = 0
    for value in raw.values():
        g = math.gcd(g, value)
    return {el: value // g for el, value in raw.items()}


def element_x(element: str) -> float:
    value = valid_float(Specie(element).X)
    return value if value is not None else float(Specie(element).Z)


def element_radius(element: str) -> float | None:
    value = valid_float(Specie(element).atomic_rad)
    return value


def composition_radius(formula: str) -> float | None:
    counts = parse_formula(formula)
    radii = []
    weights = []
    for element, amount in counts.items():
        radius = element_radius(element)
        if radius is not None and radius > 0:
            radii.append(radius)
            weights.append(float(amount))
    if not radii:
        return None
    return float(np.average(radii, weights=weights))


def family(element: str) -> str:
    return ELEMENT_FAMILY.get(element, "other")


def family_overlap(candidate: str, reference: str) -> int:
    return int(family(candidate) == family(reference))


def build_mapping(candidate_formula: str, reference_atoms: Atoms) -> dict[str, str]:
    """Map reference species to candidate species using reduced stoichiometry.

    For equal multiplicities, electronegativity and broad chemistry family are
    used as deterministic tie-breakers. This is a prototype-transfer heuristic.
    """
    candidate_counts = reduced_counts(candidate_formula)
    reference_counts = structure_reduced_counts(reference_atoms)

    if sorted(candidate_counts.values()) != sorted(reference_counts.values()):
        raise ValueError("Reduced stoichiometric multiplicities do not match.")

    mapping: dict[str, str] = {}
    grouped_ref: defaultdict[int, list[str]] = defaultdict(list)
    grouped_cand: defaultdict[int, list[str]] = defaultdict(list)

    for element, count in reference_counts.items():
        grouped_ref[count].append(element)
    for element, count in candidate_counts.items():
        grouped_cand[count].append(element)

    for count in sorted(grouped_ref):
        ref_group = sorted(grouped_ref[count], key=lambda e: (element_x(e), e))
        cand_group = sorted(grouped_cand[count], key=lambda e: (element_x(e), e))

        if len(ref_group) != len(cand_group):
            raise ValueError("Could not match equal-count species groups.")

        # First use electronegativity ordering. If two species share similar
        # X values, alphabetical ordering keeps the result reproducible.
        for ref_el, cand_el in zip(ref_group, cand_group):
            mapping[ref_el] = cand_el

    if set(mapping) != set(reference_counts):
        raise ValueError("Incomplete species mapping.")

    return mapping


def prototype_score(candidate_formula: str, reference_atoms: Atoms) -> tuple[int, float]:
    """Score a reference structure for a candidate.

    Primary criterion is exact reduced stoichiometric prototype. Secondary
    criterion is closeness of composition-averaged atomic radius, which avoids
    extreme lattice rescaling.
    """
    candidate_proto = prototype_from_counts(reduced_counts(candidate_formula))
    reference_proto = prototype_from_counts(structure_reduced_counts(reference_atoms))
    if candidate_proto != reference_proto:
        return -1, float("inf")

    c_radius = composition_radius(candidate_formula)
    r_formula = reference_atoms.composition.reduced_formula
    r_radius = composition_radius(r_formula)

    if c_radius is None or r_radius is None or r_radius <= 0:
        radius_delta = 999.0
    else:
        radius_delta = abs(math.log(c_radius / r_radius))

    # Exact prototype dominates. Radius similarity is only a tie-breaker.
    return 1, radius_delta


def collect_reference_structures() -> list[dict]:
    print("Loading JARVIS-DFT 3D reference structures...")
    records = jarvis_data(dataset="dft_3d")
    print(f"Reference structures loaded: {len(records):,}")

    references = []
    for record in records:
        try:
            atoms = Atoms.from_dict(record["atoms"])
            counts = structure_reduced_counts(atoms)
            proto = prototype_from_counts(counts)
            references.append(
                {
                    "jid": str(record.get("jid", "")),
                    "atoms": atoms,
                    "formula": atoms.composition.reduced_formula,
                    "prototype": proto,
                    "counts": counts,
                }
            )
        except Exception:
            continue

    print(f"Usable reference structures: {len(references):,}")
    return references


def choose_references(
    candidate_formula: str,
    references: list[dict],
    max_prototypes: int,
) -> list[dict]:
    candidate_proto = prototype_from_counts(reduced_counts(candidate_formula))
    matches = [r for r in references if r["prototype"] == candidate_proto]

    if not matches:
        return []

    scored = []
    for ref in matches:
        score, radius_delta = prototype_score(candidate_formula, ref["atoms"])
        if score < 0:
            continue
        scored.append((radius_delta, ref["jid"], ref))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [item[2] for item in scored[:max_prototypes]]


def transfer_structure(
    candidate_formula: str,
    reference: dict,
) -> tuple[Atoms, dict]:
    ref_atoms = reference["atoms"]
    mapping = build_mapping(candidate_formula, ref_atoms)

    new_elements = [mapping[str(el)] for el in ref_atoms.elements]

    candidate_radius = composition_radius(candidate_formula)
    reference_radius = composition_radius(reference["formula"])

    scale = 1.0
    if candidate_radius is not None and reference_radius is not None and reference_radius > 0:
        scale = candidate_radius / reference_radius

    # Avoid pathological starting cells. DFT relaxation will determine the
    # physically appropriate lattice; this is only an initial geometry.
    scale = float(np.clip(scale, 0.75, 1.35))
    lattice = np.asarray(ref_atoms.lattice_mat, dtype=float) * scale

    # JARVIS stores fractional coordinates separately and accepts them directly.
    frac_coords = np.asarray(ref_atoms.frac_coords, dtype=float)

    new_atoms = Atoms(
        lattice_mat=lattice.tolist(),
        coords=frac_coords.tolist(),
        elements=new_elements,
        cartesian=False,
    )

    metadata = {
        "source_jid": reference["jid"],
        "source_formula": reference["formula"],
        "prototype": reference["prototype"],
        "lattice_scale": scale,
        "mapping": json.dumps(mapping, sort_keys=True),
    }
    return new_atoms, metadata


def write_candidate_structures(
    candidate_formula: str,
    candidate_row: pd.Series,
    references: list[dict],
    max_prototypes: int,
) -> list[dict]:
    chosen = choose_references(candidate_formula, references, max_prototypes)
    if not chosen:
        return []

    safe_name = candidate_formula.replace("/", "_")
    candidate_dir = OUTPUT_ROOT / safe_name
    candidate_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, reference in enumerate(chosen, start=1):
        try:
            atoms, metadata = transfer_structure(candidate_formula, reference)
            prototype_dir = candidate_dir / f"prototype_{index}"
            prototype_dir.mkdir(parents=True, exist_ok=True)

            poscar_path = prototype_dir / "POSCAR"
            cif_path = prototype_dir / "structure.cif"

            Poscar(atoms).write_file(str(poscar_path))
            atoms.write_cif(str(cif_path))

            readme = prototype_dir / "README.txt"
            readme.write_text(
                f"""Hypothetical structure initialization
========================================

Candidate composition: {candidate_formula}
Predicted band gap (ML): {candidate_row['predicted_bandgap_eV']:.6f} eV
RF uncertainty proxy: {candidate_row['uncertainty_proxy_eV']:.6f} eV
Chemistry score: {candidate_row['chemistry_score']:.4f}

Prototype source JARVIS ID: {metadata['source_jid']}
Prototype source formula: {metadata['source_formula']}
Stoichiometric prototype: {metadata['prototype']}
Uniform lattice scale: {metadata['lattice_scale']:.6f}

Species mapping:
{metadata['mapping']}

Interpretation
--------------
This is a prototype-transferred INITIAL STRUCTURE. The source structure
provides a known geometric arrangement with the same reduced stoichiometric
prototype. The lattice was uniformly scaled using composition-averaged
atomic-radius information.

This structure is NOT a predicted ground-state structure. It has not been
relaxed and has no validated formation energy, stability, phonon spectrum,
or DFT band gap.

Next step: relax this structure with DFT or a validated interatomic potential.
""",
                encoding="utf-8",
            )

            rows.append(
                {
                    "formula": candidate_formula,
                    "shortlist_rank": int(candidate_row["shortlist_rank"]),
                    "predicted_bandgap_eV": float(candidate_row["predicted_bandgap_eV"]),
                    "uncertainty_proxy_eV": float(candidate_row["uncertainty_proxy_eV"]),
                    "chemistry_score": float(candidate_row["chemistry_score"]),
                    "prototype_index": index,
                    "prototype_source_jid": metadata["source_jid"],
                    "prototype_source_formula": metadata["source_formula"],
                    "prototype": metadata["prototype"],
                    "lattice_scale": metadata["lattice_scale"],
                    "poscar": str(poscar_path.relative_to(ROOT)),
                    "cif": str(cif_path.relative_to(ROOT)),
                    "status": "prototype_transferred_initial_structure",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "formula": candidate_formula,
                    "shortlist_rank": int(candidate_row["shortlist_rank"]),
                    "predicted_bandgap_eV": float(candidate_row["predicted_bandgap_eV"]),
                    "uncertainty_proxy_eV": float(candidate_row["uncertainty_proxy_eV"]),
                    "chemistry_score": float(candidate_row["chemistry_score"]),
                    "prototype_index": index,
                    "prototype_source_jid": reference["jid"],
                    "prototype_source_formula": reference["formula"],
                    "prototype": reference["prototype"],
                    "lattice_scale": np.nan,
                    "poscar": "",
                    "cif": "",
                    "status": f"failed: {exc}",
                }
            )

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate prototype-transferred initial structures for the hypothetical shortlist."
    )
    parser.add_argument("--input", type=Path, default=SHORTLIST_FILE)
    parser.add_argument("--max-prototypes", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(
            f"Missing {args.input}. Run build_hypothetical_shortlist.py first."
        )
    if args.max_prototypes < 1:
        raise ValueError("--max-prototypes must be >= 1")

    shortlist = pd.read_csv(args.input)
    required = {
        "formula",
        "shortlist_rank",
        "predicted_bandgap_eV",
        "uncertainty_proxy_eV",
        "chemistry_score",
    }
    missing = sorted(required - set(shortlist.columns))
    if missing:
        raise ValueError("Shortlist is missing columns: " + ", ".join(missing))

    if args.limit is not None:
        shortlist = shortlist.head(args.limit).copy()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    references = collect_reference_structures()
    all_rows = []
    no_match = []

    print("=" * 72)
    print("Prototype-transferred hypothetical structure generation")
    print("=" * 72)

    for _, row in shortlist.iterrows():
        formula = str(row["formula"])
        try:
            chosen = choose_references(formula, references, args.max_prototypes)
            if not chosen:
                no_match.append(formula)
                print(f"[NO PROTOTYPE MATCH] {formula}")
                continue

            rows = write_candidate_structures(
                formula, row, references, args.max_prototypes
            )
            all_rows.extend(rows)
            successful = sum(r["status"].startswith("prototype_transferred") for r in rows)
            print(
                f"[OK] {formula}: {successful}/{len(chosen)} structures "
                f"(prototype={chosen[0]['prototype']})"
            )
        except Exception as exc:
            no_match.append(formula)
            print(f"[FAILED] {formula}: {exc}")

    manifest = pd.DataFrame(all_rows)
    manifest.to_csv(MANIFEST_FILE, index=False)

    summary = {
        "shortlist_candidates": int(len(shortlist)),
        "candidate_structures_generated": int(
            sum(str(s).startswith("prototype_transferred") for s in manifest.get("status", []))
        ),
        "candidates_without_prototype_match": no_match,
        "method": "JARVIS prototype transfer with uniform radius-based lattice scaling",
        "scientific_status": "initial geometries only; no relaxation or stability validation",
    }
    (OUTPUT_ROOT / "generation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print()
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Structures root: {OUTPUT_ROOT}")
    print(f"Candidates without prototype match: {len(no_match)}")
    if no_match:
        print("  " + ", ".join(no_match))
    print()
    print(
        "IMPORTANT: generated structures are prototype-transferred initial "
        "geometries, not validated ground-state structures."
    )


if __name__ == "__main__":
    main()
