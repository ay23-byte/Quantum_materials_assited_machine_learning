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
from itertools import permutations
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import spglib
from jarvis.core.atoms import Atoms
from jarvis.core.specie import Specie
from jarvis.db.figshare import data as jarvis_data
from jarvis.io.vasp.inputs import Poscar

try:
    from src.descriptors import parse_formula
except ModuleNotFoundError:
    # Allows direct execution with: python src\\generate_hypothetical_structures.py
    from descriptors import parse_formula

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


def chemical_role(element: str) -> str:
    """Assign a coarse chemical role for structure-transfer matching.

    This is a heuristic used only to avoid obviously incompatible species
    substitutions. It is not an oxidation-state or bonding prediction.
    """
    fam = family(element)
    if fam in {"alkali", "alkaline_earth", "post_transition", "transition"}:
        return "cation"
    if fam in {"halogen", "chalcogen", "pnictogen"}:
        return "anion"
    if fam in {"metalloid", "nonmetal"}:
        return "ambiguous"
    return "ambiguous"


def role_compatibility(candidate: str, reference: str) -> float:
    """Score compatibility between candidate and reference chemical roles."""
    c_role = chemical_role(candidate)
    r_role = chemical_role(reference)

    if c_role == r_role:
        return 3.0
    if "ambiguous" in {c_role, r_role}:
        return 1.0
    return -4.0


def species_pair_score(candidate: str, reference: str) -> float:
    """Score a proposed species substitution.

    Role compatibility dominates, followed by broad family similarity,
    electronegativity similarity, and atomic-radius similarity.
    """
    score = role_compatibility(candidate, reference)

    if family(candidate) == family(reference):
        score += 2.0

    try:
        dx = abs(element_x(candidate) - element_x(reference))
        score += max(0.0, 1.5 - dx)
    except Exception:
        pass

    c_radius = element_radius(candidate)
    r_radius = element_radius(reference)
    if c_radius is not None and r_radius is not None and c_radius > 0 and r_radius > 0:
        radius_ratio = max(c_radius, r_radius) / min(c_radius, r_radius)
        score += max(0.0, 1.0 - abs(math.log(radius_ratio)))

    return score


def build_mapping(candidate_formula: str, reference_atoms: Atoms) -> dict[str, str]:
    """Map reference species to candidate species using chemistry-aware matching.

    Stoichiometric multiplicity is enforced exactly. Among species with equal
    multiplicity, all possible one-to-one assignments are evaluated using a
    coarse chemical-role/family/electronegativity/radius score. The best
    assignment is selected deterministically.

    This remains a prototype-transfer heuristic; it does not determine
    oxidation states, bonding, or the true crystal structure.
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
        ref_group = sorted(grouped_ref[count])
        cand_group = sorted(grouped_cand[count])

        if len(ref_group) != len(cand_group):
            raise ValueError("Could not match equal-count species groups.")

        best_assignment = None
        best_score = -float("inf")

        for perm in permutations(cand_group):
            score = sum(
                species_pair_score(candidate, reference)
                for reference, candidate in zip(ref_group, perm)
            )
            tie_key = tuple(perm)
            if score > best_score or (
                math.isclose(score, best_score)
                and (best_assignment is None or tie_key < tuple(best_assignment))
            ):
                best_score = score
                best_assignment = perm

        if best_assignment is None:
            raise ValueError("Could not determine a chemistry-aware species mapping.")

        for reference, candidate in zip(ref_group, best_assignment):
            mapping[reference] = candidate

    if set(mapping) != set(reference_counts):
        raise ValueError("Incomplete species mapping.")

    pair_scores = [
        species_pair_score(candidate, reference)
        for reference, candidate in mapping.items()
    ]
    incompatible_pairs = sum(score < 0.0 for score in pair_scores)
    if incompatible_pairs > 0:
        raise ValueError(
            "Chemistry-aware mapping rejected: incompatible cation/anion "
            f"substitution detected ({incompatible_pairs} pair(s))."
        )

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

    print(f"Command-line limit: {args.limit}")

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
        limit = max(1, int(args.limit))
        shortlist = shortlist.iloc[:limit].copy()
        print(f"Applied shortlist limit: {limit}")

    # Defensive check: never process more rows than requested.
    if args.limit is not None and len(shortlist) > int(args.limit):
        shortlist = shortlist.iloc[:int(args.limit)].copy()

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Processing {len(shortlist)} hypothetical candidate(s)...")
    references = collect_reference_structures()

    all_rows = []
    successful_candidates = 0
    failed_candidates = 0

    for _, candidate_row in shortlist.iterrows():
        formula = str(candidate_row["formula"])
        print(f"\nProcessing {formula}...")
        rows = write_candidate_structures(
            candidate_formula=formula,
            candidate_row=candidate_row,
            references=references,
            max_prototypes=args.max_prototypes,
        )

        if not rows:
            failed_candidates += 1
            print(f"[FAILED] {formula}: no compatible prototype structures found")
            continue

        all_rows.extend(rows)
        successful = sum(
            row["status"] == "prototype_transferred_initial_structure"
            for row in rows
        )
        failed = len(rows) - successful

        if successful > 0:
            successful_candidates += 1
            print(
                f"[OK] {formula}: {successful}/{len(rows)} structures generated"
            )
        else:
            failed_candidates += 1
            print(f"[FAILED] {formula}: all prototype transfers failed")

    manifest = pd.DataFrame(all_rows)
    if manifest.empty:
        manifest = pd.DataFrame(
            columns=[
                "formula", "shortlist_rank", "predicted_bandgap_eV",
                "uncertainty_proxy_eV", "chemistry_score", "prototype_index",
                "prototype_source_jid", "prototype_source_formula", "prototype",
                "lattice_scale", "poscar", "cif", "status",
            ]
        )

    manifest.to_csv(MANIFEST_FILE, index=False)

    summary = {
        "input_file": str(args.input.relative_to(ROOT)),
        "requested_limit": args.limit,
        "processed_candidates": int(len(shortlist)),
        "max_prototypes": int(args.max_prototypes),
        "reference_structures": int(len(references)),
        "successful_candidates": int(successful_candidates),
        "failed_candidates": int(failed_candidates),
        "successful_structures": int(
            (manifest["status"] == "prototype_transferred_initial_structure").sum()
        ),
        "failed_structures": int(
            manifest["status"].astype(str).str.startswith("failed:").sum()
        ),
        "manifest": str(MANIFEST_FILE.relative_to(ROOT)),
        "interpretation": (
            "Prototype-transferred initial structures only; not evidence of "
            "ground-state stability or synthesizability."
        ),
    }

    summary_file = OUTPUT_ROOT / "generation_summary.json"
    summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nStructure generation complete.")
    print(f"Successful candidates: {successful_candidates}")
    print(f"Failed candidates: {failed_candidates}")
    print(f"Generated structures: {summary['successful_structures']}")
    print(f"Failed structures: {summary['failed_structures']}")
    print(f"Manifest: {MANIFEST_FILE}")
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
