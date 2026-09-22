"""Generate hypothetical composition candidates for ML-assisted materials discovery.

This tool explores a restricted composition space of known chemical elements,
removes compositions already represented in JARVIS-DFT, and ranks remaining
hypothetical compositions with the existing 25-descriptor Random Forest model.

The output is a hypothesis-generation/prioritization result. It does not prove
that a formula is experimentally novel, stable, synthesizable, or structurally
realizable.

Run from the repository root:
    python src/generate_hypothetical_candidates.py --target 1.5 --top-k 200

Output:
    results/hypothetical_candidates.csv
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from jarvis.db.figshare import data
from jarvis.core.specie import Specie

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.descriptors import material_descriptors, parse_formula
from src.models import create_model_pipeline
from src.preprocessing import FEATURE_COLUMNS, NUMERIC_FEATURES

DATA_FILE = ROOT / "data" / "enhanced_material_descriptors.csv"
OUTPUT_FILE = ROOT / "results" / "hypothetical_candidates.csv"

OXIDATION_STATES = {
    "Li": (1,), "Na": (1,), "K": (1,), "Rb": (1,), "Cs": (1,),
    "Mg": (2,), "Ca": (2,), "Sr": (2,), "Ba": (2,),
    "Al": (3,), "Ga": (3,), "In": (3,),
    "Si": (4,), "Ge": (4,), "Sn": (2, 4),
    "P": (3, 5), "As": (3, 5), "Sb": (3, 5), "Bi": (3, 5),
    "B": (3,), "C": (4,),
    "N": (-3,), "O": (-2,), "S": (-2,), "Se": (-2,), "Te": (-2,),
    "F": (-1,), "Cl": (-1,), "Br": (-1,), "I": (-1,),
}

ELEMENTS = tuple(OXIDATION_STATES)


def gcd_many(values: list[int]) -> int:
    g = 0
    for value in values:
        g = math.gcd(g, abs(int(value)))
    return max(g, 1)


def canonical_key(formula: str) -> str:
    composition = parse_formula(formula)
    if not composition:
        return ""
    values = [round(v * 1000) for v in composition.values()]
    g = gcd_many(values)
    reduced = {e: v // g for e, v in zip(composition, values)}
    return ";".join(f"{e}:{reduced[e]}" for e in sorted(reduced))


def format_formula(elements: tuple[str, ...], amounts: tuple[int, ...]) -> str:
    return "".join(
        element if amount == 1 else f"{element}{amount}"
        for element, amount in zip(elements, amounts)
    )



def safe_material_descriptors(formula: str) -> dict:
    """Compute the same 23 elemental descriptors while tolerating missing JARVIS properties.

    Some hypothetical compositions contain elements for which one or more
    JARVIS elemental properties are unavailable. Missing values are retained
    as NaN and imputed later from the training-set descriptor medians.
    """
    composition = parse_formula(formula)
    if not composition:
        return {}

    elements = list(composition)
    amounts = np.array([composition[e] for e in elements], dtype=float)
    total_atoms = float(amounts.sum())

    def values(attr):
        out = []
        for element in elements:
            try:
                value = float(getattr(Specie(element), attr))
                if value <= -9990:
                    value = np.nan
            except Exception:
                value = np.nan
            out.append(value)
        return out

    def safe_mean(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.mean(vals)) if vals else np.nan

    def safe_min(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.min(vals)) if vals else np.nan

    def safe_max(vals):
        vals = [v for v in vals if np.isfinite(v)]
        return float(np.max(vals)) if vals else np.nan

    atomic_numbers = values("Z")
    atomic_masses = values("atomic_mass")
    atomic_radii = values("atomic_rad")
    electronegativities = values("X")
    ionization_energies = values("first_ion_en")
    electron_affinities = values("elec_aff")
    s_valence = values("nsvalence")
    p_valence = values("npvalence")
    d_valence = values("ndvalence")
    f_valence = values("nfvalence")
    periods = values("row")
    groups = values("coulmn")

    en_min = safe_min(electronegativities)
    en_max = safe_max(electronegativities)

    return {
        "num_elements": len(elements),
        "total_atoms": total_atoms,
        "mean_atomic_number": safe_mean(atomic_numbers),
        "min_atomic_number": safe_min(atomic_numbers),
        "max_atomic_number": safe_max(atomic_numbers),
        "mean_atomic_mass": safe_mean(atomic_masses),
        "min_atomic_mass": safe_min(atomic_masses),
        "max_atomic_mass": safe_max(atomic_masses),
        "mean_atomic_radius": safe_mean(atomic_radii),
        "min_atomic_radius": safe_min(atomic_radii),
        "max_atomic_radius": safe_max(atomic_radii),
        "mean_electronegativity": safe_mean(electronegativities),
        "min_electronegativity": en_min,
        "max_electronegativity": en_max,
        "electronegativity_difference": (
            en_max - en_min
            if np.isfinite(en_max) and np.isfinite(en_min)
            else np.nan
        ),
        "mean_ionization_energy": safe_mean(ionization_energies),
        "mean_electron_affinity": safe_mean(electron_affinities),
        "mean_s_valence": safe_mean(s_valence),
        "mean_p_valence": safe_mean(p_valence),
        "mean_d_valence": safe_mean(d_valence),
        "mean_f_valence": safe_mean(f_valence),
        "mean_period": safe_mean(periods),
        "mean_group": safe_mean(groups),
    }

def neutral_ratios(
    elements: tuple[str, ...],
    states: tuple[int, ...],
    max_atoms: int = 12,
) -> list[tuple[int, ...]]:
    """Return small positive integer stoichiometries with zero net charge."""
    results: list[tuple[int, ...]] = []
    n = len(elements)

    def recurse(i: int, current: list[int], charge: int) -> None:
        if i == n - 1:
            z = states[i]
            if z == 0:
                return
            numerator = -charge
            if numerator % z != 0:
                return
            amount = numerator // z
            if amount < 1:
                return
            candidate = current + [amount]
            if sum(candidate) <= max_atoms and gcd_many(candidate) == 1:
                results.append(tuple(candidate))
            return

        for amount in range(1, max_atoms - sum(current) + 1):
            recurse(
                i + 1,
                current + [amount],
                charge + amount * states[i],
            )

    recurse(0, [], 0)
    return sorted(set(results))


def generate_binary_ternary_candidates() -> list[str]:
    candidates: set[str] = set()
    cations = [e for e in ELEMENTS if any(z > 0 for z in OXIDATION_STATES[e])]
    anions = [e for e in ELEMENTS if any(z < 0 for z in OXIDATION_STATES[e])]

    for cation in cations:
        for anion in anions:
            for zc in OXIDATION_STATES[cation]:
                if zc <= 0:
                    continue
                for za in OXIDATION_STATES[anion]:
                    if za >= 0:
                        continue
                    for amounts in neutral_ratios(
                        (cation, anion), (zc, za), max_atoms=12
                    ):
                        candidates.add(format_formula((cation, anion), amounts))

    for i, cation_a in enumerate(cations):
        for cation_b in cations[i + 1:]:
            for anion in anions:
                for za in OXIDATION_STATES[cation_a]:
                    if za <= 0:
                        continue
                    for zb in OXIDATION_STATES[cation_b]:
                        if zb <= 0:
                            continue
                        for zc in OXIDATION_STATES[anion]:
                            if zc >= 0:
                                continue
                            for amounts in neutral_ratios(
                                (cation_a, cation_b, anion),
                                (za, zb, zc),
                                max_atoms=10,
                            ):
                                candidates.add(
                                    format_formula(
                                        (cation_a, cation_b, anion), amounts
                                    )
                                )

    for i, anion_a in enumerate(anions):
        for anion_b in anions[i + 1:]:
            for cation in cations:
                for za in OXIDATION_STATES[cation]:
                    if za <= 0:
                        continue
                    for zb in OXIDATION_STATES[anion_a]:
                        if zb >= 0:
                            continue
                        for zc in OXIDATION_STATES[anion_b]:
                            if zc >= 0:
                                continue
                            for amounts in neutral_ratios(
                                (cation, anion_a, anion_b),
                                (za, zb, zc),
                                max_atoms=10,
                            ):
                                candidates.add(
                                    format_formula(
                                        (cation, anion_a, anion_b), amounts
                                    )
                                )

    return sorted(candidates)


def generate_quaternary_candidates() -> list[str]:
    """Generate small charge-balanced 2-cation/2-anion compositions."""
    candidates: set[str] = set()
    cations = [e for e in ELEMENTS if any(z > 0 for z in OXIDATION_STATES[e])]
    anions = [e for e in ELEMENTS if any(z < 0 for z in OXIDATION_STATES[e])]

    for i, ca in enumerate(cations):
        for cb in cations[i + 1:]:
            for j, aa in enumerate(anions):
                for ab in anions[j + 1:]:
                    elements = (ca, cb, aa, ab)
                    for za in OXIDATION_STATES[ca]:
                        if za <= 0:
                            continue
                        for zb in OXIDATION_STATES[cb]:
                            if zb <= 0:
                                continue
                            for zc in OXIDATION_STATES[aa]:
                                if zc >= 0:
                                    continue
                                for zd in OXIDATION_STATES[ab]:
                                    if zd >= 0:
                                        continue

                                    for a in range(1, 5):
                                        for b in range(1, 5):
                                            for c in range(1, 5):
                                                numerator = -(a * za + b * zb + c * zc)
                                                if numerator % zd != 0:
                                                    continue
                                                d = numerator // zd
                                                if not 1 <= d <= 4:
                                                    continue

                                                amounts = (a, b, c, d)
                                                if sum(amounts) > 12:
                                                    continue
                                                if gcd_many(list(amounts)) != 1:
                                                    continue

                                                candidates.add(
                                                    format_formula(elements, amounts)
                                                )

    return sorted(candidates)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=float, default=1.5)
    parser.add_argument("--max-distance", type=float, default=0.5)
    parser.add_argument("--max-uncertainty", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    print("=" * 72)
    print("Hypothetical composition-space exploration")
    print("=" * 72)

    print("Loading JARVIS-DFT metadata...")
    jarvis = pd.DataFrame(data("dft_3d"))

    known_keys = {
        canonical_key(formula)
        for formula in jarvis["formula"].dropna().astype(str)
    }

    print("Generating charge-balanced binary and ternary compositions...")
    formulas = generate_binary_ternary_candidates()
    print(f"Generated raw binary/ternary candidates: {len(formulas):,}")

    print("Generating charge-balanced quaternary compositions...")
    quaternary = generate_quaternary_candidates()
    formulas = sorted(set(formulas).union(quaternary))
    print(f"Generated raw candidates after quaternaries: {len(formulas):,}")

    records = []
    known_composition_count = 0
    descriptor_failure_count = 0

    for formula in formulas:
        key = canonical_key(formula)
        if not key:
            continue

        if key in known_keys:
            known_composition_count += 1
            continue

        try:
            desc = material_descriptors(formula)
        except Exception:
            # Fall back to a tolerant implementation when a JARVIS property
            # is unavailable for one of the hypothetical elements.
            desc = safe_material_descriptors(formula)

        if not desc:
            descriptor_failure_count += 1
            continue

        # Hypothetical compositions do not yet have a known crystal structure.
        desc["crys"] = "unknown"
        desc["spg_number"] = -1

        records.append({"formula": formula, "composition_key": key, **desc})

    candidates = pd.DataFrame(records)

    print(f"Known composition candidates removed: {known_composition_count:,}")
    print(f"Descriptor generation failures: {descriptor_failure_count:,}")
    print(f"Novel composition candidates after filtering: {len(candidates):,}")

    if candidates.empty:
        raise RuntimeError(
            "No hypothetical candidates survived generation. Check descriptor "
            "generation and the composition-space diagnostics above."
        )

    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Generate enhanced_material_descriptors.csv first."
        )

    training = pd.read_csv(DATA_FILE)
    missing = [
        column
        for column in FEATURE_COLUMNS + ["target_bandgap"]
        if column not in training.columns
    ]
    if missing:
        raise ValueError("Training data missing columns: " + ", ".join(missing))

    # Hypothetical compositions may contain valid elements with missing JARVIS
    # elemental properties. Use training-set medians for those descriptor
    # values rather than discarding the entire composition. This keeps the
    # feature definition identical to the trained model while making the
    # candidate-generation stage robust to sparse elemental metadata.
    imputed_columns = []
    for column in NUMERIC_FEATURES:
        training_median = pd.to_numeric(training[column], errors="coerce").median()
        if column in candidates.columns:
            values = pd.to_numeric(candidates[column], errors="coerce")
            missing_count = int(values.isna().sum())
            if missing_count:
                imputed_columns.append((column, missing_count))
            candidates[column] = values.fillna(training_median)

    if imputed_columns:
        print("Imputed missing hypothetical descriptors from training medians:")
        for column, count in imputed_columns:
            print(f"  {column}: {count:,}")

    X_train = training[FEATURE_COLUMNS].copy()
    y_train = pd.to_numeric(training["target_bandgap"], errors="coerce")
    valid = y_train.notna()
    X_train = X_train.loc[valid].reset_index(drop=True)
    y_train = y_train.loc[valid].reset_index(drop=True)

    print(f"Training model on {len(X_train):,} known materials...")
    model = create_model_pipeline()
    model.fit(X_train, y_train)

    X_candidates = candidates[FEATURE_COLUMNS].copy()
    predictions = model.predict(X_candidates)

    transformed = model.named_steps["preprocessor"].transform(X_candidates)
    forest = model.named_steps["model"]

    tree_predictions = np.vstack(
        [tree.predict(transformed) for tree in forest.estimators_]
    )
    mean = tree_predictions.mean(axis=0)
    variance = np.maximum(tree_predictions.var(axis=0), 0.0)

    candidates["predicted_bandgap_eV"] = predictions
    candidates["uncertainty_proxy_eV"] = np.sqrt(variance)
    candidates["distance_from_target_eV"] = (
        candidates["predicted_bandgap_eV"] - args.target
    ).abs()

    candidates["screening_score"] = (
        candidates["distance_from_target_eV"]
        + 0.25 * candidates["uncertainty_proxy_eV"]
    )

    candidates = candidates[
        (candidates["distance_from_target_eV"] <= args.max_distance)
        & (candidates["uncertainty_proxy_eV"] <= args.max_uncertainty)
    ].sort_values(
        ["screening_score", "distance_from_target_eV", "uncertainty_proxy_eV"]
    )

    candidates["rank"] = np.arange(1, len(candidates) + 1)

    output_columns = [
        "rank",
        "formula",
        "predicted_bandgap_eV",
        "uncertainty_proxy_eV",
        "distance_from_target_eV",
        "screening_score",
        "num_elements",
        "total_atoms",
    ]

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    candidates[output_columns].head(args.top_k).to_csv(
        OUTPUT_FILE, index=False
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Returned candidates: {min(args.top_k, len(candidates)):,}")
    print()
    print(
        "IMPORTANT: these are hypothetical composition candidates. "
        "The model does not establish crystal structure, thermodynamic "
        "stability, synthesizability, or experimental novelty."
    )

    if not candidates.empty:
        print()
        print(candidates[output_columns].head(20).to_string(index=False))


if __name__ == "__main__":
    main()
