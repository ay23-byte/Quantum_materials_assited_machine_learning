"""Build a diverse research shortlist from hypothetical composition candidates.

Input:
    results/hypothetical_candidates.csv

Outputs:
    results/hypothetical_shortlist.csv
    results/hypothetical_shortlist_report.md
    figures/hypothetical_shortlist_ranking.png
    figures/hypothetical_shortlist_uncertainty.png
    figures/hypothetical_shortlist_chemistry.png

The selection is a prioritization heuristic. It does not establish stability,
synthesizability, crystal structure, or experimental novelty.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from math import gcd
from functools import reduce

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.descriptors import parse_formula

INPUT_FILE = ROOT / "results" / "hypothetical_candidates.csv"
OUTPUT_FILE = ROOT / "results" / "hypothetical_shortlist.csv"
REPORT_FILE = ROOT / "results" / "hypothetical_shortlist_report.md"
FIGURE_DIR = ROOT / "figures"

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

CATION_ELEMENTS = {
    "Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba",
    "Al", "Ga", "In", "Si", "Ge", "Sn", "P", "As", "Sb", "Bi",
    "B", "C",
}


def stoichiometry_class(formula: str) -> str:
    composition = parse_formula(formula)
    n = len(composition)
    return {2: "binary", 3: "ternary", 4: "quaternary"}.get(n, f"{n}-element")


def element_families(formula: str) -> tuple[str, str]:
    composition = parse_formula(formula)
    cation_families = []
    anion_families = []
    for element in composition:
        family = ELEMENT_FAMILY.get(element, "other")
        if element in CATION_ELEMENTS:
            cation_families.append(family)
        else:
            anion_families.append(family)
    cation = "+".join(sorted(set(cation_families))) or "unknown"
    anion = "+".join(sorted(set(anion_families))) or "unknown"
    return cation, anion


def reduced_composition_signature(formula: str) -> str:
    composition = parse_formula(formula)
    ints = [int(round(float(v))) for v in composition.values()]
    divisor = reduce(gcd, ints) if ints else 1
    reduced = {element: int(count) // divisor for element, count in composition.items()}
    return "".join(f"{element}{reduced[element]}" for element in sorted(reduced))


def enrich(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for formula in df["formula"].astype(str):
        cation_family, anion_family = element_families(formula)
        rows.append({
            "stoichiometry_class": stoichiometry_class(formula),
            "cation_family": cation_family,
            "anion_family": anion_family,
            "chemistry_signature": f"{cation_family}|{anion_family}",
            "reduced_composition_signature": reduced_composition_signature(formula),
        })
    return pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def select_diverse(
    df: pd.DataFrame,
    shortlist_size: int,
    max_per_signature: int,
    max_per_reduced_family: int,
    max_per_stoich: int,
) -> pd.DataFrame:
    selected = []
    signature_counts: dict[str, int] = {}
    reduced_counts: dict[str, int] = {}
    stoich_counts: dict[str, int] = {}

    ordered = df.sort_values(
        ["screening_score", "distance_from_target_eV", "uncertainty_proxy_eV"]
    )

    for _, row in ordered.iterrows():
        signature = row["chemistry_signature"]
        reduced_family = row["reduced_composition_signature"]
        stoich = row["stoichiometry_class"]

        if signature_counts.get(signature, 0) >= max_per_signature:
            continue
        if reduced_counts.get(reduced_family, 0) >= max_per_reduced_family:
            continue
        if stoich_counts.get(stoich, 0) >= max_per_stoich:
            continue

        selected.append(row)
        signature_counts[signature] = signature_counts.get(signature, 0) + 1
        reduced_counts[reduced_family] = reduced_counts.get(reduced_family, 0) + 1
        stoich_counts[stoich] = stoich_counts.get(stoich, 0) + 1

        if len(selected) >= shortlist_size:
            break

    if len(selected) < shortlist_size:
        raise RuntimeError(
            f"Strict diversity constraints produced only {len(selected)} candidates, "
            f"but {shortlist_size} were requested. Relax the constraints explicitly "
            f"with --max-per-signature, --max-per-reduced-family, or --max-per-stoich "
            f"instead of silently filling the shortlist with less-diverse candidates."
        )

    result = pd.DataFrame(selected).reset_index(drop=True)
    result["shortlist_rank"] = range(1, len(result) + 1)
    return result


def write_report(shortlist: pd.DataFrame, source_rows: int, output_path: Path) -> None:
    lines = [
        "# Hypothetical Materials Shortlist",
        "",
        "## Purpose",
        "",
        "This report prioritizes ML-generated hypothetical compositions for the next "
        "research stage. It is a screening result, not a claim of material discovery.",
        "",
        "## Source and selection",
        "",
        f"- Source candidates read: **{source_rows:,}**",
        f"- Final shortlist: **{len(shortlist):,}**",
        "- Ranking inputs: predicted band gap, distance from target, Random Forest tree-dispersion uncertainty proxy, chemistry score, and applicability-domain filtering already performed by the generator.",
        "- Diversity controls: chemistry-family signature, reduced composition family, and stoichiometry class.",
        "- Duplicate formulas are not repeated in the final shortlist.",
        "",
        "## Scientific interpretation",
        "",
        "The compositions below are hypothetical and were generated from a restricted "
        "set of known chemical elements and simple charge-balance rules. The ML model "
        "predicts a target band gap from composition-derived descriptors. It does not "
        "establish crystal structure, thermodynamic stability, synthesizability, "
        "phonon stability, experimental novelty, or an experimentally measurable band gap.",
        "",
        "The Random Forest uncertainty column is a tree-dispersion proxy and is not a "
        "calibrated predictive interval. Candidates therefore require structural "
        "generation and first-principles validation before stronger claims are made.",
        "",
        "## Recommended next stage",
        "",
        "1. Generate plausible crystal structures for the shortlisted compositions.",
        "2. Perform structure relaxation and formation-energy calculations with DFT.",
        "3. Remove energetically unfavorable or structurally unstable candidates.",
        "4. Recalculate the electronic structure and band gap with DFT.",
        "5. Apply phonon/stability checks where computationally feasible.",
        "6. Retain only candidates supported by the combined composition, structure, and DFT evidence.",
        "",
        "## Shortlist",
        "",
    ]
    cols = [
        "shortlist_rank", "formula", "predicted_bandgap_eV",
        "uncertainty_proxy_eV", "distance_from_target_eV",
        "screening_score", "chemistry_score", "in_domain_fraction",
        "stoichiometry_class", "cation_family", "anion_family",
    ]
    table = shortlist[cols].copy()
    for col in [
        "predicted_bandgap_eV", "uncertainty_proxy_eV",
        "distance_from_target_eV", "screening_score", "chemistry_score"
    ]:
        table[col] = table[col].map(lambda x: f"{x:.4f}")
    table["in_domain_fraction"] = table["in_domain_fraction"].map(lambda x: f"{x:.2f}")
    lines.append(table.to_markdown(index=False))
    lines.append("")
    lines.append("Generated automatically by src/build_hypothetical_shortlist.py.")
    output_path.write_text("\n".join(lines), encoding="utf-8")


def make_plots(shortlist: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    rank = shortlist.sort_values("shortlist_rank")
    plt.figure(figsize=(10, 6))
    plt.bar(rank["formula"], rank["predicted_bandgap_eV"])
    plt.axhline(1.5, linestyle="--", linewidth=1.5, label="Target = 1.5 eV")
    plt.xticks(rotation=75, ha="right")
    plt.ylabel("Predicted band gap (eV)")
    plt.xlabel("Hypothetical composition")
    plt.title("Hypothetical shortlist: predicted band gaps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hypothetical_shortlist_ranking.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(shortlist["distance_from_target_eV"], shortlist["uncertainty_proxy_eV"], s=55)
    plt.xlabel("Distance from 1.5 eV target (eV)")
    plt.ylabel("Random Forest uncertainty proxy (eV)")
    plt.title("Hypothetical shortlist: target distance vs uncertainty")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hypothetical_shortlist_uncertainty.png", dpi=200)
    plt.close()

    plt.figure(figsize=(9, 6))
    plt.scatter(shortlist["chemistry_score"], shortlist["predicted_bandgap_eV"], s=55)
    plt.axhline(1.5, linestyle="--", linewidth=1.5, label="Target = 1.5 eV")
    plt.xlabel("Chemistry score (screening signal)")
    plt.ylabel("Predicted band gap (eV)")
    plt.title("Hypothetical shortlist: chemistry score vs predicted gap")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "hypothetical_shortlist_chemistry.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=INPUT_FILE)
    parser.add_argument("--shortlist-size", type=int, default=15)
    parser.add_argument("--max-per-signature", type=int, default=2)
    parser.add_argument("--max-per-reduced-family", type=int, default=1)
    parser.add_argument("--max-per-stoich", type=int, default=8)
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Missing {args.input}. Run generate_hypothetical_candidates.py first.")

    df = pd.read_csv(args.input)
    required = {
        "formula", "predicted_bandgap_eV", "uncertainty_proxy_eV",
        "distance_from_target_eV", "screening_score", "chemistry_score",
        "in_domain_fraction",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError("Input is missing columns: " + ", ".join(missing))

    df = df.drop_duplicates(subset=["formula"]).copy()
    df = df[df["in_domain_fraction"] >= 1.0].copy()
    df = enrich(df)

    shortlist = select_diverse(
        df,
        args.shortlist_size,
        args.max_per_signature,
        args.max_per_reduced_family,
        args.max_per_stoich,
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    shortlist.to_csv(OUTPUT_FILE, index=False)
    make_plots(shortlist)
    write_report(shortlist, source_rows=len(df), output_path=REPORT_FILE)

    print("=" * 72)
    print("Hypothetical candidate shortlist")
    print("=" * 72)
    print(f"Source candidates after duplicate/domain filtering: {len(df):,}")
    print(f"Final shortlist: {len(shortlist):,}")
    print("Diversity constraints:")
    print(f"  max per chemistry signature: {args.max_per_signature}")
    print(f"  max per reduced composition family: {args.max_per_reduced_family}")
    print(f"  max per stoichiometry class: {args.max_per_stoich}")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Saved: {REPORT_FILE}")
    print("Saved figures:")
    print(f"  {FIGURE_DIR / 'hypothetical_shortlist_ranking.png'}")
    print(f"  {FIGURE_DIR / 'hypothetical_shortlist_uncertainty.png'}")
    print(f"  {FIGURE_DIR / 'hypothetical_shortlist_chemistry.png'}")
    print()
    print(shortlist[
        ["shortlist_rank", "formula", "predicted_bandgap_eV",
         "uncertainty_proxy_eV", "chemistry_score",
         "stoichiometry_class", "chemistry_signature"]
    ].to_string(index=False))


if __name__ == "__main__":
    main()
