"""Generate predictions for every JARVIS-DFT 3D material.

This script is intentionally separate from model evaluation. Model selection and
the untouched test-set metrics are completed in the notebooks first. For the
screening application, the final tuned model is then refit on all labelled
materials and used to generate predictions for all 93,902 rows.

Run from the repository root:
    python src/generate_screening_predictions.py

Requires the research environment because jarvis-tools is needed.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from jarvis.db.figshare import data

# Allow imports such as "from src.models import ..." when this file is run
# directly from the repository root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models import create_model_pipeline
from src.preprocessing import FEATURE_COLUMNS


DATA_FILE = ROOT / "data" / "enhanced_material_descriptors.csv"
OUTPUT_FILE = ROOT / "results" / "all_material_predictions.csv"


def load_material_data() -> pd.DataFrame:
    """Load descriptors and attach JARVIS IDs/formulas if needed."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing {DATA_FILE}. Run the descriptor-generation notebook first."
        )

    df = pd.read_csv(DATA_FILE)

    missing = [c for c in FEATURE_COLUMNS + ["target_bandgap"] if c not in df.columns]
    if missing:
        raise ValueError(
            "The descriptor file is missing required columns: " + ", ".join(missing)
        )

    # The enhanced descriptor CSV used by the notebooks does not necessarily
    # contain JID/formula, so recover them from the same JARVIS-DFT 3D dataset.
    if "jid" not in df.columns or "formula" not in df.columns:
        print("Loading JARVIS-DFT 3D metadata...")
        jarvis = pd.DataFrame(data("dft_3d"))

        if len(jarvis) != len(df):
            raise ValueError(
                f"Row-count mismatch: descriptor file has {len(df)} rows, "
                f"JARVIS-DFT has {len(jarvis)} rows."
            )

        df.insert(0, "jid", jarvis["jid"].to_numpy())
        df.insert(1, "formula", jarvis["formula"].to_numpy())

    return df.reset_index(drop=True)


def main() -> None:
    print("=" * 72)
    print("Quantum Materials ML — all-material screening prediction")
    print("=" * 72)

    df = load_material_data()
    X = df[FEATURE_COLUMNS].copy()
    y = pd.to_numeric(df["target_bandgap"], errors="coerce")

    valid = y.notna()
    if not valid.all():
        print(f"Dropping {(~valid).sum()} rows with missing target values.")
        df = df.loc[valid].reset_index(drop=True)
        X = X.loc[valid].reset_index(drop=True)
        y = y.loc[valid].reset_index(drop=True)

    print(f"Materials: {len(df):,}")
    print(f"Features: {len(FEATURE_COLUMNS)}")
    print()
    print("Training final screening model on ALL labelled materials...")
    print(
        "Note: this refit is for screening only. It does not change the "
        "previous untouched test-set evaluation."
    )

    pipeline = create_model_pipeline()
    pipeline.fit(X, y)

    print("Generating predictions...")
    predictions = pipeline.predict(X)

    # Calculate the standard deviation across individual RF tree predictions
    # without creating a large (n_trees x n_materials) matrix in memory.
    print("Calculating Random Forest uncertainty proxy...")
    transformed = pipeline.named_steps["preprocessor"].transform(X)
    forest = pipeline.named_steps["model"]

    n = len(X)
    sum_predictions = np.zeros(n, dtype=np.float64)
    sum_squared_predictions = np.zeros(n, dtype=np.float64)

    for i, tree in enumerate(forest.estimators_, start=1):
        tree_prediction = tree.predict(transformed)
        sum_predictions += tree_prediction
        sum_squared_predictions += tree_prediction * tree_prediction
        if i % 25 == 0 or i == len(forest.estimators_):
            print(f"  processed {i}/{len(forest.estimators_)} trees")

    mean_prediction = sum_predictions / len(forest.estimators_)
    variance = (
        sum_squared_predictions / len(forest.estimators_)
        - mean_prediction * mean_prediction
    )
    uncertainty = np.sqrt(np.maximum(variance, 0.0))

    output = pd.DataFrame(
        {
            "jid": df["jid"].astype(str),
            "formula": df["formula"].astype(str),
            "predicted_bandgap": predictions,
            "prediction_uncertainty": uncertainty,
            # Retained for retrospective research analysis. The Streamlit app
            # does not need to display this value in the screening view.
            "target_bandgap": y.to_numpy(),
            "crys": df["crys"].to_numpy(),
            "spg_number": df["spg_number"].to_numpy(),
        }
    )

    output["distance_from_1p5_eV"] = (output["predicted_bandgap"] - 1.5).abs()
    output = output.sort_values("distance_from_1p5_eV").reset_index(drop=True)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows: {len(output):,}")
    print(
        f"Predicted band-gap range: "
        f"{output['predicted_bandgap'].min():.4f}–"
        f"{output['predicted_bandgap'].max():.4f} eV"
    )
    print(
        f"Uncertainty range: "
        f"{output['prediction_uncertainty'].min():.4f}–"
        f"{output['prediction_uncertainty'].max():.4f} eV"
    )
    print()
    print("Done.")


if __name__ == "__main__":
    main()
