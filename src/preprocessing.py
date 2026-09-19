"""Preprocessing pipeline for the quantum-materials ML project."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERIC_FEATURES = [
    "num_elements",
    "total_atoms",
    "mean_atomic_number",
    "min_atomic_number",
    "max_atomic_number",
    "mean_atomic_mass",
    "min_atomic_mass",
    "max_atomic_mass",
    "mean_atomic_radius",
    "min_atomic_radius",
    "max_atomic_radius",
    "mean_electronegativity",
    "min_electronegativity",
    "max_electronegativity",
    "electronegativity_difference",
    "mean_ionization_energy",
    "mean_electron_affinity",
    "mean_s_valence",
    "mean_p_valence",
    "mean_d_valence",
    "mean_f_valence",
    "mean_period",
    "mean_group",
]

CATEGORICAL_FEATURES = ["crys", "spg_number"]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def create_preprocessor() -> ColumnTransformer:
    """Create the leakage-safe preprocessing transformer."""
    numeric_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        [
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )
