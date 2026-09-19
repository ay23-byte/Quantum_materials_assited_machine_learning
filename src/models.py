"""Model definitions."""

from __future__ import annotations

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline

from .preprocessing import create_preprocessor


def create_tuned_random_forest() -> RandomForestRegressor:
    """Return the final tuned Random Forest configuration."""
    return RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=1.0,
        random_state=42,
        n_jobs=-1,
    )


def create_model_pipeline() -> Pipeline:
    """Return preprocessing + final Random Forest as one pipeline."""
    return Pipeline(
        [
            ("preprocessor", create_preprocessor()),
            ("model", create_tuned_random_forest()),
        ]
    )
