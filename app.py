import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Quantum Materials ML", page_icon="🔬", layout="wide")

@st.cache_data
def load_candidates():
    path = RESULTS / "uncertainty_aware_bandgap_candidates.csv"
    if not path.exists():
        path = RESULTS / "bandgap_screening_candidates.csv"
    return pd.read_csv(path)

@st.cache_data
def load_metrics():
    path = RESULTS / "final_model_metrics.json"
    return json.loads(path.read_text()) if path.exists() else {}

@st.cache_data
def load_group_metrics():
    path = RESULTS / "tuned_groupkfold_summary.json"
    return json.loads(path.read_text()) if path.exists() else {}

st.title("Machine-Learning-Assisted Discovery of Quantum Materials")
st.caption("JARVIS-DFT 3D • OptB88vdW band-gap prediction • Random Forest")

metrics = load_metrics()
group_metrics = load_group_metrics()
candidates = load_candidates()

tab1, tab2, tab3 = st.tabs(["Model performance", "Candidate screening", "About the project"])

with tab1:
    st.subheader("Final model performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{metrics.get('test_MAE_eV', 0):.3f} eV")
    c2.metric("Test RMSE", f"{metrics.get('test_RMSE_eV', 0):.3f} eV")
    c3.metric("Test R²", f"{metrics.get('test_R2', 0):.3f}")

    st.markdown("### Materials-aware validation")
    g1, g2, g3 = st.columns(3)
    g1.metric("GroupKFold MAE", f"{group_metrics.get('MAE_mean_eV', 0):.3f} ± {group_metrics.get('MAE_std_eV', 0):.3f} eV")
    g2.metric("GroupKFold RMSE", f"{group_metrics.get('RMSE_mean_eV', 0):.3f} ± {group_metrics.get('RMSE_std_eV', 0):.3f} eV")
    g3.metric("GroupKFold R²", f"{group_metrics.get('R2_mean', 0):.3f} ± {group_metrics.get('R2_std', 0):.3f}")

    st.info(
        "The random test split measures interpolation on a held-out subset. "
        "Chemical-system GroupKFold is stricter: related chemical systems "
        "are kept out of the training fold."
    )

    st.markdown("### What the model predicts")
    st.write(
        "The model predicts the JARVIS-DFT OptB88vdW band gap in eV using "
        "composition-derived atomic descriptors and crystal-structure information. "
        "HSE/MBJ band gaps are not used as input features."
    )

with tab2:
    st.subheader("Target-band-gap candidate screening")
    st.write(
        "These are known JARVIS materials retrospectively prioritized around a "
        "predicted band gap of 1.5 eV. This is a screening demonstration, not "
        "a claim of experimentally discovered materials."
    )

    c1, c2, c3 = st.columns(3)
    target = c1.number_input("Target band gap (eV)", 0.1, 5.0, 1.5, 0.1)
    max_distance = c2.slider("Maximum predicted distance (eV)", 0.01, 1.0, 0.10, 0.01)
    max_uncertainty = c3.slider("Maximum uncertainty proxy (eV)", 0.01, 1.0, 0.40, 0.01)

    df = candidates.copy()
    df["distance_from_selected_target"] = (df["predicted_bandgap"] - target).abs()
    filtered = df[
        (df["distance_from_selected_target"] <= max_distance)
        & (df["prediction_uncertainty"] <= max_uncertainty)
    ].sort_values(["distance_from_selected_target", "prediction_uncertainty"])

    st.write(f"Showing **{len(filtered)}** candidates from the stored screening results.")

    display_cols = [
        "jid", "formula", "predicted_bandgap", "prediction_uncertainty",
        "target_bandgap", "prediction_error", "crys", "spg_number"
    ]
    available = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[available].head(100), use_container_width=True, hide_index=True)

    if not filtered.empty:
        st.markdown("### Predicted band-gap distribution")
        chart_df = filtered[["formula", "predicted_bandgap"]].head(30).set_index("formula")
        st.bar_chart(chart_df)

    st.caption(
        "Uncertainty is the standard deviation of Random Forest tree predictions "
        "and should be treated as a model-dispersion proxy, not a calibrated "
        "prediction interval."
    )

with tab3:
    st.subheader("Project overview")
    st.write(
        "Research question: Can machine learning learn relationships between "
        "chemically and structurally informed descriptors and DFT-calculated "
        "band gaps, then use that model to prioritize materials for a target range?"
    )
    st.write("Dataset: JARVIS-DFT 3D, 93,902 materials.")
    st.write("Target: optb88vdw_bandgap (eV).")
    st.write("Model: Tuned Random Forest Regressor with 200 trees.")
    st.write(
        "Descriptors include composition statistics, electronegativity, ionization "
        "energy, electron affinity, valence information, periodic-table descriptors, "
        "crystal system, and space-group number."
    )
    st.warning(
        "Feature importance indicates predictive usefulness, not physical causality. "
        "The screening results are retrospective because the materials already exist "
        "in the database."
    )
    st.markdown("### Reproducibility")
    st.write(
        "The notebooks contain the data preparation, validation, tuning, screening, "
        "and interpretation workflow."
    )

st.sidebar.header("Project")
st.sidebar.write("Quantum Materials + Machine Learning")
st.sidebar.write("JARVIS-DFT 3D")
st.sidebar.write("Target: OptB88vdW band gap")
st.sidebar.markdown("[GitHub repository](https://github.com/ay23-byte/Quantum_materials_assited_machine_learning)")
