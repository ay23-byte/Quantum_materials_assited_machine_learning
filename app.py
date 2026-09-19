import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Quantum Materials ML", page_icon="🔬", layout="wide")


@st.cache_data
def load_predictions():
    path = RESULTS / "all_material_predictions.csv"
    if not path.exists():
        return pd.DataFrame()
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
predictions = load_predictions()

tab1, tab2, tab3 = st.tabs(
    ["Model performance", "Candidate screening", "About the project"]
)

with tab1:
    st.subheader("Final model performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Test MAE", f"{metrics.get('test_MAE_eV', 0):.3f} eV")
    c2.metric("Test RMSE", f"{metrics.get('test_RMSE_eV', 0):.3f} eV")
    c3.metric("Test R²", f"{metrics.get('test_R2', 0):.3f}")

    st.markdown("### Materials-aware validation")
    g1, g2, g3 = st.columns(3)
    g1.metric(
        "GroupKFold MAE",
        f"{group_metrics.get('MAE_mean_eV', 0):.3f} ± "
        f"{group_metrics.get('MAE_std_eV', 0):.3f} eV",
    )
    g2.metric(
        "GroupKFold RMSE",
        f"{group_metrics.get('RMSE_mean_eV', 0):.3f} ± "
        f"{group_metrics.get('RMSE_std_eV', 0):.3f} eV",
    )
    g3.metric(
        "GroupKFold R²",
        f"{group_metrics.get('R2_mean', 0):.3f} ± "
        f"{group_metrics.get('R2_std', 0):.3f}",
    )

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

    if predictions.empty:
        st.error(
            "The full screening file was not found. Expected: "
            "results/all_material_predictions.csv"
        )
    else:
        st.write(
            "Search the complete JARVIS-DFT dataset using the selected target "
            "band gap, prediction tolerance, and Random Forest uncertainty proxy. "
            "These are known materials retrospectively prioritized by the model; "
            "this is a screening demonstration, not a claim of experimental discovery."
        )

        c1, c2, c3 = st.columns(3)
        target = c1.number_input(
            "Target band gap (eV)",
            min_value=0.0,
            max_value=10.0,
            value=1.5,
            step=0.1,
        )
        max_distance = c2.slider(
            "Maximum predicted distance (eV)",
            min_value=0.01,
            max_value=2.0,
            value=0.10,
            step=0.01,
        )
        max_uncertainty = c3.slider(
            "Maximum uncertainty proxy (eV)",
            min_value=0.01,
            max_value=2.0,
            value=0.40,
            step=0.01,
        )

        df = predictions.copy()
        df["distance_from_selected_target"] = (
            df["predicted_bandgap"] - target
        ).abs()

        filtered = df[
            (df["distance_from_selected_target"] <= max_distance)
            & (df["prediction_uncertainty"] <= max_uncertainty)
        ].sort_values(
            ["distance_from_selected_target", "prediction_uncertainty"]
        )

        st.success(
            f"Found **{len(filtered):,}** matching materials out of "
            f"**{len(df):,}** total materials."
        )

        if not filtered.empty:
            m1, m2, m3 = st.columns(3)
            m1.metric(
                "Closest predicted band gap",
                f"{filtered.iloc[0]['predicted_bandgap']:.3f} eV",
            )
            m2.metric("Closest material", str(filtered.iloc[0]["formula"]))
            m3.metric(
                "Lowest uncertainty",
                f"{filtered['prediction_uncertainty'].min():.3f} eV",
            )

            display_cols = [
                "jid",
                "formula",
                "predicted_bandgap",
                "prediction_uncertainty",
                "target_bandgap",
                "crys",
                "spg_number",
            ]
            available = [c for c in display_cols if c in filtered.columns]

            st.markdown("### Matching materials")
            st.dataframe(
                filtered[available].head(100),
                width="stretch",
                hide_index=True,
            )

            st.markdown("### Predicted band gaps of top matches")
            chart_df = (
                filtered[["formula", "predicted_bandgap"]]
                .head(30)
                .copy()
            )
            chart_df["formula"] = chart_df["formula"].astype(str)
            chart_df = chart_df.set_index("formula")
            st.bar_chart(chart_df, width="stretch")

            st.download_button(
                "Download filtered candidates as CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="filtered_bandgap_candidates.csv",
                mime="text/csv",
                width="stretch",
            )
        else:
            st.warning(
                "No materials match the current filters. Increase the maximum "
                "distance or uncertainty, or choose another target band gap."
            )

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
        "and interpretation workflow. The all-material screening predictions are "
        "generated by src/generate_screening_predictions.py."
    )

st.sidebar.header("Project")
st.sidebar.write("Quantum Materials + Machine Learning")
st.sidebar.write("JARVIS-DFT 3D")
st.sidebar.write("Target: OptB88vdW band gap")
st.sidebar.markdown(
    "[GitHub repository](https://github.com/ay23-byte/Quantum_materials_assited_machine_learning)"
)