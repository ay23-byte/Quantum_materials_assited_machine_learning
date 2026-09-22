Machine-Learning-Assisted Discovery of Quantum Materials

A research-oriented machine learning workflow for predicting DFT band gaps of materials from composition, periodic-table, and crystal-structure descriptors, followed by uncertainty analysis, error analysis, and candidate screening.

The project uses the JARVIS-DFT 3D materials database and focuses on building a reproducible ML pipeline that can be used to investigate whether relatively simple material descriptors are sufficient to predict electronic properties and whether structure-aware information improves generalization to chemically unseen materials.

Research Question

Can machine learning learn the relationship between material composition, periodic properties, and crystal structure and use this information to predict band gaps and prioritize candidate materials for further investigation?

A second question is particularly important:

Does adding crystal-structure information improve prediction when the model is evaluated on chemical systems that were not represented during training?

The second question motivates the chemical-system GroupKFold experiments in this project.

Dataset

The project uses the JARVIS-DFT 3D dataset.

Materials: 93,902
Target property: optb88vdw_bandgap
Target unit: eV
Target range: approximately 0–18 eV

The dataset contains both zero-gap and non-zero-gap materials, making the regression problem highly imbalanced in terms of the target distribution.

The enhanced descriptor dataset contains:

25 baseline descriptors
10 structure-aware descriptors
1 target variable

for a total of 36 columns.

Feature Engineering
Baseline descriptors

The baseline model uses 25 descriptors describing composition and periodic properties:

num_elements
total_atoms
mean_atomic_number
min_atomic_number
max_atomic_number
mean_atomic_mass
min_atomic_mass
max_atomic_mass
mean_atomic_radius
min_atomic_radius
max_atomic_radius
mean_electronegativity
min_electronegativity
max_electronegativity
electronegativity_difference
mean_ionization_energy
mean_electron_affinity
mean_s_valence
mean_p_valence
mean_d_valence
mean_f_val_valence
mean_period
mean_group
crys
spg_number

Note: the implementation uses mean_f_valence as the feature name. The feature list in the source code is the authoritative version.

Categorical descriptors such as crystal system and space-group number are one-hot encoded during preprocessing.

Structure-aware descriptors

Notebook 08 extends the baseline representation with 10 descriptors obtained directly from the JARVIS crystal structures:

num_sites
volume
volume_per_atom
density
lattice_a
lattice_b
lattice_c
lattice_alpha
lattice_beta
lattice_gamma

These describe the geometry and physical dimensions of the crystal unit cell.

The resulting structure-aware representation therefore contains:

25 baseline descriptors
+
10 structural descriptors
=
35 input features
Machine Learning Workflow

The project is organized as a sequence of increasingly rigorous experiments.

01 — Initial ML model

notebooks/01_ml_quantum_materials.ipynb

Introduces the prediction problem and establishes initial regression baselines.

Models include:

Mean baseline
Ridge regression
Random Forest regression

The Random Forest substantially improves over the simple statistical baseline.

02 — Materials-aware validation

notebooks/02_materials_aware_validation.ipynb

Investigates whether ordinary random train/test splitting gives an overly optimistic estimate of performance.

Materials are grouped according to their chemical system, and GroupKFold is used to prevent the same chemical system from appearing in both training and validation folds.

This provides a more realistic estimate of generalization to chemically different materials.

03 — Hyperparameter tuning

notebooks/03_hyperparameter_tuning.ipynb

Performs controlled Random Forest hyperparameter experiments.

The final tuned model uses:

RandomForestRegressor(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features=1.0,
    random_state=42,
    n_jobs=-1
)

Final untouched random-test performance:

Metric	Result
MAE	0.2381 eV
RMSE	0.5559 eV
R²	0.8195
04 — Candidate screening

notebooks/04_candidate_screening.ipynb

Uses the trained model to screen the full JARVIS dataset.

The workflow includes:

prediction for all 93,902 materials
screening for predicted band gaps around 1–2 eV
ranking candidates around a target band gap
retrospective validation
Random Forest tree-dispersion uncertainty proxy
duplicate/reduced-formula handling
chemical-diversity filtering

Approximately 6,861 materials fall into the initial predicted 1–2 eV screening region.

Candidate screening is treated as prioritization, not proof of discovering previously unknown materials.

05 — Model interpretation

notebooks/05_Interpretation.ipynb

Investigates which descriptors contribute most strongly to prediction.

Feature importance is examined using Random Forest importance and permutation-based analysis.

Important descriptors include quantities related to:

ionization energy
valence structure
electronegativity
atomic size
atomic composition
periodic position

These importance measures are interpreted as predictive associations, not causal physical relationships.

06 — Trustworthy ML

notebooks/06_Trustworthy_ML.ipynb

Adds uncertainty and reliability analysis to the prediction workflow.

Methods include:

Random Forest uncertainty proxy
Split conformal prediction
applicability-domain analysis
SHAP feature interpretation
permutation importance
Conformal prediction

For a nominal 90% prediction interval:

Nominal coverage: 90.0%
Observed test coverage: 89.92%
Interval half-width: 0.7849 eV
Interval width: 1.5698 eV

The conformal interval provides a calibrated coverage assessment on the held-out test set under the assumptions of the split-conformal procedure.

07 — Error analysis

notebooks/07_Error_Analysis.ipynb

Performs detailed analysis of model errors rather than reporting only aggregate metrics.

The analysis includes:

predicted vs actual band gap
residual analysis
error by band-gap range
error vs Random Forest uncertainty
error by crystal system
chemical-system error analysis
worst predictions
low-uncertainty/high-error cases

The final tuned model reproduces:

MAE  : 0.2381 eV
RMSE : 0.5559 eV
R²   : 0.8195

An important observation is that Random Forest prediction dispersion generally tracks error, but it is not itself a calibrated prediction interval.

08 — Structure-Aware Generalization

notebooks/08_Structural_Descriptors.ipynb

This experiment asks whether explicit crystal-structure information improves band-gap prediction.

Two representations are compared:

Baseline
25 composition + periodic descriptors
Structure-aware
25 baseline descriptors
+
10 crystal-structure descriptors

Both representations use the same Random Forest architecture.

Random split
Model	MAE (eV)	RMSE (eV)	R²
Baseline	0.2381	0.5559	0.8195
Structure-aware	0.2467	0.5497	0.8235

The random split does not produce a uniform improvement: MAE becomes slightly worse, while RMSE and R² improve.

Therefore, the random-split result alone is not sufficient to claim that structural descriptors improve generalization.

Chemical-system GroupKFold

The more stringent experiment separates materials by chemical system so that chemically related systems are not simultaneously represented in training and validation.

Results
Model	MAE (eV)	RMSE (eV)	R²
Baseline	0.3109 ± 0.0127	0.6773 ± 0.0425	0.7324 ± 0.0174
Structure-aware	0.3044 ± 0.0108	0.6511 ± 0.0338	0.7524 ± 0.0170

Relative change from adding structural descriptors:

MAE  : -2.10%
RMSE : -3.86%
R²   : +0.0200

This provides evidence that the structure-aware representation can improve generalization under the more challenging chemical-system split.

The improvement should be interpreted as predictive evidence, rather than proof that the individual structural descriptors are physically causal.

Model Performance Summary

The current strongest random-split result is:

MAE  = 0.2381 eV
RMSE = 0.5559 eV
R²   = 0.8195

However, the more conservative chemical-system evaluation gives:

Baseline:
MAE  = 0.3109 ± 0.0127 eV
RMSE = 0.6773 ± 0.0425 eV
R²   = 0.7324 ± 0.0174

Structure-aware:
MAE  = 0.3044 ± 0.0108 eV
RMSE = 0.6511 ± 0.0338 eV
R²   = 0.7524 ± 0.0170

This difference illustrates why materials-aware validation is important: random splits can contain chemically similar materials across training and test sets.

Uncertainty and Reliability

The project uses several complementary approaches rather than treating a single uncertainty measure as definitive.

Random Forest uncertainty proxy

The standard deviation of predictions from individual Random Forest trees is used as a practical uncertainty indicator.

This is useful for identifying regions where the model behaves less consistently, but it is not a calibrated prediction interval.

Split conformal prediction

Conformal prediction is used separately when calibrated interval coverage is required.

Applicability domain

Nearest-neighbor distances in standardized descriptor space are used to investigate whether predictions occur near or far from the training distribution.

Candidate Screening

The final model is refitted on all labelled materials for large-scale screening.

The screening pipeline generates:

results/all_material_predictions.csv

with predictions and uncertainty proxies for all 93,902 materials.

The screening workflow can identify materials whose predicted band gaps fall within a specified target range.

These candidates should be interpreted as computationally prioritized materials for further investigation, not experimentally validated discoveries.

Project Structure
Quantum-materials/
│
├── notebooks/
│   ├── 01_ml_quantum_materials.ipynb
│   ├── 02_materials_aware_validation.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   ├── 04_candidate_screening.ipynb
│   ├── 05_Interpretation.ipynb
│   ├── 06_Trustworthy_ML.ipynb
│   ├── 07_Error_Analysis.ipynb
│   └── 08_Structural_Descriptors.ipynb
│
├── src/
│   ├── descriptors.py
│   ├── preprocessing.py
│   ├── models.py
│   └── evaluation.py
│
├── data/
│   └── enhanced_material_descriptors.csv
│
├── results/
│   ├── final_test_predictions.csv
│   ├── final_model_metrics.json
│   ├── hyperparameter_screening.csv
│   ├── all_material_predictions.csv
│   ├── bandgap_screening_candidates.csv
│   ├── uncertainty_aware_bandgap_candidates.csv
│   ├── chemically_diverse_bandgap_candidates.csv
│   ├── shap_feature_importance.csv
│   ├── advanced_permutation_importance.csv
│   ├── advanced_trustworthiness_summary.json
│   ├── error_analysis_summary.json
│   └── structural_feature_importance.csv
│
├── figures/
│
├── app.py
├── requirements.txt
└── README.md
Streamlit Application

The project also contains an interactive Streamlit application:

app.py

The application is intended to make the research workflow easier to explore, including model predictions, material descriptors, uncertainty information, and screening results.

Run locally with:

streamlit run app.py
Reproducibility

Create the project environment and install the required packages:

pip install -r requirements.txt

The project uses:

Python
NumPy
pandas
SciPy
scikit-learn
Matplotlib
Seaborn
SHAP
Streamlit
JARVIS-Tools

The main model uses fixed random seeds where appropriate to make the experiments reproducible.

Important Scientific Limitations

This project is a computational materials-screening study, not an experimental validation study.

Important limitations include:

DFT dependence
The target property comes from the JARVIS-DFT dataset and therefore inherits the assumptions and limitations of the underlying DFT calculations.
Random-split optimism
Random train/test splits can place chemically similar materials in both sets.
Chemical-system validation is stricter
Chemical-system GroupKFold gives a more conservative estimate of extrapolation to unseen chemistry.
Prediction is not explanation
Feature importance identifies predictive relationships, not causal physical mechanisms.
Uncertainty proxy limitations
Random Forest tree dispersion should not be interpreted as a calibrated confidence interval.
Candidate screening is prioritization
A high or low predicted band gap does not establish experimental realizability or stability.
Structural representation is limited
The current structure-aware representation uses unit-cell-level descriptors and does not explicitly encode the complete atomic environment or local bonding topology.
Future Work

Possible extensions include:

gradient-boosting benchmark models
stronger materials-aware validation strategies
richer local structural descriptors
graph-based material representations
improved uncertainty calibration
systematic structure-aware ablation studies
experimental/DFT validation of shortlisted candidates
multi-property prediction
active-learning based candidate selection

The goal is to progressively move from descriptor-based prediction toward more physically informed and trustworthy materials discovery workflows.

Research Perspective

The central idea of this project is not simply to obtain a high R² score.

Instead, the workflow asks:

Can we predict a useful materials property?
        ↓
Does the prediction survive stricter validation?
        ↓
Which descriptors contribute?
        ↓
How reliable is the prediction?
        ↓
Where does the model fail?
        ↓
Does structural information improve generalization?
        ↓
Can the model prioritize materials for further study?

This makes the project a computational investigation into machine-learning-assisted quantum-materials discovery, rather than a standalone machine-learning classification exercise.


09 — Final research results

notebooks/09_Final_Research_Results.ipynb

Consolidates the final research evaluation of the composition-based and structure-aware models.

The study reports both an untouched random test split and a stricter chemical-system GroupKFold evaluation. The structure-aware representation adds 10 crystal-structure descriptors to the 25 composition/periodic descriptors.

The structure-aware model improves the chemical-system GroupKFold results from:

MAE 0.3109 eV
RMSE 0.6773 eV
R² 0.7324

to:

MAE 0.3044 eV
RMSE 0.6511 eV
R² 0.7524

The notebook also documents retrospective screening of known JARVIS materials around a target band gap. This is prioritization of known materials, not discovery of experimentally unknown materials.

10 — Advanced model optimization

notebooks/10_Advanced_Model_Optimization.ipynb

Benchmarks additional model families using the 35-descriptor structure-aware representation, including Random Forest, Extra Trees, HistGradientBoosting, and a two-stage Extra Trees approach.

The notebook is used as a model-family optimization experiment. Its random train/test comparison should be interpreted separately from the stricter chemical-system GroupKFold evaluation.

New-materials extension

The project now includes a composition-space exploration stage for generating hypothetical materials from known chemical elements:

src/generate_hypothetical_candidates.py

The workflow:

1. Generate charge-balanced binary and ternary compositions.
2. Generate additional charge-balanced quaternary compositions.
3. Canonicalize formulas to compare compositions independently of element ordering.
4. Remove compositions already represented in the JARVIS-DFT reference set.
5. Calculate the same 25 composition/periodic descriptors used by the existing model.
6. Mark crystal system and space-group information as unknown because hypothetical compositions do not have a known crystal structure at this stage.
7. Fit the existing composition-based Random Forest model on the labelled JARVIS training data.
8. Predict the band gap of the hypothetical compositions.
9. Estimate a Random Forest tree-dispersion uncertainty proxy.
10. Filter candidates by target-band-gap distance and uncertainty.
11. Rank the remaining candidates and save them to:

results/hypothetical_candidates.csv

Example:

python src/generate_hypothetical_candidates.py --target 1.5 --top-k 200

This extension changes the research question from only screening known materials to exploring a larger composition space for hypothetical candidates. However, a generated formula is not automatically a new material in the experimental or literature sense.

The generated candidates are hypotheses only. The current model does not establish:

- crystal structure
- thermodynamic stability
- dynamical stability
- synthesizability
- formation energy
- complete chemical novelty in the literature
- experimental realization

The next research stage should therefore be structure generation, DFT relaxation/formation-energy validation, stability checks, and electronic-property validation before describing a candidate as a realistic material-discovery result.

New-materials workflow

Known JARVIS materials
        ↓
Train validated ML model
        ↓
Generate hypothetical compositions
        ↓
Remove known JARVIS compositions
        ↓
Calculate descriptors
        ↓
Predict band gap
        ↓
Uncertainty / applicability-domain filtering
        ↓
Rank candidates
        ↓
Generate plausible crystal structures
        ↓
DFT relaxation and stability validation
        ↓
Band-gap validation
        ↓
Candidates for further experimental consideration

This is deliberately framed as ML-assisted candidate generation and prioritization rather than automatic discovery.

Updated project structure

Quantum-materials/
│
├── notebooks/
│   ├── 01_ml_quantum_materials.ipynb
│   ├── 02_materials_aware_validation.ipynb
│   ├── 03_hyperparameter_tuning.ipynb
│   ├── 04_candidate_screening.ipynb
│   ├── 05_Interpretation.ipynb
│   ├── 06_Trustworthy_ML.ipynb
│   ├── 07_Error_Analysis.ipynb
│   ├── 08_Structural_Descriptors.ipynb
│   ├── 09_Final_Research_Results.ipynb
│   └── 10_Advanced_Model_Optimization.ipynb
│
├── src/
│   ├── descriptors.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── evaluation.py
│   └── generate_hypothetical_candidates.py
│
├── data/
├── results/
│   └── hypothetical_candidates.csv
├── figures/
├── app.py
├── requirements.txt
└── README.md

Author

Ayush Kumar Prajapati
B.Tech Engineering Physics
National Institute of Technology Hamirpur

Research interests:

Quantum Materials
Condensed Matter Physics
Computational Physics
Quantum Many-Body Systems
Machine Learning for Materials Science