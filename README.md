# Machine-Learning-Assisted Discovery of Novel Quantum Materials

A computational materials-science project that uses physics-informed material descriptors and machine learning to predict density-functional-theory (DFT) band gaps and screen materials for target electronic properties.

---

## Overview

The discovery of materials with desirable electronic properties requires exploring a very large chemical and structural space. First-principles calculations such as density functional theory (DFT) provide valuable information, but evaluating thousands of materials can be computationally expensive.

This project investigates whether machine learning can learn relationships between relatively simple material descriptors and the calculated electronic band gap of materials.

The workflow uses the JARVIS-DFT 3D materials database and focuses on the:

**OptB88vdW DFT band gap (`optb88vdw_bandgap`)**

The project combines:

- Materials informatics
- Computational condensed-matter physics
- Feature engineering
- Machine learning
- Materials-aware validation
- Candidate screening
- Model interpretation

The goal is not simply to build a high-performing ML model, but to investigate how well the model generalizes and whether it can be used as a computational screening tool.

---

## Research Question

> Can machine learning learn the relationship between chemically and structurally informed material descriptors and DFT-calculated band gaps, and can the resulting model be used to screen materials for a target band-gap range?

---

## Dataset

The project uses the **JARVIS-DFT 3D materials dataset**.

Dataset size:

- **93,902 materials**
- **64 original columns**
- Target: `optb88vdw_bandgap`

The target represents the OptB88vdW DFT-calculated band gap in electron volts (eV).

### Target distribution

A large fraction of the dataset has a calculated band gap of exactly 0 eV, while the remaining materials have non-zero band gaps.

This makes the prediction problem highly non-uniform and motivates detailed error analysis across different band-gap regions.

---

## Feature Engineering

Instead of directly using calculated electronic properties such as HSE or MBJ band gaps as input features, the model uses descriptors derived from composition and crystal structure.

This avoids target leakage.

### Composition descriptors

Examples include:

- Number of elements
- Total number of atoms
- Mean atomic number
- Minimum atomic number
- Maximum atomic number
- Mean atomic mass
- Minimum atomic mass
- Maximum atomic mass
- Mean atomic radius
- Minimum atomic radius
- Maximum atomic radius

### Electronegativity descriptors

- Mean electronegativity
- Minimum electronegativity
- Maximum electronegativity
- Electronegativity difference

### Electronic/atomic descriptors

- Mean ionization energy
- Mean electron affinity

### Valence descriptors

- Mean s-valence
- Mean p-valence
- Mean d-valence
- Mean f-valence

### Periodic-table descriptors

- Mean period
- Mean group

### Structural descriptors

- Crystal system (`crys`)
- Space-group number (`spg_number`)

Categorical structural variables are one-hot encoded rather than treated as continuous numerical quantities.

---

## Machine-Learning Model

The main model is a:

**Random Forest Regressor**

Final hyperparameters:

```text
n_estimators = 200
max_depth = None
min_samples_split = 2
min_samples_leaf = 1
max_features = 1.0
random_state = 42