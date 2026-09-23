# Hypothetical Materials Shortlist

## Purpose

This report prioritizes ML-generated hypothetical compositions for the next research stage. It is a screening result, not a claim of material discovery.

## Source and selection

- Source candidates read: **200**
- Final shortlist: **15**
- Ranking inputs: predicted band gap, distance from target, Random Forest tree-dispersion uncertainty proxy, chemistry score, and applicability-domain filtering already performed by the generator.
- Diversity controls: chemistry-family signature, exact element-system signature, reduced composition family, and stoichiometry class.
- No silent fallback is used to violate diversity constraints.

## Scientific interpretation

The compositions below are hypothetical and were generated from a restricted set of known chemical elements and simple charge-balance rules. The ML model predicts a target band gap from composition-derived descriptors. It does not establish crystal structure, thermodynamic stability, synthesizability, phonon stability, experimental novelty, or an experimentally measurable band gap.

The Random Forest uncertainty column is a tree-dispersion proxy and is not a calibrated predictive interval. Candidates therefore require structural generation and first-principles validation before stronger claims are made.

## Recommended next stage

1. Generate plausible crystal structures for the shortlisted compositions.
2. Perform structure relaxation and formation-energy calculations with DFT.
3. Remove energetically unfavorable or structurally unstable candidates.
4. Recalculate the electronic structure and band gap with DFT.
5. Apply phonon/stability checks where computationally feasible.
6. Retain only candidates supported by the combined composition, structure, and DFT evidence.

## Shortlist

|   shortlist_rank | formula    |   predicted_bandgap_eV |   uncertainty_proxy_eV |   distance_from_target_eV |   screening_score |   chemistry_score |   in_domain_fraction | stoichiometry_class   | cation_family          | anion_family        |
|-----------------:|:-----------|-----------------------:|-----------------------:|--------------------------:|------------------:|------------------:|---------------------:|:----------------------|:-----------------------|:--------------------|
|                1 | K3CsTe2    |                 1.5023 |                 0.5103 |                    0.0023 |            0.1651 |            0.6475 |                    1 | ternary               | alkali                 | chalcogen           |
|                2 | KCs3Te2    |                 1.5023 |                 0.5103 |                    0.0023 |            0.1651 |            0.6475 |                    1 | ternary               | alkali                 | chalcogen           |
|                3 | KCsSe      |                 1.5012 |                 0.6394 |                    0.0012 |            0.1738 |            0.8725 |                    1 | ternary               | alkali                 | chalcogen           |
|                4 | RbInNCl    |                 1.4988 |                 0.7201 |                    0.0012 |            0.1912 |            0.9    |                    1 | quaternary            | alkali+post_transition | halogen+pnictogen   |
|                5 | K3InO2Se   |                 1.498  |                 0.6957 |                    0.002  |            0.1912 |            0.8475 |                    1 | quaternary            | alkali+post_transition | chalcogen           |
|                6 | K3InOSe2   |                 1.498  |                 0.6957 |                    0.002  |            0.1912 |            0.8475 |                    1 | quaternary            | alkali+post_transition | chalcogen           |
|                7 | CsAlOSe    |                 1.5068 |                 0.7077 |                    0.0068 |            0.194  |            0.8975 |                    1 | quaternary            | alkali+post_transition | chalcogen           |
|                8 | Rb2AlNI2   |                 1.5081 |                 0.6831 |                    0.0081 |            0.1971 |            0.8175 |                    1 | quaternary            | alkali+post_transition | halogen+pnictogen   |
|                9 | RbAl2N2I   |                 1.5081 |                 0.6831 |                    0.0081 |            0.1971 |            0.8175 |                    1 | quaternary            | alkali+post_transition | halogen+pnictogen   |
|               10 | Ca2Sr4N3I3 |                 1.521  |                 0.71   |                    0.021  |            0.2048 |            0.9375 |                    1 | quaternary            | alkaline_earth         | halogen+pnictogen   |
|               11 | Ca4Sr2N3I3 |                 1.521  |                 0.71   |                    0.021  |            0.2048 |            0.9375 |                    1 | quaternary            | alkaline_earth         | halogen+pnictogen   |
|               12 | Ca3BaN2Br2 |                 1.4577 |                 0.6776 |                    0.0423 |            0.2117 |            1      |                    1 | quaternary            | alkaline_earth         | halogen+pnictogen   |
|               13 | K5NTe      |                 1.4983 |                 0.7988 |                    0.0017 |            0.2139 |            0.875  |                    1 | ternary               | alkali                 | chalcogen+pnictogen |
|               14 | Rb2Si2Te5  |                 1.5151 |                 0.5935 |                    0.0151 |            0.2265 |            0.37   |                    1 | ternary               | alkali+metalloid       | chalcogen           |
|               15 | Rb4SiTe4   |                 1.5151 |                 0.5935 |                    0.0151 |            0.2265 |            0.37   |                    1 | ternary               | alkali+metalloid       | chalcogen           |

Generated automatically by src/build_hypothetical_shortlist.py.