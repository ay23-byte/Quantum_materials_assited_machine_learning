# K3CsTe2: final non-SOC DFT result

## Calculation status

K3CsTe2 was structurally relaxed with Quantum ESPRESSO using a non-SOC PBE calculation. The resulting relaxed geometry was then used for fixed-geometry electronic-structure validation.

SOC validation is a separate ongoing calculation and is **not** included in the numerical results below.

## Relaxation result

| Quantity | Result |
|---|---:|
| Final total energy | -911.04528610 Ry |
| Final force reported | 0.002107 Ry/Bohr |
| Cell gradient | 0.15 kbar |
| Residual stress | < 0.2 kbar |

The relaxed cell and fractional atomic coordinates are stored in `relaxed_structure.in`.

## Non-SOC electronic structure

The fixed relaxed structure gives:

| Quantity | Result |
|---|---:|
| Method | PBE, non-SOC |
| VBM | 2.293961 eV |
| VBM band | 48 |
| CBM | 4.052416 eV |
| CBM location | Gamma |
| CBM band | 49 |
| Indirect band gap | **1.758455 eV** |

The dense band-structure path was generated for the triclinic P-1 structure (space group #2).

## Interpretation

The 1.758455 eV value is a **non-SOC PBE band gap for the relaxed prototype-transferred K3CsTe2 structure**. It should not be interpreted as an experimentally established band gap or as proof of thermodynamic stability.

The structure originated from prototype transfer and was subsequently relaxed computationally. Experimental synthesizability, thermodynamic stability, dynamical stability, and SOC-corrected electronic properties require additional validation.

## Reproducibility

The repository contains the starting QE relaxation/SCF inputs under:

`results/qe_jobs/K3CsTe2/prototype_1/`

The relaxed geometry used for the final non-SOC electronic result is stored separately in:

`results/qe_jobs/K3CsTe2/prototype_1/relaxed_structure.in`
