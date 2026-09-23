Quantum ESPRESSO starting package

Formula: K3CsTe2

Prototype: 1

Source: JVASP-87920 (K3InP2)

Validation: valid

Warnings: nan

Files:

- pw_relax.in: vc-relax starting input
- pw_scf.in: SCF starting input
- POSCAR: source structure
- structure.cif: source CIF when available
- pseudo/: selected .UPF/.upf files
- metadata.json: provenance

IMPORTANT:

Pseudopotentials are selected from
results/qe_pseudopotentials when exactly one
matching pseudopotential exists for each element.

Pseudopotentials are not downloaded or generated
by this script.

Use one internally consistent pseudopotential family
and exchange-correlation functional for production
calculations.

The cutoff (60/480 Ry), k-point mesh, smearing and
other settings are starting values only.

Perform convergence tests before production DFT.

The structure is hypothetical and prototype-transferred.
Stability, synthesizability and experimental
realizability are not established.

This script prepares Quantum ESPRESSO inputs only.
It does not run pw.x.
