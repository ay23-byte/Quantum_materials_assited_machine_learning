# DFT preparation workflow

After hypothetical structures pass the structural sanity validator, the project
can package them into reproducible DFT starting directories.

Run:

```powershell
python .\src\prepare_dft_jobs.py
```

The script reads:

```
results/hypothetical_structures/structure_validation.csv
```

and prepares every structure with status `valid` or
`valid_with_warnings`.

Output:

```
results/dft_jobs/
├── <formula>/
│   └── prototype_<N>/
│       ├── POSCAR
│       ├── INCAR.template
│       ├── KPOINTS.template
│       ├── metadata.json
│       └── README.txt
├── dft_job_manifest.csv
└── dft_job_summary.json
```

The package is a **DFT starting point**, not a DFT result. The POSCARs are
prototype-transferred initial structures. The supplied INCAR and KPOINTS are
conservative starting templates; convergence tests, magnetic settings,
pseudopotential compatibility, and the final exchange-correlation/dispersion
method must be checked before production calculations.

POTCAR files are deliberately not generated because they depend on the
user's licensed pseudopotential library.

The intended scientific sequence is:

1. Prototype-transferred initial structure.
2. Structural sanity validation.
3. DFT geometry relaxation.
4. Formation energy / energy-above-hull analysis.
5. Electronic-structure and band-gap calculation.
6. Dynamical-stability checks for selected candidates.
7. Experimental consideration only after the computational evidence supports it.

Passing structural validation does not establish stability, synthesizability,
or experimental realizability.
