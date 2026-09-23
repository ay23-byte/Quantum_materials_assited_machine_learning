"""Prepare validated hypothetical structures for Quantum ESPRESSO.

Creates pw.x vc-relax and scf starting inputs. It does not run DFT or
download pseudopotentials.
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
VALIDATION_FILE = ROOT / "results" / "hypothetical_structures" / "structure_validation.csv"
DEFAULT_OUTPUT = ROOT / "results" / "qe_jobs"

RELAX = """&CONTROL
  calculation = 'vc-relax'
  prefix = '{prefix}'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tstress = .true.
  tprnfor = .true.
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = 60
  ecutrho = 480
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-8
  mixing_beta = 0.3
/
&IONS
  ion_dynamics = 'bfgs'
/
&CELL
  cell_dynamics = 'bfgs'
  press = 0.0
  cell_dofree = 'all'
/
ATOMIC_SPECIES
{species}
CELL_PARAMETERS angstrom
{cell}
ATOMIC_POSITIONS crystal
{positions}
K_POINTS automatic
{kpoints}
"""

SCF = """&CONTROL
  calculation = 'scf'
  prefix = '{prefix}'
  outdir = './tmp'
  pseudo_dir = './pseudo'
  tstress = .true.
  tprnfor = .true.
/
&SYSTEM
  ibrav = 0
  nat = {nat}
  ntyp = {ntyp}
  ecutwfc = 60
  ecutrho = 480
  occupations = 'smearing'
  smearing = 'mv'
  degauss = 0.02
/
&ELECTRONS
  conv_thr = 1.0d-10
  mixing_beta = 0.3
/
ATOMIC_SPECIES
{species}
CELL_PARAMETERS angstrom
{cell}
ATOMIC_POSITIONS crystal
{positions}
K_POINTS automatic
{kpoints}
"""

def args():
    p = argparse.ArgumentParser()
    p.add_argument("--validation-file", type=Path, default=VALIDATION_FILE)
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--status", choices=["all", "valid", "valid_with_warnings"], default="all")
    return p.parse_args()

def root(p):
    return p if p.is_absolute() else ROOT / p

def read_poscar(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    scale = float(lines[1])
    if abs(scale - 1.0) > 1e-10:
        raise ValueError("Expected POSCAR scale factor 1.0.")
    elements = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    i = 7
    if lines[i].strip().lower().startswith("s"):
        i += 1
    mode = lines[i].strip().lower()
    i += 1
    lattice = [[float(x) for x in lines[2+j].split()[:3]] for j in range(3)]
    raw = [[float(x) for x in lines[i+j].split()[:3]] for j in range(sum(counts))]
    if mode.startswith("d"):
        frac = raw
    elif mode.startswith(("c", "k")):
        import numpy as np
        frac = (np.array(raw) @ np.linalg.inv(np.array(lattice).T)).tolist()
    else:
        raise ValueError("Unsupported POSCAR coordinate mode.")
    positions = []
    n = 0
    for el, count in zip(elements, counts):
        for j in range(count):
            positions.append((el, frac[n+j]))
        n += count
    return elements, counts, lattice, positions

def mesh(row):
    vals = []
    for key in ("a_A", "b_A", "c_A"):
        try: v = max(float(row[key]), 1.0)
        except (TypeError, ValueError): v = 10.0
        vals.append(max(1, min(8, round(25.0/v))))
    return tuple(vals)

def make_input(row, out):
    formula = str(row["formula"])
    idx = int(row["prototype_index"])
    source = root(Path(str(row["poscar"])))
    elements, counts, lattice, positions = read_poscar(source)
    job = out / formula / f"prototype_{idx}"
    job.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, job / "POSCAR")
    cif = source.with_name("structure.cif")
    if cif.exists(): shutil.copy2(cif, job / "structure.cif")
    ka, kb, kc = mesh(row)
    species = "\n".join(f"{e} 1.0 {e}.UPF" for e in elements)
    cell = "\n".join(" ".join(f"{x:.10f}" for x in v) for v in lattice)
    pos = "\n".join(f"{e} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}" for e,p in positions)
    data = dict(prefix=f"{formula}_p{idx}", nat=sum(counts), ntyp=len(elements),
                species=species, cell=cell, positions=pos,
                kpoints=f"{ka} {kb} {kc} 0 0 0")
    (job/"pw_relax.in").write_text(RELAX.format(**data), encoding="utf-8")
    (job/"pw_scf.in").write_text(SCF.format(**data), encoding="utf-8")
    (job/"pseudo").mkdir(exist_ok=True)
    warnings = str(row.get("warnings","")).strip() or "none"
    metadata = {
        "formula": formula, "prototype_index": idx,
        "prototype_source_jid": str(row["prototype_source_jid"]),
        "prototype_source_formula": str(row["prototype_source_formula"]),
        "prototype": str(row["prototype"]),
        "lattice_scale": float(row["lattice_scale"]),
        "validation_status": str(row["status"]),
        "validation_warnings": warnings,
        "kpoint_mesh": [ka,kb,kc],
        "ecutwfc_Ry_start": 60, "ecutrho_Ry_start": 480,
        "pseudopotentials": [f"{e}.UPF" for e in elements],
        "scientific_status": "prototype-transferred initial structure; not relaxed or stability-validated"
    }
    (job/"metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    (job/"README.txt").write_text(
        f"""Quantum ESPRESSO starting package
Formula: {formula}
Prototype: {idx}
Source: {row["prototype_source_jid"]} ({row["prototype_source_formula"]})
Validation: {row["status"]}
Warnings: {warnings}

Files:
- pw_relax.in: vc-relax starting input
- pw_scf.in: SCF starting input
- POSCAR: source structure
- structure.cif: source CIF when available
- pseudo/: place the chosen .UPF files here
- metadata.json: provenance

IMPORTANT:
Pseudopotentials are not included. Choose one consistent pseudopotential
family and verify the exchange-correlation functional.

The cutoff (60/480 Ry), k-point mesh, smearing and other settings are
starting values only. Perform convergence tests before production DFT.
The structure is hypothetical and prototype-transferred; stability,
synthesizability and experimental realizability are not established.
""", encoding="utf-8")
    return {"formula":formula, "prototype_index":idx,
            "validation_status":str(row["status"]),
            "prototype_source_jid":str(row["prototype_source_jid"]),
            "prototype_source_formula":str(row["prototype_source_formula"]),
            "lattice_scale":float(row["lattice_scale"]),
            "kpoints":f"{ka} {kb} {kc}",
            "job_directory":str(job.relative_to(ROOT)),
            "pw_relax":str((job/"pw_relax.in").relative_to(ROOT)),
            "pw_scf":str((job/"pw_scf.in").relative_to(ROOT)),
            "pseudo_directory":str((job/"pseudo").relative_to(ROOT)),
            "warnings":warnings}

def main():
    a = args()
    vf, out = root(a.validation_file), root(a.output_root)
    if not vf.exists(): raise FileNotFoundError(f"Validation file not found: {vf}")
    df = pd.read_csv(vf)
    df = df[df["status"].isin(["valid","valid_with_warnings"])].copy()
    if a.status != "all": df = df[df["status"] == a.status].copy()
    if df.empty: raise RuntimeError("No structures match the requested validation status.")
    out.mkdir(parents=True, exist_ok=True)
    print(f"Structures to prepare: {len(df)}")
    rows=[]
    for _, row in df.iterrows():
        print(f"Preparing {row['formula']} / prototype {int(row['prototype_index'])}...")
        rows.append(make_input(row,out))
        print("  [OK] Quantum ESPRESSO starting package prepared")
    manifest=pd.DataFrame(rows)
    mf=out/"qe_job_manifest.csv"; sf=out/"qe_job_summary.json"
    manifest.to_csv(mf,index=False)
    summary={"source_validation_file":str(vf.relative_to(ROOT)),
             "output_root":str(out.relative_to(ROOT)),
             "prepared_structures":len(manifest),
             "valid":int((manifest["validation_status"]=="valid").sum()),
             "valid_with_warnings":int((manifest["validation_status"]=="valid_with_warnings").sum()),
             "pseudopotentials_generated":False,
             "interpretation":"Quantum ESPRESSO input preparation only; convergence and pseudopotential checks are required.",
             "manifest_file":str(mf.relative_to(ROOT))}
    sf.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("\nQuantum ESPRESSO job preparation complete.")
    print(f"Prepared: {summary['prepared_structures']}")
    print(f"Manifest: {mf}")
    print(f"Summary: {sf}")

if __name__ == "__main__":
    main()
