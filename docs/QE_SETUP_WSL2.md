# Quantum ESPRESSO setup on Windows with WSL2

This project uses Quantum ESPRESSO (QE) for DFT validation of the hypothetical-material structures.

## Recommended environment

Use WSL2 + Ubuntu on Windows. Microsoft documents `wsl --install` as the standard installation route and installs Ubuntu by default.

Official references:
- Microsoft WSL: https://learn.microsoft.com/en-us/windows/wsl/install
- Quantum ESPRESSO documentation: https://www.quantum-espresso.org/documentation/
- Quantum ESPRESSO pseudopotentials: https://www.quantum-espresso.org/pseudopotentials/
- SSSP pseudopotential library: https://www.materialscloud.org/discover/sssp

## 1. Install WSL2

Open **PowerShell as Administrator**:

```powershell
wsl --install
```

Restart Windows if requested.

Then open **Ubuntu** from the Start menu and create your Linux username/password.

Check WSL from PowerShell:

```powershell
wsl --status
wsl -l -v
```

The Ubuntu distribution should use version 2.

## 2. Update Ubuntu

Inside Ubuntu:

```bash
sudo apt update
sudo apt upgrade -y
```

Install the compiler/build dependencies:

```bash
sudo apt install -y \
  build-essential \
  gfortran \
  git \
  wget \
  curl \
  libfftw3-dev \
  libopenblas-dev \
  liblapack-dev \
  libscalapack-mpi-dev \
  openmpi-bin \
  libopenmpi-dev
```

## 3. Install Quantum ESPRESSO

The current QE documentation identifies **7.5.0** as the current stable release. Use the official download page rather than an unofficial binary.

Official download:
https://www.quantum-espresso.org/download/

After downloading the source archive, place it in Ubuntu and extract it:

```bash
tar -xzf qe-7.5.0.tar.gz
cd qe-7.5
```

Configure:

```bash
./configure
```

Build the PWscf code:

```bash
make -j2 pw
```

Check that the executable exists:

```bash
ls -lh bin/pw.x
```

Test:

```bash
./bin/pw.x -h
```

If `pw.x` starts and prints its help/usage information, the basic installation is working.

## 4. Access the Windows project from Ubuntu

Your Windows F: drive is normally mounted at:

```text
/mnt/f
```

So the project should be accessible as:

```bash
cd /mnt/f/Quantum-materials
```

Check:

```bash
ls
```

You should see `src`, `results`, `data`, etc.

## 5. Pseudopotentials

Do **not** download random UPF files independently.

Quantum ESPRESSO supports NC, ultrasoft and PAW pseudopotentials, and its documentation recommends testing pseudopotentials before serious calculations. The SSSP collection is a curated option designed for consistent accuracy/efficiency.

For this project we will choose one internally consistent PBE-based pseudopotential family and record the exact filenames and provenance.

Create the shared Windows-side directory:

```powershell
New-Item -ItemType Directory -Force F:\Quantum-materials\results\qe_pseudopotentials
```

The repository's QE preparation script now searches this directory. If exactly one `.UPF` file is present for an element, it copies that file into each relevant job and writes the real filename into `ATOMIC_SPECIES`.

If no pseudopotential is available, the script leaves a placeholder such as `K.UPF` and marks the missing element in `metadata.json`.

For strict checking after pseudopotentials are installed:

```powershell
python .\src\prepare_qe_jobs.py --strict-pseudo
```

## 6. Important scientific rule

The generated structures are prototype-transferred hypothetical structures. They are not assumed to be stable.

The current values:

- ecutwfc = 60 Ry
- ecutrho = 480 Ry
- automatic k-point mesh

are **starting values only**.

Before production calculations we will perform:

1. pseudopotential sanity checks
2. plane-wave cutoff convergence
3. k-point convergence
4. `vc-relax`
5. final SCF
6. band-structure / band-gap calculation
7. formation-energy or related stability analysis
8. phonon stability where computationally feasible

Only after those checks can a candidate be treated as a DFT-supported prediction rather than an ML-generated hypothesis.
