"""Descriptor-generation utilities for composition-based materials ML."""

from __future__ import annotations

import re
from typing import Dict

import numpy as np
from jarvis.core.specie import Specie


_FORMULA_PATTERN = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> Dict[str, float]:
    """Parse a simple chemical formula into element -> stoichiometric amount."""
    composition: Dict[str, float] = {}

    for element, count in _FORMULA_PATTERN.findall(str(formula)):
        amount = float(count) if count else 1.0
        composition[element] = composition.get(element, 0.0) + amount

    return composition


def element_system(formula: str) -> str:
    """Return a sorted unique-element chemical-system key."""
    elements = sorted(parse_formula(formula))
    return "-".join(elements)


def _valid_property(value):
    """Convert JARVIS sentinel values to NaN."""
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if value <= -9990:
        return np.nan

    return value


def _safe_mean(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.mean(values)) if values else np.nan


def _safe_min(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.min(values)) if values else np.nan


def _safe_max(values):
    values = [v for v in values if np.isfinite(v)]
    return float(np.max(values)) if values else np.nan


def material_descriptors(formula: str) -> dict:
    """
    Calculate composition and periodic/electronic descriptors.

    The descriptors are derived from element-level JARVIS Specie
    properties and do not use calculated band-gap values.
    """
    composition = parse_formula(formula)
    elements = list(composition)

    if not elements:
        return {}

    amounts = np.array([composition[e] for e in elements], dtype=float)
    total_atoms = float(amounts.sum())

    atomic_numbers = []
    atomic_masses = []
    atomic_radii = []
    electronegativities = []
    ionization_energies = []
    electron_affinities = []
    s_valence = []
    p_valence = []
    d_valence = []
    f_valence = []
    periods = []
    groups = []

    for element in elements:
        specie = Specie(element)

        atomic_numbers.append(_valid_property(specie.Z))
        atomic_masses.append(_valid_property(specie.atomic_mass))
        atomic_radii.append(_valid_property(specie.atomic_rad))
        electronegativities.append(
            _valid_property(specie.electronegativity)
        )
        ionization_energies.append(
            _valid_property(specie.ionization_energy)
        )
        electron_affinities.append(
            _valid_property(specie.electron_affinity)
        )
        s_valence.append(_valid_property(specie.s_valence))
        p_valence.append(_valid_property(specie.p_valence))
        d_valence.append(_valid_property(specie.d_valence))
        f_valence.append(_valid_property(specie.f_valence))
        periods.append(_valid_property(specie.period))
        groups.append(_valid_property(specie.group))

    return {
        "num_elements": len(elements),
        "total_atoms": total_atoms,
        "mean_atomic_number": _safe_mean(atomic_numbers),
        "min_atomic_number": _safe_min(atomic_numbers),
        "max_atomic_number": _safe_max(atomic_numbers),
        "mean_atomic_mass": _safe_mean(atomic_masses),
        "min_atomic_mass": _safe_min(atomic_masses),
        "max_atomic_mass": _safe_max(atomic_masses),
        "mean_atomic_radius": _safe_mean(atomic_radii),
        "min_atomic_radius": _safe_min(atomic_radii),
        "max_atomic_radius": _safe_max(atomic_radii),
        "mean_electronegativity": _safe_mean(electronegativities),
        "min_electronegativity": _safe_min(electronegativities),
        "max_electronegativity": _safe_max(electronegativities),
        "electronegativity_difference": (
            _safe_max(electronegativities)
            - _safe_min(electronegativities)
            if np.isfinite(_safe_max(electronegativities))
            and np.isfinite(_safe_min(electronegativities))
            else np.nan
        ),
        "mean_ionization_energy": _safe_mean(ionization_energies),
        "mean_electron_affinity": _safe_mean(electron_affinities),
        "mean_s_valence": _safe_mean(s_valence),
        "mean_p_valence": _safe_mean(p_valence),
        "mean_d_valence": _safe_mean(d_valence),
        "mean_f_valence": _safe_mean(f_valence),
        "mean_period": _safe_mean(periods),
        "mean_group": _safe_mean(groups),
    }
