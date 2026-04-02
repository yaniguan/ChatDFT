"""Shared pytest fixtures for ChatDFT tests."""
import numpy as np
import pytest


@pytest.fixture
def cu111_positions():
    """4-atom Cu(111) slab positions (Angstrom)."""
    return np.array([
        [0.000, 0.000, 10.000],
        [1.277, 0.737, 10.000],
        [0.000, 1.475, 10.000],
        [1.277, 2.212, 10.000],
        [0.639, 0.369, 12.087],
        [1.916, 1.106, 12.087],
        [0.639, 1.843, 12.087],
        [1.916, 2.581, 12.087],
    ], dtype=np.float64)


@pytest.fixture
def cu111_elements():
    return ["Cu"] * 8


@pytest.fixture
def cu111_cell():
    return np.array([
        [2.555, 0.000, 0.000],
        [1.277, 2.212, 0.000],
        [0.000, 0.000, 25.000],
    ], dtype=np.float64)


@pytest.fixture
def sample_reaction_network():
    return {
        "reaction_network": [
            {"lhs": ["CO2(g)", "*", "H+", "e-"], "rhs": ["COOH*"]},
            {"lhs": ["COOH*", "H+", "e-"], "rhs": ["CO*", "H2O(g)"]},
        ],
        "intermediates": ["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        "ts_edges": [["CO2*", "COOH*"], ["COOH*", "CO*"]],
        "coads_pairs": [["CO*", "H*"]],
        "surface": "Cu(111)",
        "reactant": "CO2",
        "product": "CO",
    }


@pytest.fixture
def sample_dG_profile():
    return [0.0, 0.22, -0.15, -0.45, -1.10]


@pytest.fixture
def sample_scf_trajectory():
    """Simulated SCF convergence: exponential decay with noise."""
    rng = np.random.default_rng(42)
    n = 40
    A, lam = 1.0, 0.15
    dE = A * np.exp(-lam * np.arange(n)) + rng.normal(0, 0.001, n)
    return np.abs(dE).tolist()


@pytest.fixture
def sample_sloshing_trajectory():
    """Simulated charge sloshing: oscillatory divergence."""
    n = 50
    t = np.arange(n)
    dE = 0.1 * np.sin(0.4 * np.pi * t) * np.exp(-0.02 * t) + 0.001
    return np.abs(dE).tolist()
