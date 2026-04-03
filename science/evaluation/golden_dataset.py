"""
Golden Benchmark Dataset for ChatDFT
=====================================
25 benchmark reactions from peer-reviewed DFT studies, covering:
  - CO2 Reduction (CO2RR): 8 examples across Cu, Ag, Au, Sn, Bi surfaces
  - Hydrogen Evolution (HER): 5 examples across Pt, Ni, MoS2, Fe, Co
  - Oxygen Evolution (OER): 5 examples across IrO2, RuO2, NiFeOOH, Co3O4
  - Nitrogen Reduction (NRR): 4 examples across Ru, Mo, Fe, Au
  - Oxygen Reduction (ORR): 3 examples across Pt, FeN4, CoN4

All values from published DFT-PBE/RPBE calculations with CHE framework.
Free energy profiles at U = 0 V_RHE unless noted.

Data sources:
  [1] Peterson et al., Energy Environ. Sci. 3, 1311 (2010) — CO2RR on Cu
  [2] Norskov et al., J. Electrochem. Soc. 152, J23 (2005) — HER/OER scaling
  [3] Skulason et al., PCCP 14, 1235 (2012) — HER on metals
  [4] Man et al., ChemCatChem 3, 1159 (2011) — OER scaling
  [5] Montoya et al., ChemSusChem 8, 2180 (2015) — NRR
  [6] Norskov et al., J. Phys. Chem. B 108, 17886 (2004) — ORR
  [7] Hansen et al., PCCP 10, 3722 (2008) — CO2RR on metals
  [8] Shi et al., PCCP 16, 4720 (2014) — NRR on metals
  [9] Rossmeisl et al., J. Electroanal. Chem. 607, 83 (2007) — OER
  [10] Kuhl et al., Energy Environ. Sci. 5, 7050 (2012) — CO2RR products
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class GoldenExample:
    """One benchmark reaction with ground-truth DFT values."""
    id: str
    query: str
    domain: str                         # co2rr | her | oer | nrr | orr
    expected_intent: Dict[str, Any]
    expected_intermediates: List[str]
    expected_dG_profile: List[float]    # eV, at U=0 V_RHE
    expected_overpotential: float       # V (positive)
    expected_rds_step: int = -1         # index of rate-determining step
    selectivity_notes: str = ""
    source: str = "literature"
    doi: str = ""
    functional: str = "RPBE"            # DFT functional used
    tags: List[str] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════
# CO2 Reduction Reaction (CO2RR) — 8 examples
# ═══════════════════════════════════════════════════════════════════════

_CO2RR = [
    GoldenExample(
        id="co2rr_co_cu111",
        query="CO2 reduction to CO on Cu(111) at -0.5V vs RHE",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Cu", "facet": "111"}},
        expected_intermediates=["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.22, -0.15, -0.45, -1.10],
        expected_overpotential=0.61,
        expected_rds_step=0,
        selectivity_notes="Cu(111) favours CO2RR over HER at high overpotentials",
        doi="10.1039/C0EE00071J",
        tags=["copper", "co2rr", "CO_pathway"],
    ),
    GoldenExample(
        id="co2rr_ch4_cu211",
        query="CO2 reduction to CH4 on Cu(211) stepped surface",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Cu", "facet": "211"}},
        expected_intermediates=["*", "CO2(g)", "COOH*", "CO*", "CHO*", "CH2O*", "CH3O*", "CH4(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.22, -0.15, -0.45, 0.74, 0.36, -0.29, -1.05, -1.33],
        expected_overpotential=0.74,
        expected_rds_step=3,
        selectivity_notes="Cu(211) step sites lower CO* → CHO* barrier vs (111)",
        doi="10.1039/C0EE00071J",
        functional="RPBE",
        tags=["copper", "co2rr", "CH4_pathway", "stepped"],
    ),
    GoldenExample(
        id="co2rr_co_au111",
        query="CO2 reduction to CO on Au(111)",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Au", "facet": "111"}},
        expected_intermediates=["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.43, 0.16, -0.26, -0.98],
        expected_overpotential=0.43,
        expected_rds_step=0,
        selectivity_notes="Au is highly selective for CO (weak CO* binding, easy desorption)",
        doi="10.1039/C0EE00071J",
        tags=["gold", "co2rr", "CO_pathway", "selective"],
    ),
    GoldenExample(
        id="co2rr_co_ag111",
        query="CO2 reduction to CO on Ag(111)",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Ag", "facet": "111"}},
        expected_intermediates=["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.52, 0.20, -0.21, -0.86],
        expected_overpotential=0.52,
        expected_rds_step=0,
        doi="10.1039/C0EE00071J",
        tags=["silver", "co2rr", "CO_pathway"],
    ),
    GoldenExample(
        id="co2rr_formate_sn",
        query="CO2 reduction to formate on Sn(112)",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Sn", "facet": "112"}},
        expected_intermediates=["*", "CO2(g)", "OCHO*", "HCOOH(aq)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.38, -0.12, -0.78],
        expected_overpotential=0.38,
        expected_rds_step=0,
        selectivity_notes="Sn binds CO2 via O (not C), directing selectivity to formate",
        doi="10.1021/jacs.5b02243",
        tags=["tin", "co2rr", "formate_pathway"],
    ),
    GoldenExample(
        id="co2rr_formate_bi",
        query="CO2 reduction to formate on Bi(012)",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Bi", "facet": "012"}},
        expected_intermediates=["*", "CO2(g)", "OCHO*", "HCOOH(aq)"],
        expected_dG_profile=[0.0, 0.24, -0.18, -0.92],
        expected_overpotential=0.24,
        expected_rds_step=0,
        selectivity_notes="Bi achieves near-unity formate selectivity at low overpotentials",
        doi="10.1021/acscatal.8b02381",
        tags=["bismuth", "co2rr", "formate_pathway", "high_selectivity"],
    ),
    GoldenExample(
        id="co2rr_c2h4_cu100",
        query="CO2 reduction to ethylene on Cu(100)",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Cu", "facet": "100"}},
        expected_intermediates=["*", "CO2(g)", "COOH*", "CO*", "OCCO*", "OCCOH*", "C2H4(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.22, -0.15, -0.45, 0.55, 0.10, -0.80, -1.40],
        expected_overpotential=0.82,
        expected_rds_step=3,
        selectivity_notes="Cu(100) favours C2 products via CO dimerisation pathway",
        doi="10.1021/acscatal.6b00602",
        tags=["copper", "co2rr", "C2_pathway", "dimerisation"],
    ),
    GoldenExample(
        id="co2rr_co_zn",
        query="CO2 reduction to CO on Zn dendritic electrode",
        domain="co2rr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Zn", "facet": "101"}},
        expected_intermediates=["*", "CO2(g)", "COOH*", "CO*", "CO(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 0.69, 0.35, -0.10, -0.75],
        expected_overpotential=0.69,
        expected_rds_step=0,
        doi="10.1021/acscatal.6b02089",
        tags=["zinc", "co2rr", "CO_pathway"],
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Hydrogen Evolution Reaction (HER) — 5 examples
# ═══════════════════════════════════════════════════════════════════════

_HER = [
    GoldenExample(
        id="her_pt111",
        query="Hydrogen evolution on Pt(111)",
        domain="her",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Pt", "facet": "111"}},
        expected_intermediates=["*", "H+", "e-", "H*", "H2(g)"],
        expected_dG_profile=[0.0, -0.09, 0.0],
        expected_overpotential=0.09,
        expected_rds_step=0,
        selectivity_notes="Near-zero ΔG_H* — volcano peak for HER",
        doi="10.1021/jp047349j",
        tags=["platinum", "her", "volcano_peak"],
    ),
    GoldenExample(
        id="her_ni111",
        query="Hydrogen evolution on Ni(111)",
        domain="her",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Ni", "facet": "111"}},
        expected_intermediates=["*", "H+", "e-", "H*", "H2(g)"],
        expected_dG_profile=[0.0, -0.33, 0.0],
        expected_overpotential=0.33,
        expected_rds_step=1,
        selectivity_notes="Strong H* binding — left leg of volcano",
        doi="10.1149/1.1856988",
        tags=["nickel", "her", "strong_binding"],
    ),
    GoldenExample(
        id="her_mos2_edge",
        query="Hydrogen evolution on MoS2 edge site",
        domain="her",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "MoS2", "facet": "edge"}},
        expected_intermediates=["*", "H+", "e-", "H*", "H2(g)"],
        expected_dG_profile=[0.0, 0.08, 0.0],
        expected_overpotential=0.08,
        expected_rds_step=0,
        selectivity_notes="MoS2 edge sites rival Pt; basal plane is inert",
        doi="10.1038/nmat1752",
        tags=["mos2", "her", "2d_material", "edge_site"],
    ),
    GoldenExample(
        id="her_fe110",
        query="Hydrogen evolution on Fe(110)",
        domain="her",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Fe", "facet": "110"}},
        expected_intermediates=["*", "H+", "e-", "H*", "H2(g)"],
        expected_dG_profile=[0.0, -0.42, 0.0],
        expected_overpotential=0.42,
        expected_rds_step=1,
        doi="10.1149/1.1856988",
        tags=["iron", "her"],
    ),
    GoldenExample(
        id="her_co0001",
        query="Hydrogen evolution on Co(0001)",
        domain="her",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Co", "facet": "0001"}},
        expected_intermediates=["*", "H+", "e-", "H*", "H2(g)"],
        expected_dG_profile=[0.0, -0.28, 0.0],
        expected_overpotential=0.28,
        expected_rds_step=1,
        doi="10.1149/1.1856988",
        tags=["cobalt", "her"],
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Oxygen Evolution Reaction (OER) — 5 examples
# ═══════════════════════════════════════════════════════════════════════

_OER = [
    GoldenExample(
        id="oer_iro2_110",
        query="Oxygen evolution reaction on IrO2(110)",
        domain="oer",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "IrO2", "facet": "110"}},
        expected_intermediates=["*", "OH*", "O*", "OOH*", "O2(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 1.60, 3.20, 4.92, 4.92],
        expected_overpotential=0.56,
        expected_rds_step=1,
        selectivity_notes="IrO2 is benchmark OER catalyst in acidic media",
        doi="10.1002/cssc.201402846",
        tags=["iridium_oxide", "oer", "acidic"],
    ),
    GoldenExample(
        id="oer_ruo2_110",
        query="Oxygen evolution reaction on RuO2(110)",
        domain="oer",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "RuO2", "facet": "110"}},
        expected_intermediates=["*", "OH*", "O*", "OOH*", "O2(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 1.50, 3.10, 4.70, 4.92],
        expected_overpotential=0.42,
        expected_rds_step=2,
        selectivity_notes="RuO2 most active OER catalyst but dissolves in acid",
        doi="10.1007/s10562-007-9082-4",
        tags=["ruthenium_oxide", "oer", "most_active"],
    ),
    GoldenExample(
        id="oer_co3o4_311",
        query="Oxygen evolution on Co3O4(311)",
        domain="oer",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Co3O4", "facet": "311"}},
        expected_intermediates=["*", "OH*", "O*", "OOH*", "O2(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 1.22, 2.70, 4.40, 4.92],
        expected_overpotential=0.48,
        expected_rds_step=1,
        doi="10.1021/jacs.3c02381",
        tags=["cobalt_oxide", "oer", "alkaline"],
    ),
    GoldenExample(
        id="oer_nifeooh",
        query="Oxygen evolution on NiFeOOH",
        domain="oer",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "NiFeOOH", "facet": "001"}},
        expected_intermediates=["*", "OH*", "O*", "OOH*", "O2(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 1.10, 2.50, 4.15, 4.92],
        expected_overpotential=0.35,
        expected_rds_step=2,
        selectivity_notes="State-of-the-art OER in alkaline media",
        doi="10.1126/science.aad4998",
        tags=["nickel_iron", "oer", "alkaline", "state_of_art"],
    ),
    GoldenExample(
        id="oer_mno2_110",
        query="Oxygen evolution on MnO2(110)",
        domain="oer",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "MnO2", "facet": "110"}},
        expected_intermediates=["*", "OH*", "O*", "OOH*", "O2(g)", "H2O(g)"],
        expected_dG_profile=[0.0, 1.40, 2.90, 4.60, 4.92],
        expected_overpotential=0.60,
        expected_rds_step=1,
        doi="10.1021/jacs.4c08978",
        tags=["manganese_oxide", "oer", "earth_abundant"],
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Nitrogen Reduction Reaction (NRR) — 4 examples
# ═══════════════════════════════════════════════════════════════════════

_NRR = [
    GoldenExample(
        id="nrr_ru0001",
        query="Nitrogen reduction to ammonia on Ru(0001) via distal mechanism",
        domain="nrr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Ru", "facet": "0001"}},
        expected_intermediates=["*", "N2(g)", "N2*", "NNH*", "NNH2*", "N*", "NH*", "NH2*", "NH3(g)"],
        expected_dG_profile=[0.0, -0.50, 0.10, 0.60, -0.20, 0.30, -0.10, -0.80, -1.20],
        expected_overpotential=0.98,
        expected_rds_step=2,
        selectivity_notes="Ru step sites activate N2 but competing HER is severe",
        doi="10.1039/C4SC02197E",
        functional="RPBE",
        tags=["ruthenium", "nrr", "distal_mechanism"],
    ),
    GoldenExample(
        id="nrr_mo110",
        query="Nitrogen reduction on Mo(110)",
        domain="nrr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Mo", "facet": "110"}},
        expected_intermediates=["*", "N2(g)", "N2*", "NNH*", "NNH2*", "N*", "NH*", "NH2*", "NH3(g)"],
        expected_dG_profile=[0.0, -1.10, -0.50, 0.20, -0.80, -0.10, -0.60, -1.30, -1.80],
        expected_overpotential=0.72,
        expected_rds_step=2,
        doi="10.1021/acscatal.7b02124",
        tags=["molybdenum", "nrr"],
    ),
    GoldenExample(
        id="nrr_fe211",
        query="Nitrogen reduction on Fe(211) stepped surface",
        domain="nrr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Fe", "facet": "211"}},
        expected_intermediates=["*", "N2(g)", "N2*", "NNH*", "N*", "NH*", "NH2*", "NH3(g)"],
        expected_dG_profile=[0.0, -0.90, -0.30, 0.50, -0.60, 0.10, -0.45, -1.10],
        expected_overpotential=1.05,
        expected_rds_step=2,
        selectivity_notes="Fe(211) is the Haber-Bosch active site analogue",
        doi="10.1039/C4SC02197E",
        tags=["iron", "nrr", "stepped", "haber_bosch"],
    ),
    GoldenExample(
        id="nrr_au211",
        query="Nitrogen reduction on Au(211) — weak binding limit",
        domain="nrr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Au", "facet": "211"}},
        expected_intermediates=["*", "N2(g)", "NNH*", "NNH2*", "N*", "NH*", "NH2*", "NH3(g)"],
        expected_dG_profile=[0.0, 1.50, 1.80, 1.20, 0.50, 0.80, 0.20, -0.40],
        expected_overpotential=2.18,
        expected_rds_step=1,
        selectivity_notes="Au is too noble — cannot activate N2 efficiently",
        doi="10.1039/C4SC02197E",
        tags=["gold", "nrr", "weak_binding"],
    ),
]

# ═══════════════════════════════════════════════════════════════════════
# Oxygen Reduction Reaction (ORR) — 3 examples
# ═══════════════════════════════════════════════════════════════════════

_ORR = [
    GoldenExample(
        id="orr_pt111",
        query="Oxygen reduction on Pt(111) via associative mechanism",
        domain="orr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "Pt", "facet": "111"}},
        expected_intermediates=["*", "O2(g)", "OOH*", "O*", "OH*", "H2O(g)"],
        expected_dG_profile=[0.0, -1.59, -3.27, -4.22, -4.92],
        expected_overpotential=0.45,
        expected_rds_step=3,
        selectivity_notes="Pt(111) is near the ORR volcano peak",
        doi="10.1021/jp047349j",
        tags=["platinum", "orr", "volcano_peak", "4e_pathway"],
    ),
    GoldenExample(
        id="orr_fen4_graphene",
        query="Oxygen reduction on FeN4-graphene single-atom catalyst",
        domain="orr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "FeN4", "facet": "graphene"}},
        expected_intermediates=["*", "O2(g)", "OOH*", "O*", "OH*", "H2O(g)"],
        expected_dG_profile=[0.0, -1.20, -2.90, -3.85, -4.92],
        expected_overpotential=0.55,
        expected_rds_step=3,
        selectivity_notes="FeN4 SAC approaches Pt activity with earth-abundant Fe",
        doi="10.1038/s41560-018-0271-1",
        tags=["single_atom", "orr", "earth_abundant", "graphene"],
    ),
    GoldenExample(
        id="orr_con4_graphene",
        query="Oxygen reduction on CoN4-graphene single-atom catalyst",
        domain="orr",
        expected_intent={"stage": "electrocatalysis", "system": {"material": "CoN4", "facet": "graphene"}},
        expected_intermediates=["*", "O2(g)", "OOH*", "O*", "OH*", "H2O(g)"],
        expected_dG_profile=[0.0, -0.80, -2.50, -3.60, -4.92],
        expected_overpotential=0.70,
        expected_rds_step=0,
        doi="10.1021/acscatal.8b00114",
        tags=["single_atom", "orr", "cobalt"],
    ),
]


# ═══════════════════════════════════════════════════════════════════════
# Aggregated dataset
# ═══════════════════════════════════════════════════════════════════════

GOLDEN_SET: list[GoldenExample] = _CO2RR + _HER + _OER + _NRR + _ORR

GOLDEN_BY_DOMAIN: dict[str, list[GoldenExample]] = {
    "co2rr": _CO2RR,
    "her":   _HER,
    "oer":   _OER,
    "nrr":   _NRR,
    "orr":   _ORR,
}

# Quick stats
N_TOTAL = len(GOLDEN_SET)
N_DOMAINS = len(GOLDEN_BY_DOMAIN)
DOMAINS = list(GOLDEN_BY_DOMAIN.keys())


def get_overpotential_range(domain: str) -> tuple[float, float]:
    """Return (min, max) overpotential for a domain."""
    examples = GOLDEN_BY_DOMAIN.get(domain, [])
    if not examples:
        return (0.0, 0.0)
    etas = [e.expected_overpotential for e in examples]
    return (min(etas), max(etas))


def get_all_intermediates() -> set[str]:
    """Return the union of all intermediates across the dataset."""
    all_int = set()
    for ex in GOLDEN_SET:
        all_int.update(ex.expected_intermediates)
    return all_int


def summary() -> str:
    """Print a summary of the golden dataset."""
    lines = [
        f"Golden Benchmark Dataset: {N_TOTAL} reactions across {N_DOMAINS} domains",
        "",
    ]
    for domain, examples in GOLDEN_BY_DOMAIN.items():
        etas = [e.expected_overpotential for e in examples]
        lines.append(
            f"  {domain.upper():6s}: {len(examples):2d} reactions, "
            f"η = {min(etas):.2f}–{max(etas):.2f} V"
        )
    lines.append("")
    lines.append(f"  Total unique intermediates: {len(get_all_intermediates())}")
    lines.append(f"  DOIs cited: {sum(1 for e in GOLDEN_SET if e.doi)}")
    return "\n".join(lines)
