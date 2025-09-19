# server/chat/intent_mechanisms.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional

try:
    from server.mechanisms.registry import REGISTRY
except Exception:
    REGISTRY = {}

# electrocatalysis/thermo/homogeneous/photo aliases
MECH_ALIASES = [
    (r"\bco2rr\b|\bco2\b.*\breduc", ["CO2RR_CO_path","CO2RR_HCOO_path","CO2RR_to_ethanol_CO_coupling"]),
    (r"\bnrr\b|\bn2\b.*\breduc",    ["NRR_distal","NRR_alternating","NRR_dissociative"]),
    (r"\borr\b|\boxygen\s+reduction", ["ORR_4e"]),
    (r"\boer\b|\boxygen\s+evolution", ["OER_lattice_oxo_skeleton"]),
    (r"\bher\b|\bhydrogen\s+evolution", ["HER_VHT"]),
    (r"\bno3rr\b|\bnitrate\b.*\breduc", ["NO3RR_to_NH3_skeleton"]),
    (r"\bmsr\b|\bmethane\s+steam\s+reform", ["MSR_basic"]),
    (r"\bhaber\b|\bnh3\b.*\bsynth", ["Haber_Bosch_Fe"]),
    (r"\bco\s+oxidation\b", ["CO_oxidation_LH","CO_oxidation_MvK"]),
    (r"\bisomeriz", ["Hydroisomerization_zeolite"]),
    (r"\balkylation\b", ["Alkylation_acid"]),
    (r"\bdehydration\b|\bto\s+olefin\b", ["Alcohol_dehydration"]),
    (r"\bwilkinson\b|\brhcl\(pph3\)3\b|\balkene\s+hydrogenation", ["Wilkinson_hydrogenation"]),
    (r"\bhydroformylation\b|\brh\(pph3\)3cl\b|\bhco\(co\)4\b", ["Hydroformylation_Rh"]),
    (r"\bheck\b", ["Heck_Pd"]),
    (r"\bsuzuki\b", ["Suzuki_Pd"]),
    (r"\bsonogashira\b", ["Sonogashira_Pd_Cu"]),
    (r"\bepoxidation\b|\bsharpless\b|\bjacobsen\b", ["Epoxidation_Sharpless"]),
    (r"\bnoyori\b|\bknowles\b|\basymmetric\b.*\bhydrogenation", ["Asymmetric_Hydrogenation_Noyori"]),
    (r"\bphotocatalysis\b|\bphoto\s+water\s+split", ["Photocatalytic_water_splitting"]),
    (r"\bphotothermal\b.*co2", ["Photothermal_CO2RR_skeleton"]),
    (r"\bphotothermal\b.*methane|\bphotothermal\b.*ch4", ["Photothermal_methane_conversion"]),
]


def guess_mechanisms(text: str, guided: Dict[str, Any]) -> List[str]:
    tl = (text or "").lower()
    mechs: List[str] = []
    if re.search(r"\bch3oh\b|\bmethanol\b", tl):
        mechs += ["CO2RR_CO_path","CO2RR_HCOO_path"]
    if re.search(r"\bethanol\b|\bch3ch2oh\b", tl):
        mechs += ["CO2RR_to_ethanol_CO_coupling"]
    for k in (guided.get("tags") or []):
        if isinstance(k, str) and k in REGISTRY:
            mechs.append(k)
    for pat, keys in MECH_ALIASES:
        if re.search(pat, tl):
            mechs += keys
    mechs = [k for i, k in enumerate(mechs) if k in REGISTRY and k not in mechs[:i]]
    return mechs[:4]


def _apply_variant(entry: Dict[str, Any], substrate: Optional[str], facet: Optional[str]) -> Dict[str, Any]:
    vs = entry.get("variants") or {}
    if facet and facet in vs:
        return vs[facet] or {}
    if substrate and substrate in vs:
        return vs[substrate] or {}
    return {}


def expand_mechanism_network(keys: List[str], substrate: Optional[str], facet: Optional[str]) -> Dict[str, Any]:
    inters: List[Any] = []
    steps: List[Any] = []
    coads: List[Any] = []
    for k in keys:
        base = REGISTRY.get(k) or {}
        var = _apply_variant(base, substrate, facet)
        inters += list(base.get("intermediates") or []) + list(var.get("intermediates") or [])
        steps  += list(base.get("steps") or [])         + list(var.get("steps") or [])
        coads  += list(base.get("coads") or [])         + list(var.get("coads") or [])
    def _uniq(seq):
        seen = set(); out = []
        for x in seq:
            j = json.dumps(x, sort_keys=True) if isinstance(x, (dict, list)) else str(x)
            if j not in seen:
                seen.add(j); out.append(x)
        return out
    return {"intermediates": _uniq(inters), "steps": _uniq(steps), "coads_pairs": _uniq(coads)}


def family_domain_from_keys(keys: List[str]) -> tuple[Optional[str], Optional[str]]:
    fam, dom = None, None
    for k in keys:
        ent = REGISTRY.get(k) or {}
        if not fam:
            fam = ent.get("family")
        if not dom:
            dom = ent.get("domain")
    return fam, dom

