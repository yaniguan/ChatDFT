# -*- coding: utf-8 -*-
# server/mechanisms/registry.py
from __future__ import annotations
from typing import Dict, List

# 统一字段：
# "intermediates": 所有可能出现的物种（包含气相与表面*物种；同相用中间体简写）
# "steps": 列表，每个元素 {"r":[...], "p":[...], "kind":"PCET|chem|des|ads|diss|coupling|Volmer|Heyrovsky|Tafel|oxidative_addition|reductive_elimination|migratory_insertion|transmetalation|β-H|photo|therm|LM|MvK|acid_base|carbenium|..."}
# "coads": 形如 [["CO*","H*"], ["CO*","OH*"]]
# "variants": 可按表面/催化剂覆写 steps/coads，例如 "Cu(111)": {"steps":[...], "coads":[...]}

REGISTRY: Dict[str, Dict] = {
    # =========================
    # CO2RR —— 甲醇两条路径
    # =========================
    "CO2RR_CO_path": {
        "family": "CO2RR",
        "domain": "electrocatalysis",
        "intermediates": ["*","H*","OH*","H2O(g)","CO2(g)","CO2*","COOH*","CO*","CHO*","H2CO*","H3CO*","CH3OH*","CH3OH(g)","CO(g)"],
        "steps": [
            {"r":["CO2*","H+","e-"], "p":["COOH*"], "kind":"PCET"},
            {"r":["COOH*","H+","e-"], "p":["CO*","H2O(g)"], "kind":"PCET"},
            {"r":["CO*","H+","e-"], "p":["CHO*"], "kind":"PCET"},
            {"r":["CHO*","H+","e-"], "p":["H2CO*"], "kind":"PCET"},
            {"r":["H2CO*","H+","e-"], "p":["H3CO*"], "kind":"PCET"},
            {"r":["H3CO*","H+","e-"], "p":["CH3OH*"], "kind":"PCET"},
            {"r":["CH3OH*"], "p":["CH3OH(g)","*"], "kind":"des"},
            {"r":["CO*"], "p":["CO(g)","*"], "kind":"des"},
        ],
        "coads": [["CO*","H*"],["CO*","OH*"]],
        "variants": {"Cu(111)": {}}
    },
    "CO2RR_HCOO_path": {
        "family": "CO2RR",
        "domain": "electrocatalysis",
        "intermediates": ["*","H*","OH*","H2O(g)","CO2(g)","CO2*","HCOO*","HCOOH*","H2CO*","H3CO*","CH3OH*","CH3OH(g)"],
        "steps": [
            {"r":["CO2*","H+","e-"], "p":["HCOO*"], "kind":"PCET"},
            {"r":["HCOO*","H+","e-"], "p":["HCOOH*"], "kind":"PCET"},
            {"r":["HCOOH*","H+","e-"], "p":["H2CO*","OH*"], "kind":"PCET"},
            {"r":["OH*","H+","e-"], "p":["H2O(g)","*"], "kind":"PCET"},
            {"r":["H2CO*","H+","e-"], "p":["H3CO*"], "kind":"PCET"},
            {"r":["H3CO*","H+","e-"], "p":["CH3OH*"], "kind":"PCET"},
            {"r":["CH3OH*"], "p":["CH3OH(g)","*"], "kind":"des"}
        ],
        "coads": [["HCOO*","H*"]],
        "variants": {}
    },

    # =========================
    # CO2RR —— 乙醇（细化）
    # =========================
    "CO2RR_to_ethanol_CO_coupling": {
        "family": "CO2RR",
        "domain": "electrocatalysis",
        "intermediates": [
            "*","H*","CO2(g)","CO2*","CO*","CHO*","OCCO*","CHOCO*","CH2CHO*","CH3CHO*","CH3CH2O*","CH3CH2OH*","CH3CH2OH(g)"
        ],
        "steps": [
            {"r":["CO2(g)","*"], "p":["CO2*"], "kind":"ads"},
            {"r":["CO2*","H+","e-"], "p":["CO*","OH*"], "kind":"PCET"},              # 简化进入 CO*
            {"r":["CO*","CO*"], "p":["OCCO*"], "kind":"coupling"},
            {"r":["OCCO*","H+","e-"], "p":["CHOCO*"], "kind":"PCET"},
            {"r":["CHOCO*","H+","e-"], "p":["CH2CHO*","OH*"], "kind":"PCET"},
            {"r":["CH2CHO*","H+","e-"], "p":["CH3CHO*"], "kind":"PCET"},             # 乙醛*路过
            {"r":["CH3CHO*","H+","e-"], "p":["CH3CH2O*"], "kind":"PCET"},
            {"r":["CH3CH2O*","H+","e-"], "p":["CH3CH2OH*"], "kind":"PCET"},
            {"r":["CH3CH2OH*"], "p":["CH3CH2OH(g)","*"], "kind":"des"}
        ],
        "coads": [["CO*","CO*"],["CO*","H*"]],
        "variants": {"Cu(100)": {}, "Cu(111)": {}}
    },

    # =========================
    # NRR（三机理）
    # =========================
    "NRR_distal": {
        "family": "NRR",
        "domain": "electrocatalysis",
        "intermediates": ["*","N2(g)","N2*","NNH*","NNH2*","NNH3*","N*","NH*","NH2*","NH3*","NH3(g)"],
        "steps": [
            {"r":["N2(g)","*"], "p":["N2*"], "kind":"ads"},
            {"r":["N2*","H+","e-"], "p":["NNH*"], "kind":"PCET"},
            {"r":["NNH*","H+","e-"], "p":["NNH2*"], "kind":"PCET"},
            {"r":["NNH2*","H+","e-"], "p":["NNH3*"], "kind":"PCET"},
            {"r":["NNH3*"], "p":["NH3(g)","N*"], "kind":"des"},
            {"r":["N*","H+","e-"], "p":["NH*"], "kind":"PCET"},
            {"r":["NH*","H+","e-"], "p":["NH2*"], "kind":"PCET"},
            {"r":["NH2*","H+","e-"], "p":["NH3*"], "kind":"PCET"},
            {"r":["NH3*"], "p":["NH3(g)","*"], "kind":"des"}
        ],
        "coads": [["N2*","H*"],["NNH*","H*"]],
        "variants": {"Pt(111)": {}}
    },
    "NRR_alternating": {
        "family": "NRR",
        "domain": "electrocatalysis",
        "intermediates": ["*","N2(g)","N2*","NNH*","NHNH*","NH2NH*","NH2NH2*","NH2*","NH3*","NH3(g)"],
        "steps": [
            {"r":["N2(g)","*"], "p":["N2*"], "kind":"ads"},
            {"r":["N2*","H+","e-"], "p":["NNH*"], "kind":"PCET"},
            {"r":["NNH*","H+","e-"], "p":["NHNH*"], "kind":"PCET"},
            {"r":["NHNH*","H+","e-"], "p":["NH2NH*"], "kind":"PCET"},
            {"r":["NH2NH*","H+","e-"], "p":["NH2NH2*"], "kind":"PCET"},
            {"r":["NH2NH2*"], "p":["NH3(g)","NH2*"], "kind":"des"},
            {"r":["NH2*","H+","e-"], "p":["NH3*"], "kind":"PCET"},
            {"r":["NH3*"], "p":["NH3(g)","*"], "kind":"des"}
        ],
        "coads": [["NNH*","H*"]],
        "variants": {}
    },
    "NRR_dissociative": {
        "family": "NRR",
        "domain": "electrocatalysis",
        "intermediates": ["*","N2(g)","N2*","N*","NH*","NH2*","NH3*","NH3(g)"],
        "steps": [
            {"r":["N2(g)","*"], "p":["N2*"], "kind":"ads"},
            {"r":["N2*"], "p":["N*","N*"], "kind":"diss"},
            {"r":["N*","H+","e-"], "p":["NH*"], "kind":"PCET"},
            {"r":["NH*","H+","e-"], "p":["NH2*"], "kind":"PCET"},
            {"r":["NH2*","H+","e-"], "p":["NH3*"], "kind":"PCET"},
            {"r":["NH3*"], "p":["NH3(g)","*"], "kind":"des"}
        ],
        "coads": [],
        "variants": {}
    },

    # =========================
    # ORR / OER / HER
    # =========================
    "ORR_4e": {
        "family": "ORR",
        "domain": "electrocatalysis",
        "intermediates": ["*","O2(g)","O2*","OOH*","O*","OH*","H2O(g)"],
        "steps": [
            {"r":["O2(g)","*"], "p":["O2*"], "kind":"ads"},
            {"r":["O2*","H+","e-"], "p":["OOH*"], "kind":"PCET"},
            {"r":["OOH*","H+","e-"], "p":["O*","H2O(g)"], "kind":"PCET"},
            {"r":["O*","H+","e-"], "p":["OH*"], "kind":"PCET"},
            {"r":["OH*","H+","e-"], "p":["H2O(g)","*"], "kind":"PCET"}
        ],
        "coads": [["O*","OH*"]],
        "variants": {"Pt/C": {}}
    },
    "OER_lattice_oxo_skeleton": {
        "family": "OER",
        "domain": "electrocatalysis",
        "intermediates": ["*","OH*","O*","OOH*","O2(g)"],
        "steps": [
            {"r":["*","H2O"], "p":["OH*","H+","e-"], "kind":"PCET"},
            {"r":["OH*","H+","e-"], "p":["O*"], "kind":"PCET"},
            {"r":["O*","H2O","H+","e-"], "p":["OOH*"], "kind":"PCET"},
            {"r":["OOH*","H+","e-"], "p":["O2(g)","*"], "kind":"PCET"}
        ],
        "coads": [["O*","OH*"]],
        "variants": {}
    },
    "HER_VHT": {
        "family": "HER",
        "domain": "electrocatalysis",
        "intermediates": ["*","H*","H2(g)"],
        "steps": [
            {"r":["H+","e-","*"], "p":["H*"], "kind":"Volmer"},
            {"r":["H*","H+","e-"], "p":["H2(g)","*"], "kind":"Heyrovsky"},
            {"r":["H*","H*"], "p":["H2(g)","*","*"], "kind":"Tafel"}
        ],
        "coads": [["H*","H*"]],
        "variants": {"Pt(111)": {}, "Ru(0001)": {}}
    },

    # =========================
    # NO3RR（骨架）
    # =========================
    "NO3RR_to_NH3_skeleton": {
        "family": "NO3RR",
        "domain": "electrocatalysis",
        "intermediates": ["*","NO3-","NO3*","NO2*","NO*","N*","NH*","NH2*","NH3*","NH3(g)"],
        "steps": [
            {"r":["NO3-","*"], "p":["NO3*"], "kind":"ads"},
            {"r":["NO3*","H+","e-"], "p":["NO2*","OH*"], "kind":"PCET"},
            {"r":["NO2*","H+","e-"], "p":["NO*","H2O"], "kind":"PCET"},
            {"r":["NO*","H+","e-"], "p":["N*","OH*"], "kind":"PCET"},
            {"r":["N*","H+","e-"], "p":["NH*"], "kind":"PCET"},
            {"r":["NH*","H+","e-"], "p":["NH2*"], "kind":"PCET"},
            {"r":["NH2*","H+","e-"], "p":["NH3*"], "kind":"PCET"},
            {"r":["NH3*"], "p":["NH3(g)","*"], "kind":"des"}
        ],
        "coads": [["NO*","H*"],["NO*","OH*"]],
        "variants": {}
    },

    # =========================
    # MSR 甲烷蒸汽重整
    # =========================
    "MSR_basic": {
        "family": "MSR",
        "domain": "thermocatalysis",
        "intermediates": ["*","CH4*","CH3*","H*","H2O*","OH*","O*","C*","CO*","CO(g)","H2(g)"],
        "steps": [
            {"r":["CH4*"], "p":["CH3*","H*"], "kind":"chem"},
            {"r":["H2O*"], "p":["OH*","H*"], "kind":"chem"},
            {"r":["C*","O*"], "p":["CO*"], "kind":"chem"},
            {"r":["CO*"], "p":["CO(g)","*"], "kind":"des"},
            {"r":["H*","H*"], "p":["H2(g)","*"], "kind":"chem"}
        ],
        "coads": [["CO*","O*"],["CO*","OH*"],["CH3*","H*"]],
        "variants": {"Ni(111)": {}, "Rh(111)": {}}
    },

    # =========================
    # CO 氧化（两机理骨架）
    # =========================
    "CO_oxidation_LH": {
        "family": "CO_oxidation",
        "domain": "thermocatalysis",
        "intermediates": ["*","CO(g)","O2(g)","CO*","O2*","O*","CO2*","CO2(g)"],
        "steps": [
            {"r":["CO(g)","*"], "p":["CO*"], "kind":"ads"},
            {"r":["O2(g)","*"], "p":["O2*"], "kind":"ads"},
            {"r":["O2*"], "p":["O*","O*"], "kind":"diss"},
            {"r":["CO*","O*"], "p":["CO2*","*"], "kind":"chem"},
            {"r":["CO2*"], "p":["CO2(g)","*"], "kind":"des"}
        ],
        "coads": [["CO*","O*"]],
        "variants": {"Pt(111)": {}}
    },
    "CO_oxidation_MvK": {
        "family": "CO_oxidation",
        "domain": "thermocatalysis",
        "intermediates": ["*","CO(g)","CO*","CO2(g)","lattice-O"],
        "steps": [
            {"r":["CO(g)","*"], "p":["CO*"], "kind":"ads"},
            {"r":["CO*","lattice-O"], "p":["CO2(g)","*"], "kind":"MvK"},   # 消耗晶格氧
            {"r":["*","O2(g)"], "p":["lattice-O"], "kind":"LM"}           # 晶格再生（抽象表示）
        ],
        "coads": [],
        "variants": {"Au/TiO2": {}}
    },

    # =========================
    # Haber–Bosch 合成氨（表面热催化）
    # =========================
    "Haber_Bosch_Fe": {
        "family": "Haber_Bosch",
        "domain": "thermocatalysis",
        "intermediates": ["*","N2(g)","H2(g)","N2*","N*","H*","NH*","NH2*","NH3*","NH3(g)"],
        "steps": [
            {"r":["H2(g)","*","*"], "p":["H*","H*"], "kind":"diss"},
            {"r":["N2(g)","*"], "p":["N2*"], "kind":"ads"},
            {"r":["N2*"], "p":["N*","N*"], "kind":"diss"},
            {"r":["N*","H*"], "p":["NH*","*"], "kind":"chem"},
            {"r":["NH*","H*"], "p":["NH2*","*"], "kind":"chem"},
            {"r":["NH2*","H*"], "p":["NH3*","*"], "kind":"chem"},
            {"r":["NH3*"], "p":["NH3(g)","*"], "kind":"des"}
        ],
        "coads": [["N*","H*"]],
        "variants": {"Fe(111)": {}, "Ru(0001)": {}}
    },

    # =========================
    # 同相有机金属循环（骨架）
    # =========================
    # Wilkinson 催化剂：烯烃加氢
    "Wilkinson_hydrogenation": {
        "family": "alkene_hydrogenation",
        "domain": "homogeneous_thermo",
        "intermediates": ["RhCl(PPh3)3","RhH2Cl(PPh3)2","Alkene•Rh","Alkyl•Rh","Alkane","H2"],
        "steps": [
            {"r":["RhCl(PPh3)3","H2"], "p":["RhH2Cl(PPh3)2","PPh3"], "kind":"oxidative_addition"},
            {"r":["RhH2Cl(PPh3)2","Alkene"], "p":["Alkene•Rh"], "kind":"association"},
            {"r":["Alkene•Rh"], "p":["Alkyl•Rh"], "kind":"migratory_insertion"},
            {"r":["Alkyl•Rh"], "p":["Alkane","RhCl(PPh3)2"], "kind":"reductive_elimination"},
            {"r":["RhCl(PPh3)2","PPh3"], "p":["RhCl(PPh3)3"], "kind":"association"}
        ],
        "coads": [],
        "variants": {}
    },

    # 氢甲酰化（Rh/Co）
    "Hydroformylation_Rh": {
        "family": "hydroformylation",
        "domain": "homogeneous_thermo",
        "intermediates": ["HRh(CO)2(PPh3)2","Alkene•Rh","Alkyl•Rh(CO)","Acyl•Rh","Aldehyde","H2","CO"],
        "steps": [
            {"r":["HRh(CO)2(PPh3)2","Alkene"], "p":["Alkene•Rh"], "kind":"association"},
            {"r":["Alkene•Rh","CO"], "p":["Alkyl•Rh(CO)"], "kind":"migratory_insertion"},  # 表示烯烃插入/再 CO 插入
            {"r":["Alkyl•Rh(CO)"], "p":["Acyl•Rh"], "kind":"migratory_insertion"},
            {"r":["Acyl•Rh","H2"], "p":["Aldehyde","HRh(CO)2(PPh3)2"], "kind":"reductive_elimination"}
        ],
        "coads": [],
        "variants": {"HCo(CO)4": {}}
    },

    # Heck 偶联（Pd）
    "Heck_Pd": {
        "family": "Heck",
        "domain": "homogeneous_thermo",
        "intermediates": ["Pd(0)Ln","Ar–X","Ar–Pd(II)–X","Alkene•Pd","Ar–alkyl–Pd","Product","HX"],
        "steps": [
            {"r":["Pd(0)Ln","Ar–X"], "p":["Ar–Pd(II)–X"], "kind":"oxidative_addition"},
            {"r":["Ar–Pd(II)–X","Alkene"], "p":["Alkene•Pd"], "kind":"association"},
            {"r":["Alkene•Pd"], "p":["Ar–alkyl–Pd"], "kind":"migratory_insertion"},
            {"r":["Ar–alkyl–Pd"], "p":["Product","Pd(0)Ln","HX"], "kind":"β-H"}
        ],
        "coads": [],
        "variants": {}
    },

    # Suzuki 偶联（Pd）
    "Suzuki_Pd": {
        "family": "Suzuki",
        "domain": "homogeneous_thermo",
        "intermediates": ["Pd(0)Ln","Ar–X","Ar–Pd(II)–X","Ar'–B(OH)2","Ar–Pd(II)–Ar'","Product"],
        "steps": [
            {"r":["Pd(0)Ln","Ar–X"], "p":["Ar–Pd(II)–X"], "kind":"oxidative_addition"},
            {"r":["Ar–Pd(II)–X","Ar'–B(OH)2","OH-"], "p":["Ar–Pd(II)–Ar'","X-","B(OH)3"], "kind":"transmetalation"},
            {"r":["Ar–Pd(II)–Ar'"], "p":["Product","Pd(0)Ln"], "kind":"reductive_elimination"}
        ],
        "coads": [],
        "variants": {}
    },

    # Sonogashira 偶联（Pd/Cu）
    "Sonogashira_Pd_Cu": {
        "family": "Sonogashira",
        "domain": "homogeneous_thermo",
        "intermediates": ["Pd(0)Ln","Ar–X","Ar–Pd(II)–X","Cu–alkynyl","Ar–Pd(II)–alkynyl","Product"],
        "steps": [
            {"r":["Pd(0)Ln","Ar–X"], "p":["Ar–Pd(II)–X"], "kind":"oxidative_addition"},
            {"r":["Terminal_alkyne","Base","Cu(I)"], "p":["Cu–alkynyl"], "kind":"acid_base"},
            {"r":["Ar–Pd(II)–X","Cu–alkynyl"], "p":["Ar–Pd(II)–alkynyl","CuX"], "kind":"transmetalation"},
            {"r":["Ar–Pd(II)–alkynyl"], "p":["Product","Pd(0)Ln"], "kind":"reductive_elimination"}
        ],
        "coads": [],
        "variants": {}
    },

    # 烯烃环氧化（Sharpless / Jacobsen 骨架）
    "Epoxidation_Sharpless": {
        "family": "epoxidation",
        "domain": "homogeneous_thermo",
        "intermediates": ["Ti(tartrate)","Allylic_alcohol","TBHP","Ti–OOtBu","Epoxide","tBuOH"],
        "steps": [
            {"r":["Ti(tartrate)","TBHP"], "p":["Ti–OOtBu","tBuO-"], "kind":"association"},
            {"r":["Ti–OOtBu","Allylic_alcohol"], "p":["Epoxide","Ti(tartrate)","tBuOH"], "kind":"oxygen_transfer"}
        ],
        "coads": [],
        "variants": {"Jacobsen_Mn": {}}
    },

    # 不对称氢化（Noyori/Knowles 骨架）
    "Asymmetric_Hydrogenation_Noyori": {
        "family": "asymmetric_hydrogenation",
        "domain": "homogeneous_thermo",
        "intermediates": ["Ru(BINAP)(diamine)H2","Prochiral_substrate","Hydrogenated_product"],
        "steps": [
            {"r":["Ru(BINAP)(diamine)H2","Prochiral_substrate"], "p":["Hydrogenated_product","Ru(BINAP)(diamine)"], "kind":"outer_sphere_HT"},
            {"r":["Ru(BINAP)(diamine)","H2"], "p":["Ru(BINAP)(diamine)H2"], "kind":"association"}
        ],
        "coads": [],
        "variants": {"Knowles_Rh": {}}
    },

    # =========================
    # 酸/分子筛催化：异构化、烷基化、脱水
    # =========================
    "Hydroisomerization_zeolite": {
        "family": "isomerization",
        "domain": "thermocatalysis",
        "intermediates": ["Brønsted_site(H+)","Alkane","Carbenium","Isomer","H2"],
        "steps": [
            {"r":["Alkane","Brønsted_site(H+)"], "p":["Carbenium"], "kind":"carbenium"},
            {"r":["Carbenium"], "p":["Carbenium"], "kind":"hydride_shift"},
            {"r":["Carbenium"], "p":["Isomer","Brønsted_site(H+)"], "kind":"deprotonation"}
        ],
        "coads": [],
        "variants": {"HZSM-5": {}, "SAPO-34": {}}
    },
    "Alkylation_acid": {
        "family": "alkylation",
        "domain": "thermocatalysis",
        "intermediates": ["Alkene_or_R+","Aromatic","σ-complex","Alkylated_product"],
        "steps": [
            {"r":["Alkene_or_R+","Aromatic"], "p":["σ-complex"], "kind":"electrophilic_substitution"},
            {"r":["σ-complex"], "p":["Alkylated_product"], "kind":"deprotonation"}
        ],
        "coads": [],
        "variants": {"HF/AlCl3": {}, "solid_acid": {}}
    },
    "Alcohol_dehydration": {
        "family": "dehydration_to_olefin",
        "domain": "thermocatalysis",
        "intermediates": ["Alcohol","Protonated_alcohol","Carbocation","Olefin","H2O"],
        "steps": [
            {"r":["Alcohol","H+"], "p":["Protonated_alcohol"], "kind":"acid_base"},
            {"r":["Protonated_alcohol"], "p":["Carbocation","H2O"], "kind":"E1"},
            {"r":["Carbocation"], "p":["Olefin","H+"], "kind":"deprotonation"}
        ],
        "coads": [],
        "variants": {"γ-Al2O3": {}, "phosphates": {}}
    },

    # =========================
    # 光/光热催化（骨架）
    # =========================
    "Photocatalytic_water_splitting": {
        "family": "photocatalysis",
        "domain": "photocatalysis",
        "intermediates": ["e-","h+","H+","*","H*","H2(g)","O2(g)"],
        "steps": [
            {"r":["hv"], "p":["e-","h+"], "kind":"photo"},
            {"r":["h+","H2O"], "p":["O2(g)","H+"], "kind":"photo"},
            {"r":["e-","H+"], "p":["H*"], "kind":"photo"},
            {"r":["H*","H*"], "p":["H2(g)"], "kind":"chem"}
        ],
        "coads": [],
        "variants": {"TiO2": {}, "g-C3N4": {}}
    },
    "Photothermal_CO2RR_skeleton": {
        "family": "CO2RR",
        "domain": "photothermal",
        "intermediates": ["hv","heat","CO2(g)","*","CO2*","CO*","CHO*","CHxOy*","Products"],
        "steps": [
            {"r":["hv"], "p":["e-","h+"], "kind":"photo"},
            {"r":["heat"], "p":["*"], "kind":"therm"},
            {"r":["CO2(g)","*"], "p":["CO2*"], "kind":"ads"},
            {"r":["CO2*","e-","h+"], "p":["CO*","O*"], "kind":"photo"},
            {"r":["CO*","e-","h+"], "p":["CHO*"], "kind":"photo"},
            {"r":["CHO*","heat"], "p":["CHxOy*"], "kind":"therm"},
            {"r":["CHxOy*"], "p":["Products"], "kind":"des"}
        ],
        "coads": [["CO*","O*"]],
        "variants": {}
    },
    "Photothermal_methane_conversion": {
        "family": "photothermal_CH4_conversion",
        "domain": "photothermal",
        "intermediates": ["hv","heat","CH4","*","CH3*","H*","C*","CxHy","H2(g)"],
        "steps": [
            {"r":["hv"], "p":["e-","h+"], "kind":"photo"},
            {"r":["CH4","*","heat"], "p":["CH3*","H*"], "kind":"therm"},
            {"r":["H*","H*"], "p":["H2(g)"], "kind":"chem"},
            {"r":["CH3*"], "p":["CxHy"], "kind":"chem"}
        ],
        "coads": [],
        "variants": {}
    },
}