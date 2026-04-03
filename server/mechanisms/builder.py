# server/mechanisms/builder.py
# -*- coding: utf-8 -*-
"""
Dynamic Mechanism Builder — replaces the static REGISTRY dict.

Design
------
Instead of hardcoding every mechanism, we define REACTION_TYPE_TEMPLATES that
capture the *patterns* of a reaction class (what kind of intermediates to expect,
what bond-breaking/forming events matter, what DFT settings are relevant).

The builder then:
1. Checks the DB (ReactionSystem + MechanismGraph) for an existing cached result.
2. If not cached, does a literature RAG search for context.
3. Prompts the LLM with the matching template + RAG context to generate the mechanism.
4. Validates + normalises the output.
5. Stores it in the DB for future reuse.

Usage (from plan_agent or hypothesis_agent)
--------------------------------------------
from server.mechanisms.builder import MechanismBuilder

builder = MechanismBuilder()
graph = await builder.build(
    domain="electrochemical",
    surface="Pt(111)",
    reactant="C4H10",
    product="C4H8",
    conditions={"pH": 0, "potential": -0.5},
    session_id=123,
)
# graph is a MechanismResult dataclass
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)


# ===========================================================================
# Reaction-type TEMPLATES
# Each template describes the *pattern*, not the specific mechanism.
# The LLM uses this as structured guidance to generate the actual steps.
# ===========================================================================

REACTION_TYPE_TEMPLATES: Dict[str, Dict[str, Any]] = {

    # ------------------------------------------------------------------
    # ELECTROCHEMICAL
    # ------------------------------------------------------------------
    "electrochemical_reduction": {
        "keywords": ["co2rr", "co2 reduction", "n2 reduction", "nrr", "no3rr",
                     "oxygen reduction", "orr", "electrochemical reduction",
                     "h2 evolution", "her"],
        "domain": "electrochemical",
        "direction": "reduction",
        "bond_events": ["C-O cleavage", "N≡N cleavage", "protonation", "C-C coupling"],
        "key_species": ["H⁺", "e⁻", "H₂O", "OH⁻"],
        "step_kind": "PCET",
        "thermodynamics": "CHE (computational hydrogen electrode)",
        "dft_notes": "Use VASPsol or implicit solvation; include potential correction.",
        "prompt_template": """
You are a DFT/catalysis expert. Generate the elementary reaction mechanism for:
  Reaction: {reactant} → {product} on {surface}
  Domain: Electrochemical reduction
  Conditions: {conditions}

Rules:
- Each step must be elementary (single bond break or form).
- All surface species end with *.  Gas-phase species end with (g).
- Electrochemical steps use the PCET pattern: X* + H⁺ + e⁻ → XH*.
- Include adsorption (ads) and desorption (des) steps.
- List all plausible intermediates including branching paths if relevant.

Literature hints:
{rag_context}

Return strict JSON:
{{
  "name": "<short_name>",
  "family": "<CO2RR|NRR|HER|ORR|NO3RR|...>",
  "intermediates": ["A*", "B*", "C(g)", ...],
  "steps": [
    {{"r": ["A*", "H+", "e-"], "p": ["B*"], "kind": "PCET"}},
    ...
  ],
  "coads": [["A*","H*"], ...],
  "ts_candidates": ["A*→B*", ...],
  "confidence": 0.0-1.0,
  "rationale": "brief explanation"
}}
""",
    },

    "electrochemical_oxidation": {
        "keywords": ["oer", "oxygen evolution", "co oxidation", "electrochemical oxidation",
                     "chlorine evolution", "cer", "alcohol oxidation"],
        "domain": "electrochemical",
        "direction": "oxidation",
        "bond_events": ["O-H cleavage", "O-O formation", "C-H cleavage"],
        "key_species": ["H₂O", "OH⁻", "O²⁻", "H⁺", "e⁻"],
        "step_kind": "PCET",
        "thermodynamics": "CHE with overpotential analysis",
        "dft_notes": "Include Hubbard U for metal oxides; use solvation correction.",
        "prompt_template": """
You are a DFT/catalysis expert. Generate the elementary reaction mechanism for:
  Reaction: {reactant} → {product} on {surface}
  Domain: Electrochemical oxidation
  Conditions: {conditions}

Rules:
- Oxidation steps: X* → X* + H⁺ + e⁻ (reverse PCET).
- Consider Mars-van Krevelen if surface oxygen is involved.
- All surface species end with *.

Literature hints:
{rag_context}

Return strict JSON (same schema as electrochemical_reduction template).
""",
    },

    # ------------------------------------------------------------------
    # THERMAL CATALYSIS
    # ------------------------------------------------------------------
    "thermal_dehydrogenation": {
        "keywords": ["dehydrogenation", "c-h activation", "c-h bond", "alkane dehydrogenation",
                     "propane dehydrogenation", "pdh", "butane dehydrogenation",
                     "ethane dehydrogenation", "beta-h elimination", "oxidative dehydrogenation"],
        "domain": "thermal",
        "bond_events": ["C-H cleavage", "β-H elimination", "H₂ recombination", "alkene desorption"],
        "key_species": ["H*", "H₂(g)"],
        "step_kind": "chem",
        "thermodynamics": "ΔG(T,P) with ZPE + entropy corrections",
        "dft_notes": "Use vdW-DF2 or D3 dispersion for alkane adsorption. ISMEAR=1, SIGMA=0.2.",
        "prompt_template": """
You are a DFT/catalysis expert. Generate the elementary reaction mechanism for:
  Reaction: {reactant} → {product} on {surface}
  Domain: Thermal dehydrogenation
  Conditions: {conditions}

Rules:
- Include: adsorption of reactant, sequential C-H cleavage steps (each elementary),
  β-H elimination, H* recombination to H₂(g), alkene desorption.
- All surface intermediates end with *.  Gas species end with (g).
- Consider both direct C-H cleavage (oxidative addition) and β-H elimination routes.
- For each C-H cleavage: list the carbon position (α, β, γ) if relevant.
- Mark transition states (TS) between each consecutive intermediate pair.

Literature hints:
{rag_context}

Return strict JSON:
{{
  "name": "Alkane_dehydrogenation_{surface_clean}",
  "family": "dehydrogenation",
  "intermediates": ["{reactant}*", "{reactant_minus_H}*", ..., "{product}*", "{product}(g)", "H*", "H2(g)"],
  "steps": [
    {{"r": ["{reactant}(g)", "*"], "p": ["{reactant}*"], "kind": "ads"}},
    {{"r": ["{reactant}*", "*"], "p": ["{reactant_minus_H}*", "H*"], "kind": "chem"}},
    ...
    {{"r": ["H*", "H*"], "p": ["H2(g)", "*", "*"], "kind": "chem"}},
    {{"r": ["{product}*"], "p": ["{product}(g)", "*"], "kind": "des"}}
  ],
  "coads": [["{reactant_minus_H}*","H*"], ["{product}*","H*"]],
  "ts_candidates": ["{reactant}*→{reactant_minus_H}*+H*", ...],
  "confidence": 0.0-1.0,
  "rationale": "brief explanation"
}}
""",
    },

    "thermal_oxidation": {
        "keywords": ["co oxidation", "methane oxidation", "partial oxidation", "combustion",
                     "mars van krevelen", "mvk", "langmuir hinshelwood", "lh", "eley rideal"],
        "domain": "thermal",
        "bond_events": ["O₂ dissociation", "C-H cleavage", "C-O formation", "CO₂ desorption"],
        "key_species": ["O*", "O₂(g)", "OH*"],
        "step_kind": "chem",
        "dft_notes": "Check spin polarisation (O₂ is triplet). DFT+U for oxides.",
        "prompt_template": """
Generate the elementary thermal oxidation mechanism for {reactant} → {product} on {surface}.
Consider both Langmuir-Hinshelwood (both reactants adsorbed) and Mars-van Krevelen (surface O participates).
Conditions: {conditions}.
Literature: {rag_context}
Return strict JSON (same schema).
""",
    },

    "thermal_hydrogenation": {
        "keywords": ["hydrogenation", "co hydrogenation", "co2 hydrogenation", "sabatier",
                     "methanation", "methanol synthesis", "fischer tropsch", "alkene hydrogenation"],
        "domain": "thermal",
        "bond_events": ["H₂ dissociation", "C-O/C=C hydrogenation", "product desorption"],
        "key_species": ["H*", "H₂(g)"],
        "step_kind": "chem",
        "dft_notes": "Include coverage effects if H* coverage is high.",
        "prompt_template": """
Generate the elementary thermal hydrogenation mechanism for {reactant} → {product} on {surface}.
H₂ dissociative adsorption is usually the first step.
Conditions: {conditions}.  Literature: {rag_context}
Return strict JSON (same schema).
""",
    },

    "thermal_coupling": {
        "keywords": ["c-c coupling", "ethylene coupling", "dehydrogenative coupling",
                     "oxidative coupling", "methane coupling", "ocm"],
        "domain": "thermal",
        "bond_events": ["C-H cleavage", "C-C formation", "radical coupling"],
        "key_species": ["CH₃*", "C₂H₅*"],
        "step_kind": "chem",
        "prompt_template": """
Generate elementary C-C coupling mechanism for {reactant} → {product} on {surface}.
Conditions: {conditions}.  Literature: {rag_context}
Return strict JSON (same schema).
""",
    },

    # ------------------------------------------------------------------
    # PHOTOCATALYSIS
    # ------------------------------------------------------------------
    "photocatalytic": {
        "keywords": ["photocatalysis", "photocatalytic", "photoreduction", "photooxidation",
                     "water splitting", "hyd photocatalysis", "co2 photoreduction"],
        "domain": "photo",
        "bond_events": ["photoexcitation", "charge separation", "radical formation", "product release"],
        "key_species": ["h⁺ (hole)", "e⁻ (CB electron)", "•OH", "O₂⁻•"],
        "step_kind": "photo",
        "dft_notes": "Use HSE06 or GW for accurate band gaps. Include semiconductor substrate.",
        "prompt_template": """
Generate the elementary photocatalytic mechanism for {reactant} → {product} on {surface}.
Include: photoexcitation, charge transfer to surface, radical steps, product desorption.
Conditions: {conditions}.  Literature: {rag_context}
Return strict JSON (same schema).
""",
    },

    # ------------------------------------------------------------------
    # HOMOGENEOUS
    # ------------------------------------------------------------------
    "homogeneous_organometallic": {
        "keywords": ["homogeneous", "organometallic", "oxidative addition", "reductive elimination",
                     "migratory insertion", "transmetalation", "cross coupling",
                     "suzuki", "heck", "buchwald"],
        "domain": "homogeneous",
        "bond_events": ["oxidative addition", "reductive elimination", "migratory insertion",
                        "transmetalation", "ligand substitution"],
        "key_species": ["L_n M", "oxidative addition product", "reductive elimination product"],
        "step_kind": "organometallic",
        "dft_notes": "Use M06-L or B3LYP-D3 with implicit solvent (CPCM). Include dispersion.",
        "prompt_template": """
Generate the catalytic cycle for {reactant} → {product} catalysed by {surface} complex.
Include all oxidative addition, transmetalation, reductive elimination steps.
Conditions: {conditions}.  Literature: {rag_context}
Return strict JSON (same schema).
""",
    },
}


# ===========================================================================
# MechanismResult dataclass
# ===========================================================================

@dataclass
class MechanismResult:
    name: str
    family: str
    domain: str
    surface: Optional[str]
    reactant: str
    product: str
    intermediates: List[str] = field(default_factory=list)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    coads: List[List[str]] = field(default_factory=list)
    ts_candidates: List[str] = field(default_factory=list)
    confidence: float = 0.0
    rationale: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    db_id: Optional[int] = None   # MechanismGraph.id if cached


# ===========================================================================
# Clarification request
# ===========================================================================

@dataclass
class ClarificationNeeded:
    """Returned when critical information is missing from the user's request."""
    missing_fields: List[str]
    questions: List[str]
    safe_suggestions: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "clarification_needed": True,
            "missing_fields": self.missing_fields,
            "questions": self.questions,
            "safe_suggestions": self.safe_suggestions,
        }


# ===========================================================================
# MechanismBuilder
# ===========================================================================

class MechanismBuilder:
    """
    Build or retrieve a reaction mechanism graph for a given system.
    This is the single entry point that replaces the static REGISTRY.
    """

    def __init__(self):
        self._llm_call = None   # injected lazily to avoid circular import
        self._rag = None

    def _lazy_imports(self):
        if self._llm_call is None:
            try:
                from server.utils.openai_wrapper import chatgpt_call
                self._llm_call = chatgpt_call
            except ImportError:
                self._llm_call = self._llm_stub

        if self._rag is None:
            try:
                from server.utils.rag_utils import hybrid_search
                self._rag = hybrid_search
            except ImportError:
                self._rag = self._rag_stub

    @staticmethod
    async def _llm_stub(messages, **kw):
        return {"choices": [{"message": {"content": "{}"}}]}

    @staticmethod
    async def _rag_stub(query, **kw):
        return []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_missing_info(
        self,
        domain: Optional[str],
        surface: Optional[str],
        reactant: Optional[str],
        product: Optional[str],
    ) -> Optional[ClarificationNeeded]:
        """
        Return ClarificationNeeded if required fields are absent.
        Mirrors how Claude asks clarifying questions before proceeding.
        """
        missing = []
        questions = []
        suggestions: Dict[str, List[str]] = {}

        if not domain:
            missing.append("domain")
            questions.append(
                "What type of catalysis is this? "
                "(electrochemical / thermal / photocatalytic / homogeneous)"
            )
            suggestions["domain"] = [
                "electrochemical", "thermal", "photocatalytic", "homogeneous"
            ]

        if not reactant:
            missing.append("reactant")
            questions.append("What is the starting material (reactant)?")

        if not product:
            missing.append("product")
            questions.append("What is the desired product?")

        # Surface is optional for homogeneous; required otherwise
        if not surface and domain not in ("homogeneous", None):
            missing.append("surface")
            questions.append(
                "Which catalyst surface or material should be studied? "
                "(e.g. Pt(111), Cu(100), Fe3O4(001))"
            )
            # Suggest common surfaces for the domain
            if domain == "electrochemical":
                suggestions["surface"] = ["Pt(111)", "Cu(111)", "Ag(111)", "Au(111)", "Ni(111)"]
            elif domain == "thermal":
                suggestions["surface"] = ["Pt(111)", "Pd(111)", "Rh(111)", "Ru(0001)", "Ni(111)"]

        if missing:
            return ClarificationNeeded(
                missing_fields=missing,
                questions=questions,
                safe_suggestions=suggestions,
            )
        return None

    async def build(
        self,
        domain: str,
        reactant: str,
        product: str,
        surface: Optional[str] = None,
        conditions: Optional[Dict[str, Any]] = None,
        session_id: Optional[int] = None,
        force_regenerate: bool = False,
    ) -> MechanismResult:
        """
        Main entry point. Returns a MechanismResult.

        Order of operations:
          1. Hash the system → check DB cache
          2. If cached (and not force_regenerate) → return cached
          3. RAG literature search for context
          4. Match reaction-type template
          5. LLM generation with template + RAG context
          6. Validate + normalise
          7. Save to DB
          8. Return result
        """
        self._lazy_imports()
        conditions = conditions or {}

        # 1. DB cache lookup
        sys_hash = self._system_hash(domain, surface, reactant, product)
        if not force_regenerate:
            cached = await self._db_lookup(sys_hash)
            if cached:
                log.info("MechanismBuilder: cache hit for %s", sys_hash[:8])
                return cached

        # 2. RAG context
        rag_query = f"{domain} catalysis {reactant} to {product} on {surface or 'catalyst'} mechanism"
        rag_chunks = await self._rag(rag_query, top_k=6)  # type: ignore
        rag_text = self._format_rag(rag_chunks)

        # 3. Template matching
        template = self._match_template(domain, reactant, product)

        # 4. LLM generation
        result = await self._llm_generate(
            template=template,
            domain=domain,
            surface=surface,
            reactant=reactant,
            product=product,
            conditions=conditions,
            rag_context=rag_text,
        )

        # 5. Validate
        result = self._validate_and_normalise(result)

        # 6. Save to DB
        db_id = await self._db_save(
            sys_hash=sys_hash,
            domain=domain,
            surface=surface,
            reactant=reactant,
            product=product,
            conditions=conditions,
            result=result,
        )
        result.db_id = db_id

        return result

    # ------------------------------------------------------------------
    # Template matching
    # ------------------------------------------------------------------

    def _match_template(
        self,
        domain: str,
        reactant: str,
        product: str,
    ) -> Dict[str, Any]:
        """Return the best matching REACTION_TYPE_TEMPLATE."""
        domain_lc = domain.lower()
        text = f"{domain_lc} {reactant} {product}".lower()

        best_key = None
        best_hits = 0

        for key, tpl in REACTION_TYPE_TEMPLATES.items():
            # Prefer domain match first
            if tpl.get("domain") not in (domain_lc, domain_lc.split("_")[0]):
                continue
            hits = sum(1 for kw in tpl.get("keywords", []) if kw in text)
            if hits > best_hits:
                best_hits = hits
                best_key = key

        if best_key is None:
            # Fallback: match by domain only
            for key, tpl in REACTION_TYPE_TEMPLATES.items():
                if domain_lc in tpl.get("domain", ""):
                    best_key = key
                    break

        if best_key is None:
            best_key = "thermal_dehydrogenation"   # last resort

        log.info("MechanismBuilder: matched template '%s'", best_key)
        return REACTION_TYPE_TEMPLATES[best_key]

    # ------------------------------------------------------------------
    # LLM generation
    # ------------------------------------------------------------------

    async def _llm_generate(
        self,
        template: Dict[str, Any],
        domain: str,
        surface: Optional[str],
        reactant: str,
        product: str,
        conditions: Dict,
        rag_context: str,
    ) -> MechanismResult:
        surface_clean = (surface or "catalyst").replace("(", "").replace(")", "").replace("/", "_")

        # Fill in the template prompt
        prompt_raw = template.get("prompt_template", "Generate a DFT mechanism. Return JSON.")
        prompt = prompt_raw.format(
            reactant=reactant,
            product=product,
            surface=surface or "the catalyst",
            surface_clean=surface_clean,
            reactant_minus_H=reactant[:-1] if reactant.endswith("H") else reactant + "-H",
            conditions=json.dumps(conditions) if conditions else "standard conditions",
            rag_context=rag_context[:2000] if rag_context else "No literature context available.",
        )

        system_msg = (
            "You are a computational catalysis expert specialising in DFT reaction mechanisms. "
            "You always return valid JSON with the exact schema requested. "
            "Species on the surface end with *. Gas-phase species end with (g). "
            "Do not include bare electrons or protons in intermediates list. "
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ]

        raw = None
        try:
            raw = await self._llm_call(messages, model="gpt-4o", temperature=0.1, max_tokens=2000)  # type: ignore
            parsed = self._extract_json(raw)
        except (json.JSONDecodeError, ValueError) as e:
            log.error("MechanismBuilder LLM call failed: %s", e)
            parsed = {}

        return MechanismResult(
            name=parsed.get("name", f"{domain}_{reactant}_to_{product}_{surface_clean}"),
            family=parsed.get("family", domain),
            domain=domain,
            surface=surface,
            reactant=reactant,
            product=product,
            intermediates=parsed.get("intermediates", []),
            steps=parsed.get("steps", []),
            coads=parsed.get("coads", []),
            ts_candidates=parsed.get("ts_candidates", []),
            confidence=float(parsed.get("confidence", 0.5)),
            rationale=parsed.get("rationale", ""),
            provenance={"source": "llm", "model": "gpt-4o", "template": template.get("step_kind", "?")},
        )

    # ------------------------------------------------------------------
    # Validation + normalisation
    # ------------------------------------------------------------------

    def _validate_and_normalise(self, result: MechanismResult) -> MechanismResult:
        """Normalise species notation and remove obvious errors."""

        def _norm(s: str) -> str:
            s = s.strip()
            # Ensure surface species end with *
            if not s.endswith("*") and not s.endswith("(g)") and s not in (
                "H+", "e-", "OH-", "H2O", "H+", "hv", "*"
            ):
                # Heuristic: if no parens and no (g), treat as surface species
                if "(" not in s and s not in ("H2", "N2", "O2", "CO2", "H2O"):
                    s = s + "*"
            return s

        result.intermediates = list({_norm(sp) for sp in result.intermediates if sp.strip()})

        norm_steps = []
        for step in result.steps:
            if isinstance(step, dict):
                step["r"] = [_norm(s) for s in step.get("r", [])]
                step["p"] = [_norm(s) for s in step.get("p", [])]
                norm_steps.append(step)
        result.steps = norm_steps

        result.coads = [
            [_norm(a), _norm(b)] for a, b in result.coads
            if isinstance(a, str) and isinstance(b, str)
        ]

        # Ensure product appears in intermediates
        for sp in [result.reactant + "*", result.product + "(g)", result.product + "*"]:
            if sp not in result.intermediates:
                result.intermediates.insert(0, sp)

        return result

    # ------------------------------------------------------------------
    # DB helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _system_hash(domain: str, surface: Optional[str], reactant: str, product: str) -> str:
        key = f"{domain.lower()}|{(surface or '').lower()}|{reactant.lower()}|{product.lower()}"
        return hashlib.sha256(key.encode()).hexdigest()

    async def _db_lookup(self, sys_hash: str) -> Optional[MechanismResult]:
        try:
            from server.db import AsyncSessionLocal, ReactionSystem, MechanismGraph
            from sqlalchemy import select
            async with AsyncSessionLocal() as s:
                stmt = (
                    select(ReactionSystem)
                    .where(ReactionSystem.system_hash == sys_hash)
                )
                res = await s.execute(stmt)
                sys_row = res.scalar_one_or_none()
                if sys_row is None:
                    return None

                stmt2 = (
                    select(MechanismGraph)
                    .where(MechanismGraph.system_id == sys_row.id)
                    .order_by(MechanismGraph.confidence.desc())
                    .limit(1)
                )
                res2 = await s.execute(stmt2)
                mech = res2.scalar_one_or_none()
                if mech is None:
                    return None

                return MechanismResult(
                    name=mech.name,
                    family=mech.family or "",
                    domain=sys_row.domain,
                    surface=sys_row.surface,
                    reactant=sys_row.reactant,
                    product=sys_row.product,
                    intermediates=mech.intermediates or [],
                    steps=mech.steps or [],
                    coads=mech.coads or [],
                    ts_candidates=mech.ts_candidates or [],
                    confidence=mech.confidence or 0.0,
                    provenance=mech.provenance or {},
                    db_id=mech.id,
                )
        except Exception as e:
            log.debug("_db_lookup failed: %s", e)
            return None

    async def _db_save(
        self,
        sys_hash: str,
        domain: str,
        surface: Optional[str],
        reactant: str,
        product: str,
        conditions: Dict,
        result: MechanismResult,
    ) -> Optional[int]:
        try:
            from server.db import AsyncSessionLocal, ReactionSystem, MechanismGraph
            from sqlalchemy import select
            async with AsyncSessionLocal() as s:
                # Upsert ReactionSystem
                stmt = select(ReactionSystem).where(ReactionSystem.system_hash == sys_hash)
                res = await s.execute(stmt)
                sys_row = res.scalar_one_or_none()

                if sys_row is None:
                    sys_row = ReactionSystem(
                        domain=domain,
                        surface=surface,
                        reactant=reactant,
                        product=product,
                        conditions=conditions,
                        system_hash=sys_hash,
                    )
                    s.add(sys_row)
                    await s.flush()

                mech = MechanismGraph(
                    system_id=sys_row.id,
                    name=result.name,
                    family=result.family,
                    intermediates=result.intermediates,
                    steps=result.steps,
                    coads=result.coads,
                    ts_candidates=result.ts_candidates,
                    confidence=result.confidence,
                    provenance=result.provenance,
                )
                s.add(mech)
                await s.commit()
                await s.refresh(mech)
                return mech.id
        except Exception as e:
            log.error("_db_save failed: %s", e, exc_info=True)
            return None

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(raw: Any) -> Dict:
        """Extract the JSON dict from an OpenAI response."""
        try:
            if isinstance(raw, dict):
                content = (
                    raw.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "{}")
                )
            elif isinstance(raw, str):
                content = raw
            else:
                content = str(raw)

            # Strip markdown code fences if present
            content = re.sub(r"```(?:json)?", "", content).strip("` \n")
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            # Try to extract the first {...} block
            m = re.search(r"\{.*\}", str(raw), re.DOTALL)
            if m:
                try:
                    return json.loads(m.group(0))
                except (json.JSONDecodeError, ValueError):
                    pass
            return {}

    @staticmethod
    def _format_rag(chunks: List[Dict]) -> str:
        if not chunks:
            return ""
        lines = []
        for c in chunks[:4]:
            title = c.get("title", "?")
            text  = c.get("text", "")[:400]
            year  = c.get("year", "")
            lines.append(f"[{title}, {year}] {text}")
        return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# Module-level singleton (import and use directly)
# ---------------------------------------------------------------------------
_builder = MechanismBuilder()


async def build_mechanism(
    domain: str,
    reactant: str,
    product: str,
    surface: Optional[str] = None,
    conditions: Optional[Dict] = None,
    session_id: Optional[int] = None,
    force_regenerate: bool = False,
) -> MechanismResult:
    """Convenience function wrapping MechanismBuilder.build()."""
    return await _builder.build(
        domain=domain,
        reactant=reactant,
        product=product,
        surface=surface,
        conditions=conditions,
        session_id=session_id,
        force_regenerate=force_regenerate,
    )


def check_missing_info(
    domain: Optional[str],
    surface: Optional[str],
    reactant: Optional[str],
    product: Optional[str],
) -> Optional[ClarificationNeeded]:
    """Convenience function wrapping MechanismBuilder.check_missing_info()."""
    return _builder.check_missing_info(domain, surface, reactant, product)
