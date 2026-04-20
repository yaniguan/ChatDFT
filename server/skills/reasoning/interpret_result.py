# server/skills/reasoning/interpret_result.py
"""Skill: LLM-driven interpretation of a DFT result in context of hypothesis."""

from __future__ import annotations

from server.skills.base import Domain, FieldSpec, Skill, SkillContext, SkillResult, SkillSpec
from server.skills.registry import register_skill


@register_skill
class InterpretResult(Skill):
    spec = SkillSpec(
        name="reasoning.interpret_result",
        description="Interpret a DFT result (energy, barrier, etc.) in the context of the current hypothesis and suggest follow-ups.",
        domain=Domain.REASONING,
        inputs={
            "result_type": FieldSpec(
                type="str", description="Type: adsorption_energy | activation_barrier | total_energy"
            ),
            "species": FieldSpec(type="str", description="Species involved, e.g. 'CO*'"),
            "surface": FieldSpec(type="str", description="Surface, e.g. 'Pt(111)'"),
            "value": FieldSpec(type="float", description="Primary numeric result (eV)"),
            "converged": FieldSpec(
                type="bool", description="Whether the calculation converged", default=True, required=False
            ),
            "hypothesis_excerpt": FieldSpec(
                type="str", description="Current hypothesis text (truncated)", default="", required=False
            ),
        },
        outputs={
            "interpretation": FieldSpec(type="str", description="One-paragraph physical/chemical interpretation"),
            "suggestions": FieldSpec(type="list", description="1-3 concrete follow-up recommendations"),
            "anomaly_flag": FieldSpec(type="bool", description="Whether the value seems anomalous"),
        },
        cost=0,
        side_effects=[],
        requires=[],
    )

    async def execute(self, ctx: SkillContext) -> SkillResult:
        inp = ctx.inputs
        result_type = inp.get("result_type", "unknown")
        species = inp.get("species", "")
        surface = inp.get("surface", "")
        value = inp.get("value")
        converged = inp.get("converged", True)
        hypothesis = inp.get("hypothesis_excerpt", "")

        prompt = f"""You are a DFT computational chemistry assistant. Interpret this result:
- Type: {result_type}
- Species: {species}  Surface: {surface}
- Value: {value} eV   Converged: {converged}

Hypothesis context (if any):
{hypothesis[:800] or "(none)"}

Provide:
1. One paragraph: what does this mean physically/chemically?
2. Flag if anomalous (implausibly large binding, wrong sign, etc.)
3. 1-3 concrete follow-up actions.

Be concise (<150 words). Return JSON: {{"interpretation": "...", "anomaly_flag": bool, "suggestions": ["...", ...]}}"""

        try:
            from server.utils.openai_wrapper import chatgpt_call

            resp = await chatgpt_call(
                messages=[{"role": "user", "content": prompt}],
                model="gpt-4o-mini",
                temperature=0.2,
                max_tokens=400,
                json_mode=True,
                agent_name="skill.interpret_result",
                session_id=ctx.session_id,
            )

            if "error" in resp and not resp.get("choices"):
                return SkillResult(ok=False, error=f"LLM error: {resp.get('error')}")

            import json

            text = resp["choices"][0]["message"]["content"]
            payload = json.loads(text)

            return SkillResult(
                ok=True,
                outputs={
                    "interpretation": str(payload.get("interpretation", "")),
                    "suggestions": list(payload.get("suggestions", [])),
                    "anomaly_flag": bool(payload.get("anomaly_flag", False)),
                },
            )
        except Exception as exc:
            return SkillResult(ok=False, error=f"interpret failed: {exc}")
