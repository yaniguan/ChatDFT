#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
End-to-end test: "i need to study the c4h10 to c4h8 on the ag111 surface"
Tests: session → intent → hypothesis → plan, with RAG context at each stage.
"""
import json, sys, time, requests

BASE = "http://127.0.0.1:8001"
QUERY = "i need to study the c4h10 to c4h8 on the ag111 surface"

SEP = "\n" + "="*70 + "\n"
def pp(label, data, indent=2):
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"{'─'*60}")
    if isinstance(data, (dict, list)):
        print(json.dumps(data, indent=indent, ensure_ascii=False))
    else:
        print(str(data))

def post(ep, body, timeout=60):
    t0 = time.time()
    try:
        r = requests.post(f"{BASE}{ep}", json=body, timeout=timeout)
        elapsed = time.time() - t0
        try:
            d = r.json()
        except Exception:
            d = {"raw": r.text[:500]}
        d["_status_code"] = r.status_code
        d["_elapsed_s"] = round(elapsed, 2)
        return d
    except Exception as e:
        return {"ok": False, "error": str(e)}

# ─────────────────────────────────────────────────────────────────────────────
print(SEP + "E2E TEST: " + QUERY + SEP)

# 0) Health check
print("\n[0] Health check...")
try:
    r = requests.get(f"{BASE}/chat/session/list", timeout=5)
    print(f"  Server reachable: {r.status_code}")
except Exception as e:
    print(f"  ❌ Server not reachable: {e}")
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# 1) Create session
print("\n[1] Create session...")
sess = post("/chat/session/create", {"name": "C4H10→C4H8 on Ag(111)", "project": "dehydrogenation"})
pp("Session create response", sess)

session_id = sess.get("id") or sess.get("session_id")
if not session_id:
    # Try list and pick latest
    r2 = requests.post(f"{BASE}/chat/session/list", json={})
    rows = r2.json().get("sessions") or r2.json()
    if rows:
        session_id = rows[0]["id"] if isinstance(rows[0], dict) and "id" in rows[0] else None
print(f"  → session_id = {session_id}")

# ─────────────────────────────────────────────────────────────────────────────
# 2) Intent agent
print("\n[2] Intent agent...")
intent_resp = post("/chat/intent", {"session_id": session_id, "text": QUERY})
pp("Intent response", intent_resp)

intent = intent_resp.get("intent") or {}

# Evaluate intent quality
checks = {
    # intent stores reaction in task/stage field, not "reaction"
    "has_reaction":  bool(intent.get("task") or intent.get("reaction") or intent.get("stage")),
    # substrate / material stored in system dict
    "has_substrate": bool(intent.get("substrate") or intent.get("system", {}).get("material")
                         or intent.get("system", {}).get("catalyst")),
    # domain stored in area or domain field
    "has_domain":    bool(intent.get("area") or intent.get("domain") or intent.get("intent_area")),
    "has_task":      bool(intent.get("task") or intent.get("specific_intent")),
    "has_species":   any(s in str(intent).lower() for s in ["c4h10", "butane"]),
    "has_ag":        any(s in str(intent).lower() for s in ["ag", "silver"]),
}
print("\n  Intent quality checks:")
for k, v in checks.items():
    mark = "✅" if v else "❌"
    print(f"    {mark} {k}: {v}")

# ─────────────────────────────────────────────────────────────────────────────
# 3) Knowledge / RAG search
print("\n[3] RAG knowledge search...")
rag_resp = post("/chat/knowledge", {
    "session_id": session_id,
    "query": "butane dehydrogenation silver surface DFT mechanism",
    "intent": intent,
    "limit": 5,
})
pp("RAG response (truncated)", {
    "ok": rag_resp.get("ok"),
    "n_records": len(rag_resp.get("records", [])),
    "n_fetched_from_arxiv": rag_resp.get("n_fetched_from_arxiv"),
    "n_new_ingested": rag_resp.get("n_new_ingested"),
    "elapsed": rag_resp.get("_elapsed_s"),
    "top_results": [
        {"title": r["title"][:60], "score": r.get("score"), "year": r.get("year")}
        for r in rag_resp.get("records", [])[:3]
    ],
})

# ─────────────────────────────────────────────────────────────────────────────
# 4) Hypothesis agent
print("\n[4] Hypothesis agent...")
hyp_resp = post("/chat/hypothesis", {"session_id": session_id, "intent": intent}, timeout=90)
pp("Hypothesis response keys", list(hyp_resp.keys()))

hyp = hyp_resp.get("hypothesis") or {}
graph = hyp_resp.get("graph") or {}

print("\n  === HYPOTHESIS MARKDOWN (first 800 chars) ===")
md = hyp.get("md") or ""
print(md[:800] + ("..." if len(md) > 800 else ""))

print("\n  === REACTION GRAPH ===")
pp("system", graph.get("system"))
rn = graph.get("reaction_network") or []
print(f"  reaction_network: {len(rn)} steps")
for step in rn[:5]:
    print(f"    • {step}")
print(f"  intermediates: {graph.get('intermediates', [])}")
print(f"  ts_edges: {graph.get('ts_edges', [])}")
print(f"  provenance: {graph.get('provenance', {}).get('source','?')} / {graph.get('provenance', {}).get('builder_name','?')}")

# Hypothesis quality checks
hyp_checks = {
    "has_markdown":         len(md) > 100,
    "mentions_c4h10":       any(s in md.lower() for s in ["c4h10", "butane", "c₄h₁₀"]),
    "mentions_ag":          any(s in md.lower() for s in ["ag", "silver"]),
    "has_reaction_network": len(rn) > 0,
    "has_intermediates":    len(graph.get("intermediates", [])) > 0,
    "has_ts_edges":         len(graph.get("ts_edges", [])) > 0,
    "has_provenance":       bool(graph.get("provenance")),
}
print("\n  Hypothesis quality checks:")
for k, v in hyp_checks.items():
    mark = "✅" if v else "❌"
    print(f"    {mark} {k}")

# ─────────────────────────────────────────────────────────────────────────────
# 5) Plan agent
print("\n[5] Plan agent...")
plan_resp = post("/chat/plan", {
    "session_id": session_id,
    "intent": intent,
    "hypothesis": hyp_resp.get("hypothesis"),
}, timeout=120)
pp("Plan response keys", list(plan_resp.keys()))

plan = plan_resp.get("plan") or {}
tasks = plan.get("tasks") or plan_resp.get("tasks") or []

print(f"\n  Total tasks: {len(tasks)}")
print("\n  === TASK LIST ===")
for i, t in enumerate(tasks):
    name = t.get("name") or t.get("title") or f"Task {i}"
    agent = t.get("agent") or ""
    deps = t.get("depends_on") or []
    ep = (t.get("params") or {}).get("endpoint") or ""
    print(f"  [{i:02d}] {name}")
    print(f"        agent={agent}  endpoint={ep}  deps={deps}")

# Plan quality checks
plan_task_names = " ".join(t.get("name","") + " " + t.get("agent","") for t in tasks).lower()
plan_checks = {
    "has_tasks":             len(tasks) > 3,
    "has_slab_build":        any(s in plan_task_names for s in ["slab", "ag111", "ag(111)", "build"]),
    "has_ads_opt":           any(s in plan_task_names for s in ["adsorption", "c4h10", "butane"]),
    "has_gcdft":             any(s in plan_task_names for s in ["gcdft", "gc_dft", "gc-dft", "grand", "electroch"]),
    "has_freq_thermo":       any(s in plan_task_names for s in ["freq", "zpe", "thermo", "gibbs"]),
    "has_ts":                any(s in plan_task_names for s in ["neb", "ts", "transition", "barrier"]),
    "has_analysis":          any(s in plan_task_names for s in ["analys", "diagram", "microkin"]),
    "has_dependencies":      any(len(t.get("depends_on") or []) > 0 for t in tasks),
}
print("\n  Plan quality checks:")
for k, v in plan_checks.items():
    mark = "✅" if v else "❌"
    print(f"    {mark} {k}")

# ─────────────────────────────────────────────────────────────────────────────
# 6) Diagnosis
print(SEP + "DIAGNOSIS SUMMARY" + SEP)

all_checks = {**checks, **hyp_checks, **plan_checks}
passed = sum(v for v in all_checks.values())
total  = len(all_checks)
print(f"Checks passed: {passed}/{total}")

failed = [k for k, v in all_checks.items() if not v]
if failed:
    print("\n❌ FAILED CHECKS:")
    for f in failed:
        print(f"  - {f}")
else:
    print("\n✅ All checks passed!")

print("\n--- Key issues to investigate ---")
if not checks["has_species"]:
    print("  ⚠ Intent agent did not extract C4H10 as species → needs system prompt update")
if not checks["has_ag"]:
    print("  ⚠ Intent agent did not extract Ag(111) substrate → check entity extraction")
if not hyp_checks["has_reaction_network"]:
    print("  ⚠ Hypothesis agent produced no reaction network → builder import or LLM call failed")
if not hyp_checks["has_ts_edges"]:
    print("  ⚠ No TS edges in hypothesis → LLM not generating dehydrogenation TS steps")
if not plan_checks["has_gcdft"]:
    print("  ⚠ Plan agent not generating GC-DFT tasks → update plan_agent task groups")
if not plan_checks["has_ts"]:
    print("  ⚠ Plan agent not generating NEB/TS tasks → needs task generation fix")
if not plan_checks["has_dependencies"]:
    print("  ⚠ Tasks have no dependencies → plan_agent not wiring task graph")

print("\n--- Intent structure ---")
print(json.dumps(intent, indent=2, ensure_ascii=False)[:1000])

print("\n--- Full hypothesis markdown ---")
print(md[:2000])

print("\n--- Plan tasks detail ---")
for i, t in enumerate(tasks[:20]):
    print(f"\n[{i:02d}] {t.get('name','?')}")
    p = t.get("params") or {}
    payload = p.get("payload") or {}
    print(f"     agent: {t.get('agent','?')}")
    print(f"     endpoint: {p.get('endpoint','')}")
    print(f"     depends_on: {t.get('depends_on',[])}")
    if payload:
        print(f"     payload keys: {list(payload.keys())[:6]}")
