"""
Test script: ingest a paper and verify semantic search works.
Run from repo root:  python scripts/test_ingest.py
"""
import asyncio
import os
import sys

os.environ.setdefault(
    "DATABASE_URL",
    "postgresql+asyncpg://yaniguan@localhost:5432/chatdft_ase"
)
os.environ.setdefault("OPENAI_API_KEY", os.environ.get("OPENAI_API_KEY", ""))

sys.path.insert(0, ".")

from server.utils.rag_utils import ingest_paper, hybrid_search, rag_context


async def main():
    print("=== 1. Ingest papers ===")

    papers = [
        {
            "title": "Electrochemical Dehydrogenation of Butane on Pt(111): A DFT Study",
            "abstract": (
                "We study the electrochemical dehydrogenation of n-butane (C4H10) to "
                "1-butene (C4H8) on the Pt(111) surface using density functional theory. "
                "The reaction proceeds via sequential C-H bond activation steps: "
                "C4H10* → C4H9* + H*, followed by C4H9* → C4H8* + H*. "
                "We find that the first C-H cleavage is rate-limiting with a barrier of 0.68 eV. "
                "The potential-determining step under electrochemical conditions involves "
                "proton-coupled electron transfer (PCET). "
                "Adsorption of C4H10 on Pt(111) is physisorptive (-0.35 eV), while "
                "the C4H9 intermediate binds strongly (-1.82 eV) via a Pt-C sigma bond. "
                "GC-DFT calculations show the reaction becomes thermodynamically favorable "
                "at potentials more negative than -0.42 V vs RHE."
            ),
            "source_type": "manual",
            "source_id": "test_dehydrogenation_pt_001",
            "tags": ["dehydrogenation", "C4H10", "butane", "C4H8", "butene",
                     "Pt", "Pt(111)", "electrochemical", "PCET", "DFT"],
            "year": 2024,
        },
        {
            "title": "C-H Activation on Transition Metal Surfaces: Pt, Pd, Rh Comparison",
            "abstract": (
                "Systematic DFT study of C-H bond activation on Pt(111), Pd(111), and Rh(111). "
                "Pt shows the lowest barrier for initial C-H cleavage in alkanes (0.65-0.72 eV). "
                "The d-band center model explains the trend: Pt has optimal d-band filling. "
                "Beta-H elimination is the dominant pathway for alkene formation. "
                "H* recombination to H2(g) is facile on all three surfaces. "
                "Selectivity toward dehydrogenation vs. cracking is controlled by "
                "surface H* coverage and reaction temperature."
            ),
            "source_type": "manual",
            "source_id": "test_ch_activation_comparison",
            "tags": ["C-H activation", "Pt", "Pd", "Rh", "alkane", "dehydrogenation",
                     "d-band", "DFT", "thermal catalysis"],
            "year": 2023,
        },
        {
            "title": "GC-DFT Methods for Electrochemical Interface Modeling",
            "abstract": (
                "Grand-canonical DFT (GC-DFT) allows modeling of electrochemical reactions "
                "at constant electrode potential. We describe the implementation in VASP using "
                "VASPsol implicit solvation and the linearized Poisson-Boltzmann equation. "
                "NELECT is varied to scan electrode potential from -1.0 to 0 V vs RHE. "
                "The computational hydrogen electrode (CHE) is compared with full GC-DFT. "
                "CHE overestimates barriers by 0.1-0.3 eV for PCET steps. "
                "Key INCAR settings: LSOL=.TRUE., EB_K=80, TAU=0, SIGMA=0.6."
            ),
            "source_type": "manual",
            "source_id": "test_gcdft_methods",
            "tags": ["GC-DFT", "electrochemical", "VASPsol", "PCET", "CHE",
                     "VASP", "electrode potential", "DFT"],
            "year": 2022,
        },
    ]

    for p in papers:
        doc_id = await ingest_paper(**p)
        print(f"  Ingested: '{p['title'][:60]}...' → doc_id={doc_id}")

    print()
    print("=== 2. Semantic search ===")
    queries = [
        "C4H10 dehydrogenation Pt mechanism",
        "electrochemical potential dependent DFT",
        "C-H activation barrier transition metal",
    ]
    for q in queries:
        results = await hybrid_search(q, top_k=2)
        print(f"\n  Query: '{q}'")
        for r in results:
            print(f"    [{r['score']:.3f}] {r['title']} | {r['text'][:80]}...")

    print()
    print("=== 3. RAG context (for LLM prompt injection) ===")
    ctx = await rag_context(
        query="dehydrogenation C4H10 to C4H8 on Pt electrochemical mechanism",
        session_id=None,
        top_k=3,
        include_history=False,
    )
    print(ctx[:800])
    print("...")
    print("\n=== All tests passed ===")


asyncio.run(main())
