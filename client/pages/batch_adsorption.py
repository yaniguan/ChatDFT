"""
Batch Adsorption Study — Streamlit Page

Submit H adsorption calculations on multiple metals and sites,
monitor progress, and view results with Excel export.
"""
import time
import streamlit as st
import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from client.utils import api

st.set_page_config(page_title="Batch Adsorption | ChatDFT", page_icon="⚛️", layout="wide")

st.title("⚛️ Batch Adsorption Study")
st.markdown("Submit DFT calculations for adsorbate on multiple metals and sites.")

# ── Sidebar: Input form ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Configuration")

    adsorbate = st.selectbox("Adsorbate", ["H", "CO", "OH", "O", "N", "NH3", "H2O"], index=0)
    metals = st.multiselect("Metals", ["Pt", "Pd", "Ni", "Cu", "Au", "Ag", "Rh", "Ir", "Fe"],
                            default=["Pt", "Pd", "Ni"])
    facet = st.selectbox("Surface Facet", ["111", "100", "110"], index=0)
    sites = st.multiselect("Adsorption Sites", ["ontop", "bridge", "hollow"],
                           default=["ontop", "bridge", "hollow"])

    with st.expander("Advanced Settings"):
        supercell = st.selectbox("Supercell", ["3x4", "4x4", "2x4"], index=0)
        nlayers = st.slider("Slab Layers", 3, 6, 4)
        encut = st.number_input("ENCUT (eV)", value=400, step=50)
        kpoints = st.text_input("KPOINTS", value="4 4 1")
        include_refs = st.checkbox("Include reference calcs (for E_ads)", value=True)

    n_ads = len(metals) * len(sites)
    n_ref = len(metals) + 1 if include_refs else 0
    n_total = n_ads + n_ref
    st.info(f"**{n_total} calculations** ({n_ads} adsorption + {n_ref} reference)")

    submit = st.button("🚀 Submit Batch", type="primary", use_container_width=True)

# ── Session state ────────────────────────────────────────────────────────
if "batch_uid" not in st.session_state:
    st.session_state.batch_uid = None
if "batch_jobs" not in st.session_state:
    st.session_state.batch_jobs = []

# ── Submit ───────────────────────────────────────────────────────────────
if submit:
    with st.spinner("Generating structures and submitting to Hoffman2..."):
        result = api.post("/api/batch_adsorption", {
            "adsorbate": adsorbate,
            "metals": metals,
            "facet": facet,
            "sites": sites,
            "supercell": supercell,
            "nlayers": nlayers,
            "encut": encut,
            "kpoints": kpoints,
            "server_name": "hoffman2",
            "include_references": include_refs,
        })

    if result.get("ok"):
        st.session_state.batch_uid = result["batch_uid"]
        st.session_state.batch_jobs = result.get("jobs", [])
        st.success(f"Submitted {result['n_jobs']} jobs! Batch: `{result['batch_uid'][:8]}`")
    else:
        st.error(f"Submission failed: {result.get('error', result.get('detail', 'Unknown error'))}")

# ── Status monitoring ────────────────────────────────────────────────────
if st.session_state.batch_uid:
    st.markdown("---")
    st.subheader("Job Status")

    col1, col2 = st.columns([3, 1])
    with col2:
        refresh = st.button("🔄 Refresh Status")

    if refresh or submit:
        with st.spinner("Polling job statuses..."):
            status = api.get("/api/batch_status", {"batch_uid": st.session_state.batch_uid})
    else:
        status = api.get("/api/batch_status", {"batch_uid": st.session_state.batch_uid})

    if status.get("ok"):
        jobs = status.get("jobs", [])

        # Progress bar
        n_total = status["n_total"]
        n_done = status["n_done"]
        progress = n_done / n_total if n_total > 0 else 0
        st.progress(progress, text=f"{n_done}/{n_total} completed | "
                    f"{status['n_running']} running | {status['n_queued']} queued")

        # Status table
        rows = []
        for j in jobs:
            status_emoji = {
                "submitted": "🟡", "queued": "🟡", "running": "🔵",
                "done": "🟢", "synced": "✅", "failed": "🔴",
            }.get(j["status"], "⚪")

            rows.append({
                "Status": f"{status_emoji} {j['status']}",
                "Title": j["title"],
                "Metal": j["metal"],
                "Site": j["site"],
                "PBS ID": j["pbs_id"],
                "Energy (eV)": f"{j['energy']:.4f}" if j.get("energy") is not None else "—",
                "Max Force": f"{j['max_force']:.4f}" if j.get("max_force") is not None else "—",
            })

        df_status = pd.DataFrame(rows)
        st.dataframe(df_status, use_container_width=True, hide_index=True)

        # Auto-refresh hint
        if not status["all_done"]:
            st.caption("Click 'Refresh Status' to update. Jobs typically take 10-60 minutes on Hoffman2.")

        # ── Results (when all done) ──────────────────────────────────
        if status["all_done"]:
            st.markdown("---")
            st.subheader("📊 Results")

            # Separate adsorption and reference results
            ads_jobs = [j for j in jobs if not j.get("is_reference")]
            ref_jobs = [j for j in jobs if j.get("is_reference")]

            # Reference energies
            ref_energies = {}
            gas_energy = {}
            for j in ref_jobs:
                if j.get("energy") is not None:
                    if j["site"] == "gas":
                        gas_energy[j["adsorbate"]] = j["energy"]
                    else:
                        ref_energies[j["metal"]] = j["energy"]

            # Build results table
            result_rows = []
            for j in ads_jobs:
                e_total = j.get("energy")
                e_ads = None
                if e_total is not None:
                    e_slab = ref_energies.get(j["metal"])
                    e_h2 = gas_energy.get("H2")
                    if e_slab is not None and e_h2 is not None:
                        e_ads = e_total - e_slab - 0.5 * e_h2

                result_rows.append({
                    "Metal": j["metal"],
                    "Site": j["site"],
                    "E_total (eV)": e_total,
                    "E_ads (eV)": e_ads,
                    "Max Force (eV/A)": j.get("max_force"),
                    "Converged": j.get("converged"),
                })

            df_results = pd.DataFrame(result_rows)
            st.dataframe(df_results, use_container_width=True, hide_index=True)

            # Bar chart of adsorption energies
            if "E_ads (eV)" in df_results.columns and df_results["E_ads (eV)"].notna().any():
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(10, 5))
                df_plot = df_results.dropna(subset=["E_ads (eV)"])
                df_plot["Label"] = df_plot["Metal"] + "-" + df_plot["Site"]

                colors = {"Pt": "#2196F3", "Pd": "#4CAF50", "Ni": "#FF9800",
                          "Cu": "#FF5722", "Au": "#FFC107", "Ag": "#9E9E9E"}
                bar_colors = [colors.get(m, "#607D8B") for m in df_plot["Metal"]]

                bars = ax.bar(df_plot["Label"], df_plot["E_ads (eV)"], color=bar_colors, edgecolor="white")
                ax.set_ylabel("E_ads (eV)")
                ax.set_title(f"{adsorbate} Adsorption Energy on (111) Surfaces")
                ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                st.pyplot(fig)

            # Reference energies table
            with st.expander("Reference Energies"):
                ref_rows = []
                for m, e in ref_energies.items():
                    ref_rows.append({"System": f"{m}(111) clean slab", "Energy (eV)": f"{e:.4f}"})
                for mol, e in gas_energy.items():
                    ref_rows.append({"System": f"{mol} gas-phase", "Energy (eV)": f"{e:.4f}"})
                st.table(pd.DataFrame(ref_rows))

            # Excel download
            st.markdown("### Download Results")
            try:
                import requests
                url = f"{api.BASE}/api/batch_results_excel?batch_uid={st.session_state.batch_uid}"
                r = requests.get(url, timeout=30)
                if r.status_code == 200:
                    st.download_button(
                        "📥 Download Excel",
                        data=r.content,
                        file_name=f"H_adsorption_{st.session_state.batch_uid[:8]}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    )
            except Exception as e:
                st.warning(f"Excel export error: {e}")

    else:
        st.warning(f"Could not fetch status: {status.get('error', status.get('detail', ''))}")
else:
    st.info("Configure your batch study in the sidebar and click **Submit Batch** to start.")
