#!/usr/bin/env python3
"""
Voronoi Topology Graph Visualization for Cu(111)
==================================================
Generates a publication-quality figure showing:
  (a) Top-down view: atoms + Voronoi edges + adsorption sites (4 classes)
  (b) Side view: layer structure with Voronoi connectivity
  (c) Zoomed inset: fcc vs hcp hollow site distinction
  (d) Voronoi tessellation overlay

Run:
    python -m science.benchmarks.voronoi_viz

Output:
    figures/fig8_voronoi_topology.png / .pdf
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Circle, RegularPolygon, FancyArrowPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from scipy.spatial import Voronoi as ScipyVoronoi

rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FIG_DIR = Path(__file__).resolve().parents[2] / "figures"
FIG_DIR.mkdir(exist_ok=True)

# ── Colors ──
C_ATOM_L0 = "#D32F2F"    # surface layer — red/copper
C_ATOM_L1 = "#1976D2"    # 2nd layer — blue
C_ATOM_L2 = "#7B1FA2"    # 3rd layer — purple
C_ATOM_L3 = "#455A64"    # 4th layer — grey
LAYER_COLORS = [C_ATOM_L0, C_ATOM_L1, C_ATOM_L2, C_ATOM_L3]

C_EDGE = "#90A4AE"
C_TOP = "#43A047"
C_BRIDGE = "#FF9800"
C_FCC = "#2196F3"
C_HCP = "#E91E63"

SITE_COLORS = {"top": C_TOP, "bridge": C_BRIDGE, "hollow_fcc": C_FCC, "hollow_hcp": C_HCP}
SITE_MARKERS = {"top": "o", "bridge": "s", "hollow_fcc": "^", "hollow_hcp": "v"}
SITE_LABELS = {"top": "Top", "bridge": "Bridge", "hollow_fcc": "Hollow (fcc)", "hollow_hcp": "Hollow (hcp)"}


def build_cu111():
    """Build Cu(111) 3x3 slab and run Voronoi analysis."""
    from ase.build import fcc111
    from science.representations.surface_graph import SurfaceTopologyGraph

    slab = fcc111("Cu", size=(3, 3, 4), vacuum=10.0, a=3.615)
    positions = slab.get_positions()
    elements = slab.get_chemical_symbols()
    cell = slab.get_cell()[:].copy()

    stg = SurfaceTopologyGraph(positions, elements, np.array(cell))
    stg.build()
    sites = stg.classify_adsorption_sites()

    return slab, stg, sites


def main():
    print("=" * 60)
    print("  Voronoi Topology Graph — Cu(111) Visualization")
    print("=" * 60)

    slab, stg, sites = build_cu111()
    positions = slab.get_positions()

    fig = plt.figure(figsize=(18, 10))

    # Layout: 2x2 grid
    ax_top = fig.add_subplot(2, 2, 1)     # (a) top-down: atoms + edges + sites
    ax_side = fig.add_subplot(2, 2, 2)    # (b) side view
    ax_zoom = fig.add_subplot(2, 2, 3)    # (c) zoomed hollow site
    ax_voronoi = fig.add_subplot(2, 2, 4) # (d) Voronoi tessellation

    # ─────────────────────────────────────────────────────────────────
    # Panel (a): Top-down view — atoms, bonds, adsorption sites
    # ─────────────────────────────────────────────────────────────────
    ax = ax_top
    ax.set_title("(a) Top-Down View: Topology Graph + Adsorption Sites",
                 fontweight="bold", fontsize=11)
    ax.set_aspect("equal")

    # Draw edges (Voronoi bonds) — surface and interlayer
    surface_idx = {n.index for n in stg.nodes if n.layer == 0}
    # All edges between surface atoms
    for e in stg.edges:
        p1 = positions[e.i, :2]
        p2 = positions[e.j, :2]
        dist = np.linalg.norm(p1 - p2)
        if dist < 8.0:  # generous cutoff for periodic cell
            is_surface = (e.i in surface_idx and e.j in surface_idx)
            lw = 2.0 if is_surface else 0.8
            alpha = 0.7 if is_surface else 0.2
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                    "-", color=C_EDGE, linewidth=lw, alpha=alpha, zorder=1)

    # Draw atoms by layer (surface on top)
    for layer_idx in [3, 2, 1, 0]:
        layer_nodes = [n for n in stg.nodes if n.layer == layer_idx]
        if not layer_nodes:
            continue
        xs = [positions[n.index, 0] for n in layer_nodes]
        ys = [positions[n.index, 1] for n in layer_nodes]
        size = 180 if layer_idx == 0 else 60
        alpha = 1.0 if layer_idx == 0 else 0.3
        color = LAYER_COLORS[min(layer_idx, 3)]
        label = f"Layer {layer_idx}" if layer_idx <= 1 else None
        ax.scatter(xs, ys, s=size, c=color, alpha=alpha, edgecolors="white",
                   linewidths=0.8, zorder=2 + (3 - layer_idx), label=label)

    # Draw adsorption sites
    for site in sites:
        sx, sy = site.position[0], site.position[1]
        marker = SITE_MARKERS[site.site_type]
        color = SITE_COLORS[site.site_type]
        ax.scatter(sx, sy, s=100, c=color, marker=marker, edgecolors="black",
                   linewidths=0.8, zorder=10, alpha=0.85)

    # Site legend
    for stype in ["top", "bridge", "hollow_fcc", "hollow_hcp"]:
        ax.scatter([], [], s=80, c=SITE_COLORS[stype], marker=SITE_MARKERS[stype],
                   edgecolors="black", linewidths=0.8, label=SITE_LABELS[stype])

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")

    # ─────────────────────────────────────────────────────────────────
    # Panel (b): Side view — layer structure + vertical connectivity
    # ─────────────────────────────────────────────────────────────────
    ax = ax_side
    ax.set_title("(b) Side View: Layer Structure + Voronoi Bonds",
                 fontweight="bold", fontsize=11)

    # Draw interlayer edges
    for e in stg.edges:
        p1 = positions[e.i]
        p2 = positions[e.j]
        dist_xy = np.linalg.norm(p1[:2] - p2[:2])
        if dist_xy < 5.0:
            lw = max(0.3, min(2.0, e.voronoi_area / 3.0))
            ax.plot([p1[0], p2[0]], [p1[2], p2[2]],
                    "-", color=C_EDGE, linewidth=lw, alpha=0.4, zorder=1)

    # Draw atoms
    for layer_idx in [3, 2, 1, 0]:
        layer_nodes = [n for n in stg.nodes if n.layer == layer_idx]
        if not layer_nodes:
            continue
        xs = [positions[n.index, 0] for n in layer_nodes]
        zs = [positions[n.index, 2] for n in layer_nodes]
        size = 150 if layer_idx == 0 else 80
        color = LAYER_COLORS[min(layer_idx, 3)]
        label = f"Layer {layer_idx}"
        ax.scatter(xs, zs, s=size, c=color, edgecolors="white", linewidths=0.8,
                   zorder=2 + (3 - layer_idx), label=label, alpha=0.9)

    # Draw site positions projected
    for site in sites:
        if site.site_type in ("hollow_fcc", "hollow_hcp"):
            ax.scatter(site.position[0], site.position[2],
                       s=60, c=SITE_COLORS[site.site_type],
                       marker=SITE_MARKERS[site.site_type],
                       edgecolors="black", linewidths=0.5, zorder=10, alpha=0.7)

    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax.set_xlabel("x (Å)")
    ax.set_ylabel("z (Å)")

    # ─────────────────────────────────────────────────────────────────
    # Panel (c): Zoomed inset — fcc vs hcp hollow distinction
    # ─────────────────────────────────────────────────────────────────
    ax = ax_zoom
    ax.set_title("(c) fcc vs hcp Hollow Site Distinction",
                 fontweight="bold", fontsize=11)
    ax.set_aspect("equal")

    # Find one fcc and one hcp site for demonstration
    fcc_site = next((s for s in sites if s.site_type == "hollow_fcc"), None)
    hcp_site = next((s for s in sites if s.site_type == "hollow_hcp"), None)

    # Always draw the schematic regardless of site availability
    # fcc schematic (left half)
    ax.text(0.25, 0.97, "fcc Hollow", transform=ax.transAxes,
            ha="center", va="top", fontsize=13, fontweight="bold", color=C_FCC)
    ax.text(0.75, 0.97, "hcp Hollow", transform=ax.transAxes,
            ha="center", va="top", fontsize=13, fontweight="bold", color=C_HCP)

    cx_fcc, cy_fcc = 2.5, 3.2
    tri_r = 1.5
    angles_tri = [np.pi/2, np.pi/2 + 2*np.pi/3, np.pi/2 + 4*np.pi/3]
    tri_x = [cx_fcc + tri_r * np.cos(a) for a in angles_tri]
    tri_y = [cy_fcc + tri_r * np.sin(a) for a in angles_tri]

    # Triangle edges + fill
    tri_poly = plt.Polygon(list(zip(tri_x, tri_y)), closed=True,
                           facecolor=C_FCC, alpha=0.08, edgecolor=C_EDGE, linewidth=2.5, zorder=1)
    ax.add_patch(tri_poly)

    # Surface atoms (Layer 0)
    for tx, ty in zip(tri_x, tri_y):
        ax.scatter(tx, ty, s=500, c=C_ATOM_L0, edgecolors="white",
                   linewidths=2, zorder=3)
        ax.text(tx, ty, "Cu", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)

    # fcc site marker at centroid
    cx_h = np.mean(tri_x)
    cy_h = np.mean(tri_y)
    ax.scatter(cx_h, cy_h, s=250, c=C_FCC, marker="^", edgecolors="black",
               linewidths=2, zorder=5)

    # Ghost circle where subsurface atom would be (but isn't)
    ax.scatter(cx_h, cy_h - 1.8, s=250, facecolors="none", edgecolors=C_FCC,
               linewidths=2, linestyle="--", zorder=3)
    ax.plot([cx_h, cx_h], [cy_h - 0.3, cy_h - 1.5],
            ":", color=C_FCC, linewidth=1.5, alpha=0.5)
    ax.text(cx_h, cy_h - 2.3, "No 2nd-layer\natom below", ha="center", va="top",
            fontsize=9, color=C_FCC, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#E3F2FD", edgecolor=C_FCC, alpha=0.8))

    # hcp schematic (right half)
    cx_hcp, cy_hcp = 7.5, 3.2
    tri_x2 = [cx_hcp + tri_r * np.cos(a) for a in angles_tri]
    tri_y2 = [cy_hcp + tri_r * np.sin(a) for a in angles_tri]

    tri_poly2 = plt.Polygon(list(zip(tri_x2, tri_y2)), closed=True,
                            facecolor=C_HCP, alpha=0.08, edgecolor=C_EDGE, linewidth=2.5, zorder=1)
    ax.add_patch(tri_poly2)

    for tx, ty in zip(tri_x2, tri_y2):
        ax.scatter(tx, ty, s=500, c=C_ATOM_L0, edgecolors="white",
                   linewidths=2, zorder=3)
        ax.text(tx, ty, "Cu", ha="center", va="center", fontsize=8,
                color="white", fontweight="bold", zorder=4)

    cx_h2 = np.mean(tri_x2)
    cy_h2 = np.mean(tri_y2)
    ax.scatter(cx_h2, cy_h2, s=250, c=C_HCP, marker="v", edgecolors="black",
               linewidths=2, zorder=5)

    # Subsurface atom (solid — present)
    ax.scatter(cx_h2, cy_h2 - 1.8, s=350, c=C_ATOM_L1, edgecolors="white",
               linewidths=2, zorder=3)
    ax.text(cx_h2, cy_h2 - 1.8, "Cu", ha="center", va="center", fontsize=7,
            color="white", fontweight="bold", zorder=4)
    ax.plot([cx_h2, cx_h2], [cy_h2 - 0.3, cy_h2 - 1.5],
            "-", color=C_HCP, linewidth=2, alpha=0.6)

    # Distance bracket
    ax.annotate("", xy=(cx_h2 + 0.55, cy_h2 - 1.7), xytext=(cx_h2 + 0.55, cy_h2 - 0.15),
                arrowprops=dict(arrowstyle="<->", color="#555", lw=1.5))
    ax.text(cx_h2 + 0.85, cy_h2 - 0.9, "d < 1.5 Å", fontsize=9, color="#555",
            rotation=90, va="center", fontweight="bold")

    ax.text(cx_h2, cy_h2 - 2.3, "2nd-layer atom\ndirectly below", ha="center", va="top",
            fontsize=9, color=C_HCP, style="italic",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="#FCE4EC", edgecolor=C_HCP, alpha=0.8))

    # Dividing line
    ax.axvline(5.0, color="#CCC", linewidth=1.5, linestyle="--", alpha=0.6)

    # Layer labels
    ax.text(0.02, 0.12, "Layer 0\n(surface)", transform=ax.transAxes,
            fontsize=8, color=C_ATOM_L0, fontweight="bold", va="center")
    ax.text(0.02, 0.02, "Layer 1\n(subsurface)", transform=ax.transAxes,
            fontsize=8, color=C_ATOM_L1, fontweight="bold", va="center")

    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5.5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ─────────────────────────────────────────────────────────────────
    # Panel (d): Voronoi tessellation of surface atoms
    # ─────────────────────────────────────────────────────────────────
    ax = ax_voronoi
    ax.set_title("(d) Voronoi Tessellation — Surface Layer",
                 fontweight="bold", fontsize=11)
    ax.set_aspect("equal")

    # Get surface atom positions
    surface_nodes = [n for n in stg.nodes if n.layer == 0]
    surface_pos = np.array([positions[n.index, :2] for n in surface_nodes])

    # Add periodic images for proper Voronoi at boundaries
    cell_2d = slab.get_cell()[:2, :2]
    images = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            shift = di * cell_2d[0] + dj * cell_2d[1]
            images.append(surface_pos + shift)
    all_pts = np.vstack(images)

    vor = ScipyVoronoi(all_pts)

    # Draw Voronoi edges
    for ridge_verts in vor.ridge_vertices:
        if -1 not in ridge_verts:
            v0 = vor.vertices[ridge_verts[0]]
            v1 = vor.vertices[ridge_verts[1]]
            # Only draw within original cell bounds (with margin)
            mx, my = (v0 + v1) / 2
            xmin, xmax = surface_pos[:, 0].min() - 1, surface_pos[:, 0].max() + 1
            ymin, ymax = surface_pos[:, 1].min() - 1, surface_pos[:, 1].max() + 1
            if xmin <= mx <= xmax and ymin <= my <= ymax:
                ax.plot([v0[0], v1[0]], [v0[1], v1[1]],
                        "-", color="#B0BEC5", linewidth=1.0, alpha=0.7, zorder=1)

    # Fill Voronoi cells by coordination number
    for i, node in enumerate(surface_nodes):
        pt_idx = i  # index in the original (non-periodic) set
        # Find the Voronoi region for this point
        region_idx = vor.point_region[pt_idx]
        region = vor.regions[region_idx]
        if -1 not in region and len(region) > 0:
            polygon = vor.vertices[region]
            cn = node.coordination
            # Color by CN
            cn_norm = min(cn / 12.0, 1.0)
            fc = plt.cm.YlOrRd(cn_norm * 0.6 + 0.15)
            ax.fill(*polygon.T, alpha=0.25, color=fc, zorder=0)

    # Draw surface atoms
    ax.scatter(surface_pos[:, 0], surface_pos[:, 1],
               s=200, c=C_ATOM_L0, edgecolors="white", linewidths=1.2, zorder=3)

    # Label CN on each atom
    for i, node in enumerate(surface_nodes):
        ax.text(surface_pos[i, 0], surface_pos[i, 1],
                f"{node.coordination}", ha="center", va="center",
                fontsize=7, color="white", fontweight="bold", zorder=4)

    # Draw adsorption sites on Voronoi
    for site in sites:
        sx, sy = site.position[0], site.position[1]
        xmin, xmax = surface_pos[:, 0].min() - 0.5, surface_pos[:, 0].max() + 0.5
        ymin, ymax = surface_pos[:, 1].min() - 0.5, surface_pos[:, 1].max() + 0.5
        if xmin <= sx <= xmax and ymin <= sy <= ymax:
            marker = SITE_MARKERS[site.site_type]
            color = SITE_COLORS[site.site_type]
            ax.scatter(sx, sy, s=60, c=color, marker=marker, edgecolors="black",
                       linewidths=0.6, zorder=5, alpha=0.8)

    # Legend for sites
    for stype in ["top", "bridge", "hollow_fcc", "hollow_hcp"]:
        ax.scatter([], [], s=50, c=SITE_COLORS[stype], marker=SITE_MARKERS[stype],
                   edgecolors="black", linewidths=0.6, label=SITE_LABELS[stype])
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)

    ax.set_xlabel("x (Å)")
    ax.set_ylabel("y (Å)")
    # Clip to original cell
    pad = 1.0
    ax.set_xlim(surface_pos[:, 0].min() - pad, surface_pos[:, 0].max() + pad)
    ax.set_ylim(surface_pos[:, 1].min() - pad, surface_pos[:, 1].max() + pad)

    # ─────────────────────────────────────────────────────────────────
    fig.suptitle("Voronoi Topology Graph Representation — Cu(111) 3×3 Slab",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        fig.savefig(FIG_DIR / f"fig8_voronoi_topology.{ext}")
    print(f"\n  ✓ fig8_voronoi_topology.png / .pdf saved to {FIG_DIR}")
    plt.close(fig)


if __name__ == "__main__":
    main()
