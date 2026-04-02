"""
Surface Topology Graph Representation
======================================
Scientific motivation
---------------------
Predicting where adsorbates bind on a catalyst surface requires a
representation that is (a) invariant to rigid translations/rotations,
(b) sensitive to the local chemical environment, and (c) cheap enough to
run at every step of a high-throughput screening pipeline.

This module encodes a catalytic slab as a *topology graph*:
  - Nodes  : atoms, with features derived from Voronoi tessellation
  - Edges  : Voronoi-neighbour bonds, with geometric edge attributes
  - Sites  : candidate adsorption positions, classified by geometry

The representation is the input to downstream GNN models and also drives
the rule-based site selection in the structure_agent.

Key references
--------------
[1] Ong et al., Comput. Mater. Sci. 68, 314 (2013)  — pymatgen Voronoi
[2] Zimmermann & Jain, RSC Adv. 10, 3853 (2020)     — VoronoiNN features
[3] Montoya & Persson, npj Comput. Mater. 3, 14 (2017) — site finders
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull, Voronoi

try:
    import networkx as nx
    _HAS_NX = True
except ImportError:
    _HAS_NX = False

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class AtomNode:
    index: int
    element: str
    atomic_number: int
    position: np.ndarray            # fractional (a,b,c) in unit cell
    layer: int                      # 0 = topmost surface layer
    coordination: int = 0           # Voronoi coordination number
    voronoi_volume: float = 0.0     # Å³ — proxy for atom size / stress
    surface_dist: float = 0.0       # Å above the mean surface plane


@dataclass
class BondEdge:
    i: int
    j: int
    length: float                   # Å
    angle_with_normal: float        # rad, 0 = vertical bond
    voronoi_area: float             # Å² — shared Voronoi face area


@dataclass
class AdsorptionSite:
    site_type: str                  # 'top' | 'bridge' | 'hollow_fcc' | 'hollow_hcp'
    position: np.ndarray            # Cartesian, Å
    coordinating_atoms: List[int]   # node indices in the graph
    symmetry_rank: float            # higher ⟹ more symmetric environment
    fingerprint: str                # SHA-256 hash of (type, coord_atoms)


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class SurfaceTopologyGraph:
    """
    Voronoi-based topology graph for a heterogeneous catalyst surface.

    Construction
    ------------
    >>> stg = SurfaceTopologyGraph(positions, elements, cell)
    >>> stg.build()
    >>> sites = stg.classify_adsorption_sites()
    >>> X = stg.node_feature_matrix()   # shape (N, 6)

    Algorithm sketch
    ----------------
    1.  Identify surface layer via z-coordinate histogram.
    2.  Build Voronoi tessellation with periodic images (+/- 1 image in
        each direction) to handle PBC correctly.
    3.  Assign edges for atom pairs sharing a Voronoi face of area > ε.
    4.  Compute node features: [Z, layer, CN, V_Vor, d_surf, CN_variance].
    5.  Classify candidate adsorption sites:
        - top    : directly above each surface atom
        - bridge : midpoint of every surface edge
        - hollow : centroid of surface triangles; fcc if no subsurface atom
                   underneath, hcp if a 2nd-layer atom sits below.
    6.  Score site symmetry via the eigenvalue spread of the local
        coordination tensor  M_ij = Σ_k (r_k - r_0)_i (r_k - r_0)_j.
    """

    # Coordination-number cutoff per element (empirical, Å)
    _BOND_CUTOFFS: Dict[str, float] = {
        "H": 1.2, "C": 1.9, "N": 1.9, "O": 1.9,
        "Cu": 3.0, "Ag": 3.2, "Au": 3.2, "Pt": 3.0,
        "Pd": 3.0, "Ni": 2.8, "Fe": 3.0, "Co": 2.9,
        "Ru": 3.0, "Rh": 3.0, "Ir": 3.0, "Ti": 3.2,
    }
    _DEFAULT_CUTOFF = 3.5  # Å

    def __init__(
        self,
        positions: np.ndarray,          # (N, 3) Cartesian, Å
        elements: List[str],
        cell: np.ndarray,               # (3, 3) lattice vectors, Å
        surface_normal: Optional[np.ndarray] = None,
    ):
        assert positions.shape == (len(elements), 3), "positions/elements mismatch"
        self.positions = np.array(positions, dtype=float)
        self.elements  = list(elements)
        self.cell      = np.array(cell, dtype=float)
        self.normal    = np.array(surface_normal or [0.0, 0.0, 1.0])
        self.normal   /= np.linalg.norm(self.normal)

        self.nodes: List[AtomNode] = []
        self.edges: List[BondEdge] = []
        self._sites: Optional[List[AdsorptionSite]] = None
        self._built  = False

        # atomic numbers (Z) for common elements
        _Z = {"H":1,"He":2,"Li":3,"Be":4,"B":5,"C":6,"N":7,"O":8,"F":9,
              "Ne":10,"Na":11,"Mg":12,"Al":13,"Si":14,"P":15,"S":16,"Cl":17,
              "Ar":18,"K":19,"Ca":20,"Sc":21,"Ti":22,"V":23,"Cr":24,"Mn":25,
              "Fe":26,"Co":27,"Ni":28,"Cu":29,"Zn":30,"Ga":31,"Ge":32,"As":33,
              "Se":34,"Br":35,"Kr":36,"Rb":37,"Sr":38,"Y":39,"Zr":40,"Nb":41,
              "Mo":42,"Tc":43,"Ru":44,"Rh":45,"Pd":46,"Ag":47,"Cd":48,"In":49,
              "Sn":50,"Sb":51,"Te":52,"I":53,"Xe":54,"Cs":55,"Ba":56,"La":57,
              "Hf":72,"Ta":73,"W":74,"Re":75,"Os":76,"Ir":77,"Pt":78,"Au":79,
              "Hg":80,"Tl":81,"Pb":82,"Bi":83}
        self._Z = {e: _Z.get(e, 0) for e in elements}

    # ------------------------------------------------------------------
    # Build graph
    # ------------------------------------------------------------------

    def build(self, min_voronoi_area: float = 0.5) -> "SurfaceTopologyGraph":
        """
        Build topology graph from Voronoi tessellation.

        Parameters
        ----------
        min_voronoi_area : float
            Minimum shared Voronoi face area (Å²) to register a bond.
            Filters spurious long-range connections.
        """
        N = len(self.elements)
        z_coords = self.positions[:, 2]

        # --- Layer assignment via z-coordinate clustering ----------------
        z_sorted = np.sort(np.unique(np.round(z_coords, 1)))
        # find largest gap above midpoint of cell → surface layer boundary
        diffs = np.diff(z_sorted)
        surface_z = z_sorted[-1]                     # topmost unique z
        layer_map: Dict[int, int] = {}
        z_layers = sorted(set(np.round(z_coords, 1)), reverse=True)
        for atom_i, z in enumerate(np.round(z_coords, 1)):
            layer_map[atom_i] = z_layers.index(z)

        # --- Periodic images for Voronoi (3×3 supercell) -----------------
        image_shifts = [
            i * self.cell[0] + j * self.cell[1]
            for i in [-1, 0, 1]
            for j in [-1, 0, 1]
        ]
        extended_pos  = []
        extended_orig = []                            # index in original cell
        for shift in image_shifts:
            for idx, pos in enumerate(self.positions):
                extended_pos.append(pos + shift)
                extended_orig.append(idx)
        extended_pos = np.array(extended_pos)

        # --- Voronoi tessellation ----------------------------------------
        try:
            vor = Voronoi(extended_pos)
        except Exception:
            # Fallback: simple distance cutoff
            self._build_distance_fallback(layer_map, min_voronoi_area)
            return self

        # Accumulate Voronoi volumes and shared face areas
        volumes    = np.zeros(N)
        face_areas: Dict[Tuple[int,int], float] = {}  # (i,j) → area

        for ridge_pts, ridge_verts in zip(vor.ridge_points, vor.ridge_vertices):
            orig_i = extended_orig[ridge_pts[0]]
            orig_j = extended_orig[ridge_pts[1]]
            if orig_i == orig_j:
                continue
            if -1 in ridge_verts:
                area = 0.0          # open ridge (infinite Voronoi vertex)
            else:
                verts = vor.vertices[ridge_verts]
                try:
                    hull  = ConvexHull(verts)
                    area  = hull.volume        # ConvexHull.volume = surface area in 2D
                except Exception:
                    area = np.linalg.norm(
                        np.cross(verts[1]-verts[0], verts[-1]-verts[0])
                    ) / 2.0
            key = (min(orig_i, orig_j), max(orig_i, orig_j))
            face_areas[key] = face_areas.get(key, 0.0) + area

        # Approximate Voronoi volume from convex hull of Voronoi vertices
        for pt_idx in range(N):
            region_idx = vor.point_region[pt_idx]
            region_verts = vor.regions[region_idx]
            if -1 in region_verts or len(region_verts) < 4:
                volumes[pt_idx] = 12.0       # fallback
            else:
                try:
                    hull = ConvexHull(vor.vertices[region_verts])
                    volumes[pt_idx] = hull.volume
                except Exception:
                    volumes[pt_idx] = 12.0

        # --- Build node list ---------------------------------------------
        mean_surface_z = np.mean(self.positions[
            [i for i,l in layer_map.items() if l == 0], 2
        ])
        self.nodes = []
        for i, (elem, pos) in enumerate(zip(self.elements, self.positions)):
            self.nodes.append(AtomNode(
                index        = i,
                element      = elem,
                atomic_number= self._Z.get(elem, 0),
                position     = pos.copy(),
                layer        = layer_map[i],
                voronoi_volume = volumes[i],
                surface_dist   = pos[2] - mean_surface_z,
            ))

        # --- Build edge list and coordination numbers --------------------
        self.edges = []
        seen_edges = set()
        for (i, j), area in face_areas.items():
            if area < min_voronoi_area:
                continue
            dist = np.linalg.norm(self.positions[j] - self.positions[i])
            cutoff = max(
                self._BOND_CUTOFFS.get(self.elements[i], self._DEFAULT_CUTOFF),
                self._BOND_CUTOFFS.get(self.elements[j], self._DEFAULT_CUTOFF),
            )
            if dist > cutoff:
                continue
            bond_vec  = self.positions[j] - self.positions[i]
            cos_theta = abs(np.dot(bond_vec, self.normal)) / (dist + 1e-12)
            angle     = np.arccos(np.clip(cos_theta, 0, 1))
            self.edges.append(BondEdge(
                i=i, j=j,
                length=dist,
                angle_with_normal=angle,
                voronoi_area=area,
            ))
            self.nodes[i].coordination += 1
            self.nodes[j].coordination += 1
            seen_edges.add((i, j))

        self._built = True
        return self

    def _build_distance_fallback(self, layer_map, min_voronoi_area):
        """Distance-based fallback when Voronoi fails."""
        for i, elem_i in enumerate(self.elements):
            self.nodes.append(AtomNode(
                index=i, element=elem_i, atomic_number=self._Z.get(elem_i,0),
                position=self.positions[i].copy(), layer=layer_map[i],
            ))
        cutoff = self._DEFAULT_CUTOFF
        for i in range(len(self.elements)):
            for j in range(i+1, len(self.elements)):
                dist = np.linalg.norm(self.positions[j]-self.positions[i])
                if dist < cutoff:
                    bond_vec  = self.positions[j]-self.positions[i]
                    cos_theta = abs(np.dot(bond_vec, self.normal))/(dist+1e-12)
                    self.edges.append(BondEdge(
                        i=i, j=j, length=dist,
                        angle_with_normal=np.arccos(np.clip(cos_theta,0,1)),
                        voronoi_area=1.0,
                    ))
                    self.nodes[i].coordination += 1
                    self.nodes[j].coordination += 1
        self._built = True

    # ------------------------------------------------------------------
    # Site classification
    # ------------------------------------------------------------------

    def classify_adsorption_sites(
        self, ads_height: float = 1.8
    ) -> List[AdsorptionSite]:
        """
        Enumerate and classify adsorption sites on the top surface layer.

        Algorithm
        ---------
        Top sites    : directly above each surface atom (layer == 0).
        Bridge sites : midpoint of each surface–surface edge, shifted
                       upward by ads_height.
        Hollow sites : centroid of surface triangles formed by Delaunay
                       triangulation of the top-layer atoms.
                       fcc-hollow : no 2nd-layer atom within 1.5 Å of
                                    projected centroid.
                       hcp-hollow : a 2nd-layer atom sits within 1.5 Å.

        Symmetry score
        --------------
        For each site, build the local *coordination tensor*:
            T = Σ_k  Δr_k ⊗ Δr_k    (Δr_k = neighbour − site)
        Score = 1 / (λ_max/λ_min + ε) where λ are the 2D eigenvalues of
        T projected onto the surface plane. A perfectly symmetric site
        (3-fold or 4-fold) yields a score near 1.
        """
        assert self._built, "Call build() first."

        surface_idx = [n.index for n in self.nodes if n.layer == 0]
        sub_idx     = [n.index for n in self.nodes if n.layer == 1]
        surface_xy  = self.positions[surface_idx, :2]
        sub_xy      = self.positions[sub_idx,     :2] if sub_idx else np.empty((0,2))
        surface_z   = np.mean(self.positions[surface_idx, 2])

        sites: List[AdsorptionSite] = []

        # ---- Top sites --------------------------------------------------
        for idx in surface_idx:
            pos = self.positions[idx].copy()
            pos[2] = surface_z + ads_height
            score = self._symmetry_score(pos[:2], surface_xy)
            sites.append(AdsorptionSite(
                site_type="top",
                position=pos,
                coordinating_atoms=[idx],
                symmetry_rank=score,
                fingerprint=self._fp("top", [idx]),
            ))

        # ---- Bridge sites -----------------------------------------------
        surface_edge_pairs = [
            (e.i, e.j) for e in self.edges
            if self.nodes[e.i].layer == 0 and self.nodes[e.j].layer == 0
        ]
        for i, j in surface_edge_pairs:
            pos = (self.positions[i] + self.positions[j]) / 2.0
            pos[2] = surface_z + ads_height
            score = self._symmetry_score(pos[:2], surface_xy)
            sites.append(AdsorptionSite(
                site_type="bridge",
                position=pos,
                coordinating_atoms=sorted([i, j]),
                symmetry_rank=score,
                fingerprint=self._fp("bridge", sorted([i, j])),
            ))

        # ---- Hollow sites via Delaunay triangulation --------------------
        if len(surface_xy) >= 3:
            from scipy.spatial import Delaunay
            tri = Delaunay(surface_xy)
            for simplex in tri.simplices:
                real_idx = [surface_idx[k] for k in simplex]
                centroid = np.mean(self.positions[real_idx, :2], axis=0)
                # Check for subsurface atom below centroid
                if len(sub_xy) > 0:
                    dists = np.linalg.norm(sub_xy - centroid, axis=1)
                    has_sub = np.any(dists < 1.5)
                else:
                    has_sub = False
                htype = "hollow_hcp" if has_sub else "hollow_fcc"
                pos = np.array([centroid[0], centroid[1], surface_z + ads_height])
                score = self._symmetry_score(centroid, surface_xy)
                sites.append(AdsorptionSite(
                    site_type=htype,
                    position=pos,
                    coordinating_atoms=sorted(real_idx),
                    symmetry_rank=score,
                    fingerprint=self._fp(htype, sorted(real_idx)),
                ))

        # Deduplicate by position (within 0.3 Å)
        sites = self._deduplicate_sites(sites, tol=0.3)
        self._sites = sites
        return sites

    # ------------------------------------------------------------------
    # Feature matrices for GNN
    # ------------------------------------------------------------------

    def node_feature_matrix(self) -> np.ndarray:
        """
        Return (N, 6) node feature matrix for graph neural network input.

        Features (all normalised to O(1) range):
        0  atomic_number / 100
        1  layer index (0 = surface)
        2  coordination number / 12
        3  Voronoi volume / 20  (Å³ / 20)
        4  surface distance / 5 (Å / 5)
        5  bond angle variance (rad²)
        """
        assert self._built
        N = len(self.nodes)
        X = np.zeros((N, 6), dtype=np.float32)
        # precompute per-node angle lists
        angle_lists: Dict[int, List[float]] = {i: [] for i in range(N)}
        for e in self.edges:
            angle_lists[e.i].append(e.angle_with_normal)
            angle_lists[e.j].append(e.angle_with_normal)

        for n in self.nodes:
            angles = angle_lists[n.index]
            X[n.index] = [
                n.atomic_number / 100.0,
                float(n.layer),
                n.coordination / 12.0,
                n.voronoi_volume / 20.0,
                n.surface_dist / 5.0,
                float(np.var(angles)) if angles else 0.0,
            ]
        return X

    def edge_index_and_attr(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (edge_index, edge_attr) for PyTorch Geometric.

        edge_index : (2, 2E) — bidirectional
        edge_attr  : (2E, 3) — [length/5, angle/π, voronoi_area/10]
        """
        assert self._built
        src, dst, attrs = [], [], []
        for e in self.edges:
            for (a, b) in [(e.i, e.j), (e.j, e.i)]:
                src.append(a)
                dst.append(b)
                attrs.append([
                    e.length / 5.0,
                    e.angle_with_normal / np.pi,
                    e.voronoi_area / 10.0,
                ])
        return (
            np.array([src, dst], dtype=np.int64),
            np.array(attrs, dtype=np.float32),
        )

    def to_networkx(self):
        """Export as a NetworkX DiGraph (requires networkx)."""
        if not _HAS_NX:
            raise ImportError("networkx required: pip install networkx")
        G = nx.DiGraph()
        X = self.node_feature_matrix()
        for n in self.nodes:
            G.add_node(n.index, features=X[n.index].tolist(),
                       element=n.element, layer=n.layer)
        ei, ea = self.edge_index_and_attr()
        for k in range(ei.shape[1]):
            G.add_edge(int(ei[0,k]), int(ei[1,k]),
                       attr=ea[k].tolist())
        return G

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _symmetry_score(site_xy: np.ndarray, neighbor_xy: np.ndarray,
                         radius: float = 3.5) -> float:
        """
        Local coordination tensor eigenvalue symmetry score.
        Score ∈ (0, 1], where 1 = perfectly symmetric.
        """
        dists = np.linalg.norm(neighbor_xy - site_xy, axis=1)
        near  = neighbor_xy[dists < radius] - site_xy
        if len(near) < 2:
            return 0.0
        T = near.T @ near          # 2×2 coordination tensor
        eigvals = np.linalg.eigvalsh(T)
        eigvals = np.sort(eigvals)[::-1]
        if eigvals[0] < 1e-10:
            return 0.0
        return float(eigvals[-1] / (eigvals[0] + 1e-10))

    @staticmethod
    def _fp(site_type: str, atoms: List[int]) -> str:
        s = f"{site_type}:{'_'.join(map(str, sorted(atoms)))}"
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    @staticmethod
    def _deduplicate_sites(sites: List[AdsorptionSite],
                            tol: float = 0.3) -> List[AdsorptionSite]:
        kept = []
        for s in sites:
            duplicate = False
            for k in kept:
                if np.linalg.norm(s.position - k.position) < tol:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(s)
        return kept

    def summary(self) -> str:
        lines = [
            f"SurfaceTopologyGraph — {len(self.nodes)} atoms, "
            f"{len(self.edges)} bonds",
            f"  Elements : {sorted(set(n.element for n in self.nodes))}",
            f"  Layers   : {max(n.layer for n in self.nodes)+1}",
        ]
        if self._sites is not None:
            from collections import Counter
            counts = Counter(s.site_type for s in self._sites)
            lines.append(f"  Sites    : {dict(counts)}")
        return "\n".join(lines)
