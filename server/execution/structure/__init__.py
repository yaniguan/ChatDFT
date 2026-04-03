"""
Structure agent — split into sub-modules for maintainability.

Backward-compatible: all public functions are re-exported here.
"""

from server.execution.structure.builder import (
    build_surface_ase,
    build_molecule_pubchem,
    build_interface,
    generate_neb_images,
    slab_add_layer,
    slab_delete_layer,
    slab_set_vacuum,
    slab_dope,
    slab_make_symmetric,
    build_complex,
)

from server.execution.structure.sites import (
    find_adsorption_sites_ase,
    place_adsorbate,
    generate_configurations,
    generate_ads_from_poscars,
)

from server.execution.structure.io import (
    atoms_to_plot_b64,
    atoms_to_viz_json,
    write_structure_files,
    describe_surface,
    describe_molecule,
    describe_adsorption,
    ase_code_surface,
    ase_code_molecule,
    ase_code_adsorption,
)

from server.execution.structure.utils import (
    _normalize_element,
    _parse_miller,
    _parse_supercell,
    _guess_crystal_system,
    _poscar_to_atoms,
    _atoms_to_poscar,
    _detect_layers,
    _result,
)
