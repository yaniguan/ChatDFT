"""Structure builder functions — re-exported from structure_agent.py"""
from server.execution.structure_agent import (
    build_surface_ase,
    build_molecule_pubchem,
    build_interface,
    generate_neb_images,
    slab_add_layer,
    slab_delete_layer,
    slab_set_vacuum,
    slab_dope,
    slab_make_symmetric,
)
# build_complex is a method on StructureAgent; export the class
from server.execution.structure_agent import StructureAgent
def build_complex(*args, **kwargs):
    return StructureAgent().build_complex(*args, **kwargs) if hasattr(StructureAgent, 'build_complex') else {}
