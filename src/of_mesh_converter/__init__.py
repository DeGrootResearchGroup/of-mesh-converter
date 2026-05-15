"""of-mesh-converter: Fluent CGNS -> OpenFOAM case skeleton for radiationDose."""

__version__ = "0.1.0"

from .mesh_ir import CaseData, Mesh, Patch
from .pipeline import convert

__all__ = ["CaseData", "Mesh", "Patch", "convert", "__version__"]
