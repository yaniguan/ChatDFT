# chatdft — Python SDK for scientists
# Usage:
#   from chatdft import ChatDFT
#   dft = ChatDFT()
#   result = dft.run("CO adsorption on Pt(111)")
#   print(result.poscar)
#   print(result.incar)

from chatdft.sdk import ChatDFT, DFTResult

__all__ = ["ChatDFT", "DFTResult"]
__version__ = "0.4.0"
