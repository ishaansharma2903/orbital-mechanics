"""
bridges

Bridges to external mission design and numerical propagation tools (e.g. NASA GMAT).
"""
from polaris.bridges.gmat import export_to_gmat_script

__all__ = [
    "export_to_gmat_script",
]
