"""Compatibility imports; each baseline implementation has its own directory."""

from .clam_sb import TorchMILCLAMSB
from .dsmil import TorchMILDSMIL
from .transmil import TorchMILTransMIL

__all__ = ["TorchMILCLAMSB", "TorchMILDSMIL", "TorchMILTransMIL"]
