from gread_core.detectors.base import DetectorProtocol
from gread_core.detectors.bwgnn import BWGNNDetector
from gread_core.detectors.caregnn import CAREGNNDetector
from gread_core.detectors.pyg_gnn import GATDetector, GCNDetector
from gread_core.detectors.tree_neighbor import TreeNeighborDetector

__all__ = [
    "BWGNNDetector",
    "CAREGNNDetector",
    "DetectorProtocol",
    "GATDetector",
    "GCNDetector",
    "TreeNeighborDetector",
]
