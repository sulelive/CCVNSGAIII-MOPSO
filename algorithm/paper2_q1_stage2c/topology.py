
"""
Compatibility topology module for the Paper 2 experiment.

The reconstructed Paper 2 experiment uses SYSTEMMODEL.FGCSmodel3()
with NetworkX-generated Barabasi-Albert topologies.

The original SYSTEMMODEL.py imports Entity and Topology even though
FGCSmodel3() does not require them. These minimal definitions preserve
that import without changing the original SYSTEMMODEL.py.

FGCSmodel2() is NOT supported by this compatibility module.
"""

import networkx as nx


class Entity:
    ENTITY_CLUSTER = "CLOUD"
    ENTITY_FOG = "FOG"
    ENTITY_DEVICE = "DEVICE"


class Topology:
    def __init__(self):
        self.G = nx.Graph()

    def load(self, data):
        raise NotImplementedError(
            "The compatibility topology.py is intended only for "
            "SYSTEMMODEL.FGCSmodel3(). FGCSmodel2() is not used "
            "in the Paper 2 experiment."
        )
