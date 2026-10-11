"""
export_graph.py
Utility to export python TimeExpandedCorridorGraph instance into standard text format
consumable by C binaries (both sequential and parallel OpenMP).
Format:
<num_nodes> <num_edges> <source> <sink> <max_flow>
u v cap cost
...
"""

import json
from src.dataset import generate_demand_profile
from src.graph_builder import TimeExpandedCorridorGraph


def export_corridor_to_file(filepath: str, slot_minutes: int = 30, fleet_size: int = 50, initial_buses=(25, 25)):
    data = generate_demand_profile(slot_minutes=slot_minutes)
    corridor = TimeExpandedCorridorGraph(data)
    corridor.build_graph(initial_buses=initial_buses)

    edges = corridor.mcmf.edges
    with open(filepath, "w") as f:
        f.write(f"{corridor.num_nodes} {len(edges)} {corridor.source} {corridor.sink} {fleet_size}\n")
        for e in edges:
            f.write(f"{e.u} {e.v} {e.cap} {e.cost}\n")

    return {
        "num_nodes": corridor.num_nodes,
        "num_edges": len(edges),
        "source": corridor.source,
        "sink": corridor.sink,
        "fleet_size": fleet_size,
        "filepath": filepath
    }


if __name__ == "__main__":
    info = export_corridor_to_file("data/graph_default.txt", slot_minutes=30, fleet_size=50)
    print(f"Exported graph: {info['num_nodes']} nodes, {info['num_edges']} edges -> {info['filepath']}")
