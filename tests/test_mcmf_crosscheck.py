"""
test_mcmf_crosscheck.py
Validation test cross-checking our scratch MinCostMaxFlow against Google OR-Tools SimpleMinCostFlow.
Verifies:
1. Small random DAG networks
2. Small network with negative costs
3. Complete corridor time-expanded graph test
"""

import random
from ortools.graph.python import min_cost_flow
from src.mcmf import MinCostMaxFlow
from src.dataset import generate_demand_profile
from src.graph_builder import TimeExpandedCorridorGraph


def test_random_crosscheck():
    print("[1/3] Running randomized cross-check against Google OR-Tools...")
    for seed in range(5):
        random.seed(seed)
        num_nodes = 8
        s = 0
        t = num_nodes - 1
        
        edges = []
        for u in range(num_nodes):
            for v in range(u + 1, num_nodes):
                if random.random() < 0.45:
                    cap = random.randint(1, 10)
                    cost = random.randint(1, 20)
                    edges.append((u, v, cap, cost))

        # Build scratch solver
        scratch = MinCostMaxFlow(num_nodes)
        # Build OR-Tools solver
        ort = min_cost_flow.SimpleMinCostFlow()

        for u, v, cap, cost in edges:
            scratch.add_edge(u, v, cap, cost)
            ort.add_arc_with_capacity_and_unit_cost(u, v, cap, cost)

        # First find max possible flow using scratch
        scratch_flow, scratch_cost = scratch.solve(s, t)

        if scratch_flow > 0:
            ort.set_node_supply(s, scratch_flow)
            ort.set_node_supply(t, -scratch_flow)
            status = ort.solve()
            assert status == ort.OPTIMAL, f"OR-Tools failed to solve optimal: {status}"
            ort_cost = ort.optimal_cost()
            assert scratch_cost == ort_cost, f"Mismatch! Scratch={scratch_cost}, OR-Tools={ort_cost}"
    print("      [OK] Randomized cross-check passed!")


def test_negative_cost_crosscheck():
    print("[2/3] Running negative cost cross-check against Google OR-Tools...")
    # Directed acyclic graph with negative costs (as used in profit maximization)
    num_nodes = 6
    scratch = MinCostMaxFlow(num_nodes)
    ort = min_cost_flow.SimpleMinCostFlow()

    # edges: (u, v, cap, cost)
    test_edges = [
        (0, 1, 5, -20),
        (0, 2, 4, -15),
        (1, 3, 3, -10),
        (1, 4, 4, -5),
        (2, 3, 2, -18),
        (2, 4, 3, -8),
        (3, 5, 6, -2),
        (4, 5, 5, -1)
    ]

    for u, v, cap, cost in test_edges:
        scratch.add_edge(u, v, cap, cost)
        ort.add_arc_with_capacity_and_unit_cost(u, v, cap, cost)

    target_flow = 7
    scratch_flow, scratch_cost = scratch.solve(0, 5, max_flow_target=target_flow)
    
    ort.set_node_supply(0, target_flow)
    ort.set_node_supply(5, -target_flow)
    status = ort.solve()
    assert status == ort.OPTIMAL, f"OR-Tools status: {status}"
    ort_cost = ort.optimal_cost()

    assert scratch_flow == target_flow, f"Scratch flow {scratch_flow} != {target_flow}"
    assert scratch_cost == ort_cost, f"Scratch cost {scratch_cost} != OR-Tools cost {ort_cost}"
    print(f"      [OK] Negative cost cross-check passed! Flow={scratch_flow}, Min Cost={scratch_cost}")


def test_corridor_graph_crosscheck():
    print("[3/3] Running full Time-Expanded Corridor Graph cross-check...")
    data = generate_demand_profile(slot_minutes=60) # coarser slots for fast verification
    corridor = TimeExpandedCorridorGraph(data)
    fleet_size = 20
    corridor.build_graph(initial_buses=(10, 10))

    # Solve with scratch solver
    res_scratch = corridor.solve(fleet_size)

    # Re-build exact equivalent in OR-Tools
    ort = min_cost_flow.SimpleMinCostFlow()
    for e in corridor.mcmf.edges:
        ort.add_arc_with_capacity_and_unit_cost(e.u, e.v, e.cap, e.cost)

    ort.set_node_supply(corridor.source, fleet_size)
    ort.set_node_supply(corridor.sink, -fleet_size)
    status = ort.solve()
    assert status == ort.OPTIMAL, f"OR-Tools failed: {status}"
    ort_cost = ort.optimal_cost()

    scratch_cost = -res_scratch["total_passengers_served"]
    print(f"      Scratch total passengers: {res_scratch['total_passengers_served']} (Cost: {scratch_cost})")
    print(f"      OR-Tools total cost: {ort_cost}")
    assert scratch_cost == ort_cost, f"Corridor graph mismatch: scratch={scratch_cost}, ort={ort_cost}"
    print("      [OK] Corridor graph cross-check verified exact match with Google OR-Tools!")


if __name__ == "__main__":
    test_random_crosscheck()
    test_negative_cost_crosscheck()
    test_corridor_graph_crosscheck()
    print("\nALL UNIT AND CROSS-CHECK TESTS PASSED SUCCESSFULLY!")
