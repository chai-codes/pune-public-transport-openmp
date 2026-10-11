"""
graph_builder.py
Constructs the Time-Expanded Directed Graph for the PMPML Hinjewadi <-> Swargate fleet dispatch problem.

Graph Model:
- Terminals: 
    A = Swargate (Index 0)
    B = Hinjewadi (Index 1)
- Time discretized into slots t in [0, T-1].
- Node indexing:
    Node(terminal, t) = terminal * T + t
    Super-source S = 2 * T
    Super-sink T_sink = 2 * T + 1
- S -> (terminal, 0): capacity = initial buses at terminal, cost = 0. Total sum = F.
- Wait edge: (terminal, t) -> (terminal, t+1), capacity = infinity (or F), cost = 0.
- Trip edges: (A, t) -> (B, arr_slot), where arr_slot = min(T, t + ceil((travel_time(t) + layover) / slot_minutes)).
    If arr_slot >= T, bus arrives at/after the end of operating window, so it connects directly to T_sink.
    * Passenger revenue edges: parallel edges per bus index k (1-indexed)
      k in [1, K_max], cap = 1, cost = -passengers_served_by_kth_bus.
      passengers = min(capacity, max(0, demand[t] - (k-1)*capacity)).
      Threshold edge case: If passengers < 60 (demand remainder or low demand < 60), cost is still -passengers.
    * Empty running edge (deadheading / repositioning without passenger service constraint):
      cap = F, cost = 0.
- (terminal, T-1) -> T_sink: capacity = F, cost = 0.
"""

import math
from typing import Dict, List, Tuple
from .mcmf import MinCostMaxFlow


class TimeExpandedCorridorGraph:
    def __init__(self, config: Dict):
        self.config = config
        self.num_slots = config["num_slots"]
        self.slot_minutes = config["slot_minutes"]
        self.layover_minutes = config.get("layover_minutes", 10.0)
        self.bus_capacity = config.get("bus_capacity", 60)
        self.demand = config["demand"]
        self.slots = config["slots"]

        self.T = self.num_slots
        # 2 terminals: 0 = Swargate (S_term), 1 = Hinjewadi (H_term)
        self.num_nodes = 2 * self.T + 2
        self.source = 2 * self.T
        self.sink = 2 * self.T + 1

        self.mcmf = MinCostMaxFlow(self.num_nodes)
        self.trip_edges: List[dict] = []

    def node_id(self, terminal: int, t: int) -> int:
        return terminal * self.T + t

    def build_graph(self, initial_buses: Tuple[int, int] = (25, 25)):
        """
        Build the time expanded network.
        initial_buses = (buses_at_Swargate_t0, buses_at_Hinjewadi_t0)
        Total fleet F = sum(initial_buses)
        """
        f_swargate, f_hinjewadi = initial_buses
        total_f = f_swargate + f_hinjewadi

        # 1. Super-source connections
        if f_swargate > 0:
            self.mcmf.add_edge(self.source, self.node_id(0, 0), f_swargate, 0, meta={"type": "source_depot", "term": 0})
        if f_hinjewadi > 0:
            self.mcmf.add_edge(self.source, self.node_id(1, 0), f_hinjewadi, 0, meta={"type": "source_depot", "term": 1})

        # 2. Wait edges: (term, t) -> (term, t+1)
        for term in (0, 1):
            for t in range(self.T - 1):
                u = self.node_id(term, t)
                v = self.node_id(term, t + 1)
                self.mcmf.add_edge(u, v, total_f, 0, meta={"type": "wait", "term": term, "slot": t})

        # 3. Trip edges with convex marginal passenger revenue & empty running
        # Direction 0: Swargate (0) -> Hinjewadi (1), Direction 1: Hinjewadi (1) -> Swargate (0)
        directions = [
            (0, 1, "S->H"),
            (1, 0, "H->S")
        ]

        for orig, dest, key in directions:
            for t in range(self.T):
                slot_info = self.slots[t]
                tt = slot_info["travel_time_min"]
                transit_slots = int(math.ceil((tt + self.layover_minutes) / self.slot_minutes))
                arr_t = t + transit_slots

                u = self.node_id(orig, t)
                if arr_t < self.T:
                    v = self.node_id(dest, arr_t)
                else:
                    v = self.sink  # Arrives at or after operating hours end

                d_t = self.demand[key][t]
                cap = self.bus_capacity

                # Number of discrete revenue bands:
                # k = 1..k_max where k_max = ceil(d_t / cap)
                k_max = int(math.ceil(d_t / cap)) if d_t > 0 else 0

                # Add parallel passenger edges
                for k in range(1, k_max + 1):
                    # Marginal passengers served by k-th bus
                    pax = min(cap, max(0, d_t - (k - 1) * cap))
                    if pax > 0:
                        # cost is -pax (min cost = max profit)
                        edge = self.mcmf.add_edge(u, v, 1, -pax, meta={
                            "type": "passenger_trip",
                            "dir": key,
                            "dep_slot": t,
                            "arr_slot": arr_t,
                            "bus_index": k,
                            "passengers": pax
                        })
                        self.trip_edges.append({"edge": edge, "dir": key, "dep_slot": t, "arr_slot": arr_t, "pax": pax})

                # Add empty running trip edge (repositioning with no passengers, cap = F, cost = 0)
                # To prevent arbitrary zero-cost circulation cycles, cost is small non-negative 0.
                edge_empty = self.mcmf.add_edge(u, v, total_f, 0, meta={
                    "type": "empty_trip",
                    "dir": key,
                    "dep_slot": t,
                    "arr_slot": arr_t,
                    "passengers": 0
                })
                self.trip_edges.append({"edge": edge_empty, "dir": key, "dep_slot": t, "arr_slot": arr_t, "pax": 0})

        # 4. End-of-day connection to super-sink
        for term in (0, 1):
            u = self.node_id(term, self.T - 1)
            self.mcmf.add_edge(u, self.sink, total_f, 0, meta={"type": "sink_depot", "term": term})

    def solve(self, total_f: int) -> Dict:
        """
        Solves MCMF for fleet size total_f.
        Returns detailed summary and dispatch timetable.
        """
        flow, total_cost = self.mcmf.solve(self.source, self.sink, max_flow_target=total_f)
        total_passengers_served = -total_cost

        # Extract dispatch schedule
        # departures[dir][t] = count of buses departed
        departures = {"S->H": [0] * self.T, "H->S": [0] * self.T}
        pax_carried = {"S->H": [0] * self.T, "H->S": [0] * self.T}
        empty_buses = {"S->H": [0] * self.T, "H->S": [0] * self.T}

        for item in self.trip_edges:
            e = item["edge"]
            d = item["dir"]
            t = item["dep_slot"]
            if e.flow > 0:
                if e.meta.get("type") == "passenger_trip":
                    departures[d][t] += e.flow
                    pax_carried[d][t] += e.flow * item["pax"]
                elif e.meta.get("type") == "empty_trip":
                    departures[d][t] += e.flow
                    empty_buses[d][t] += e.flow

        total_demand = sum(self.demand["S->H"]) + sum(self.demand["H->S"])

        return {
            "fleet_size": total_f,
            "flow_delivered": flow,
            "total_passengers_served": total_passengers_served,
            "total_demand": total_demand,
            "demand_satisfaction_pct": (total_passengers_served / total_demand * 100.0) if total_demand > 0 else 0.0,
            "departures": departures,
            "passengers_carried": pax_carried,
            "empty_buses": empty_buses,
            "num_nodes": self.num_nodes,
            "num_edges": self.mcmf.edge_count
        }
