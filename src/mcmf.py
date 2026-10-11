"""
mcmf.py
From-scratch implementation of Min-Cost Max-Flow (MCMF) using:
1. Successive Shortest Path (SSP) algorithm
2. Bellman-Ford / SPFA to compute initial node potentials pi for general costs (supporting negative costs without negative cycles)
3. Priority-Queue Dijkstra with node potentials pi to maintain reduced costs c_pi >= 0, achieving O(F * (M + N log N)) runtime.

Zero external graph library dependencies.
"""

import heapq
from typing import List, Tuple, Optional


class Edge:
    __slots__ = ('u', 'v', 'cap', 'flow', 'cost', 'rev', 'edge_id', 'meta')

    def __init__(self, u: int, v: int, cap: int, cost: int, rev: int, edge_id: int = -1, meta: dict = None):
        self.u = u
        self.v = v
        self.cap = cap
        self.flow = 0
        self.cost = cost
        self.rev = rev
        self.edge_id = edge_id
        self.meta = meta or {}

    @property
    def residual_capacity(self) -> int:
        return self.cap - self.flow


class MinCostMaxFlow:
    def __init__(self, n: int):
        self.n = n
        self.adj: List[List[Edge]] = [[] for _ in range(n)]
        self.edges: List[Edge] = []
        self.edge_count = 0

    def add_edge(self, u: int, v: int, cap: int, cost: int, meta: dict = None) -> Edge:
        """
        Add directed edge u -> v with given capacity and cost.
        Also creates residual back-edge v -> u with capacity 0 and cost -cost.
        """
        forward = Edge(u, v, cap, cost, rev=len(self.adj[v]), edge_id=self.edge_count, meta=meta)
        backward = Edge(v, u, 0, -cost, rev=len(self.adj[u]), edge_id=-1, meta=None)
        self.adj[u].append(forward)
        self.adj[v].append(backward)
        self.edges.append(forward)
        self.edge_count += 1
        return forward

    def spfa_potentials(self, s: int) -> Tuple[bool, List[int]]:
        """
        Compute initial potentials using Shortest Path Faster Algorithm (SPFA / queue-based Bellman-Ford).
        Returns (success, pi) where success is False if a negative cycle is detected.
        """
        INF = 10**15
        pi = [INF] * self.n
        in_queue = [False] * self.n
        count = [0] * self.n
        queue = []

        pi[s] = 0
        queue.append(s)
        in_queue[s] = True
        count[s] = 1

        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            in_queue[u] = False

            for edge in self.adj[u]:
                if edge.residual_capacity > 0 and pi[u] + edge.cost < pi[edge.v]:
                    pi[edge.v] = pi[u] + edge.cost
                    if not in_queue[edge.v]:
                        queue.append(edge.v)
                        in_queue[edge.v] = True
                        count[edge.v] += 1
                        if count[edge.v] > self.n:
                            # Negative cycle detected
                            return False, []
        
        # Replace INF with 0 for unreachable nodes so reduced costs stay defined
        for i in range(self.n):
            if pi[i] >= INF // 2:
                pi[i] = 0
        return True, pi

    def solve(self, s: int, t: int, max_flow_target: Optional[int] = None) -> Tuple[int, int]:
        """
        Compute Min-Cost Max-Flow from s to t.
        If max_flow_target is specified, pushes at most max_flow_target units of flow.
        Returns: (total_flow, total_cost)
        """
        INF = 10**15

        # Step 1: Initial potentials using SPFA
        has_no_neg_cycle, pi = self.spfa_potentials(s)
        if not has_no_neg_cycle:
            raise ValueError("Graph contains a negative cost cycle reachable from source!")

        total_flow = 0
        total_cost = 0

        # Step 2: Successive Shortest Path using Dijkstra with reduced costs
        while max_flow_target is None or total_flow < max_flow_target:
            dist = [INF] * self.n
            parent_edge: List[Optional[Edge]] = [None] * self.n
            
            dist[s] = 0
            pq = [(0, s)]  # (reduced_cost_dist, u)

            while pq:
                d, u = heapq.heappop(pq)
                if d > dist[u]:
                    continue

                for edge in self.adj[u]:
                    if edge.residual_capacity > 0:
                        # Reduced cost: c_pi(u, v) = c(u, v) + pi[u] - pi[v] >= 0
                        reduced_cost = edge.cost + pi[u] - pi[edge.v]
                        if dist[u] + reduced_cost < dist[edge.v]:
                            dist[edge.v] = dist[u] + reduced_cost
                            parent_edge[edge.v] = edge
                            heapq.heappush(pq, (dist[edge.v], edge.v))

            if dist[t] >= INF // 2:
                # No more augmenting path
                break

            # Update node potentials: pi'[u] = pi[u] + dist[u]
            for i in range(self.n):
                if dist[i] < INF // 2:
                    pi[i] += dist[i]

            # Find bottleneck capacity along shortest path
            push_amount = INF if max_flow_target is None else (max_flow_target - total_flow)
            curr = t
            while curr != s:
                edge = parent_edge[curr]
                push_amount = min(push_amount, edge.residual_capacity)
                curr = edge.u

            if push_amount <= 0:
                break

            # Augment flow along the path
            curr = t
            path_cost = 0
            while curr != s:
                edge = parent_edge[curr]
                edge.flow += push_amount
                self.adj[edge.v][edge.rev].flow -= push_amount
                path_cost += edge.cost
                curr = edge.u

            total_flow += push_amount
            total_cost += push_amount * path_cost

        return total_flow, total_cost
