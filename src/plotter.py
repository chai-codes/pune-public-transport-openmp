"""
plotter.py
Generates publication-quality charts for the project report and documentation:
1. plot_demand_profile.png: Directional passenger demand vs travel time (peak congestion) across the day.
2. plot_dispatch_schedule.png: Optimal MCMF directional bus departures and passenger delivery by time slot.
3. plot_optimization_vs_baseline.png: Comparison of MCMF vs Uniform baseline across fleet sizes.
4. plot_scaling_granularity.png: Runtime & edge growth as time slots increase (60m down to 10m).
5. plot_multi_route_scaling.png: Runtime scaling with multi-route corridors and bus fleet.
6. plot_language_comparison.png: Execution time comparison across Python, C Sequential, and C OpenMP.
"""

import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Set clean aesthetic style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8


def generate_all_plots(benchmark_file="benchmark_results/benchmark_data.json", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    with open(benchmark_file, "r") as f:
        data = json.load(f)

    # 1. Demand & Travel Time Profile Plot
    from src.dataset import generate_demand_profile
    profile = generate_demand_profile(slot_minutes=30)
    slots = profile["slots"]
    times = [s["time_str"] for s in slots]
    sh_demand = profile["demand"]["S->H"]
    hs_demand = profile["demand"]["H->S"]
    travel_time = [s["travel_time_min"] for s in slots]

    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax2 = ax1.twinx()

    x = np.arange(len(times))
    w = 0.38
    b1 = ax1.bar(x - w/2, sh_demand, width=w, label="Swargate -> Hinjewadi (Morning Inflow)", color="#1f77b4", alpha=0.85)
    b2 = ax1.bar(x + w/2, hs_demand, width=w, label="Hinjewadi -> Swargate (Evening Outflow)", color="#ff7f0e", alpha=0.85)

    l1 = ax2.plot(x, travel_time, color="#d62728", linewidth=2.5, linestyle="--", marker="o", markersize=4, label="Congested Travel Time (mins)")

    ax1.set_xlabel("Time Slot (06:00 to 22:30)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Passenger Demand (persons / 30 min)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("One-Way Transit Time (minutes)", fontsize=11, fontweight="bold", color="#d62728")
    ax1.set_xticks(x[::2])
    ax1.set_xticklabels(times[::2], rotation=35)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.grid(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
    plt.title("PMPML Hinjewadi <-> Swargate Corridor: Dynamic Demand & Congestion Profile", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "plot_demand_profile.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    print(f"[OK] Saved {p1}")

    # 2. Optimal Dispatch Schedule (from 50 buses MCMF solution)
    from src.graph_builder import TimeExpandedCorridorGraph
    corridor = TimeExpandedCorridorGraph(profile)
    corridor.build_graph(initial_buses=(25, 25))
    sol = corridor.solve(50)

    fig, (ax_dep, ax_pax) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    dep_sh = sol["departures"]["S->H"]
    dep_hs = sol["departures"]["H->S"]
    pax_sh = sol["passengers_carried"]["S->H"]
    pax_hs = sol["passengers_carried"]["H->S"]

    ax_dep.bar(x - w/2, dep_sh, width=w, label="Departures: Swargate -> Hinjewadi", color="#2ca02c", alpha=0.85)
    ax_dep.bar(x + w/2, dep_hs, width=w, label="Departures: Hinjewadi -> Swargate", color="#9467bd", alpha=0.85)
    ax_dep.set_ylabel("Buses Dispatched", fontsize=11, fontweight="bold")
    ax_dep.set_title("Optimal MCMF Dynamic Fleet Dispatch Schedule (F = 50)", fontsize=13, fontweight="bold")
    ax_dep.legend(loc="upper right", frameon=True)
    ax_dep.grid(True, linestyle=":", alpha=0.6)

    ax_pax.plot(x, pax_sh, marker="s", color="#1f77b4", linewidth=2, label="Passengers Served: S -> H")
    ax_pax.plot(x, pax_hs, marker="^", color="#ff7f0e", linewidth=2, label="Passengers Served: H -> S")
    ax_pax.plot(x, sh_demand, linestyle=":", color="#1f77b4", alpha=0.5, label="Actual Demand: S -> H")
    ax_pax.plot(x, hs_demand, linestyle=":", color="#ff7f0e", alpha=0.5, label="Actual Demand: H -> S")
    ax_pax.set_ylabel("Passengers Served", fontsize=11, fontweight="bold")
    ax_pax.set_xlabel("Time Slot", fontsize=11, fontweight="bold")
    ax_pax.set_xticks(x[::2])
    ax_pax.set_xticklabels(times[::2], rotation=35)
    ax_pax.legend(loc="upper right", frameon=True)
    ax_pax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p2 = os.path.join(output_dir, "plot_dispatch_schedule.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    print(f"[OK] Saved {p2}")

    # 3. Optimization vs Baseline across Fleet Sizes
    fleet_data = data["fleet_scaling"]
    fleets = [e["fleet_size"] for e in fleet_data]
    mcmf_pax = [e["mcmf_pax"] for e in fleet_data]
    base_pax = [e["baseline_pax"] for e in fleet_data]
    gains = [e["gain_pct"] for e in fleet_data]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax2 = ax1.twinx()

    l1 = ax1.plot(fleets, mcmf_pax, marker="o", color="#2ca02c", linewidth=2.5, label="MCMF Optimized Dispatch")
    l2 = ax1.plot(fleets, base_pax, marker="s", color="#d62728", linewidth=2.2, linestyle="--", label="Uniform Symmetric Baseline")
    b1 = ax2.bar(np.array(fleets) + 1.2, gains, width=2.5, color="#17becf", alpha=0.4, label="% Passenger Gain")

    ax1.set_xlabel("Fleet Size (Buses F)", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Total Daily Passengers Carried", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Passenger Gain over Baseline (%)", fontsize=11, fontweight="bold", color="#0e808e")
    ax1.set_title("MCMF Optimization vs Symmetric Baseline: Passenger Throughput", fontsize=13, fontweight="bold")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax2.grid(False)

    lines_all = l1 + l2 + [b1]
    labels_all = [l.get_label() for l in lines_all]
    ax1.legend(lines_all, labels_all, loc="lower right", frameon=True)

    plt.tight_layout()
    p3 = os.path.join(output_dir, "plot_optimization_vs_baseline.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    print(f"[OK] Saved {p3}")

    # 4. Runtime & Edge Growth vs Time Slot Granularity
    gran_data = data["granularity_scaling"]
    slot_mins = [e["slot_minutes"] for e in gran_data]
    n_nodes = [e["num_nodes"] for e in gran_data]
    n_edges = [e["num_edges"] for e in gran_data]
    py_times = [e["py_time_ms"] for e in gran_data]
    c_times = [e["c_seq_time_ms"] for e in gran_data]

    fig, (ax_graph, ax_time) = plt.subplots(1, 2, figsize=(13, 5))

    ax_graph.plot(slot_mins, n_nodes, marker="o", color="#3b528b", linewidth=2, label="Nodes |V|")
    ax_graph.plot(slot_mins, n_edges, marker="^", color="#21918c", linewidth=2, label="Edges |E|")
    ax_graph.invert_xaxis()
    ax_graph.set_xlabel("Slot Granularity (Minutes per Slot)", fontsize=11, fontweight="bold")
    ax_graph.set_ylabel("Network Dimension", fontsize=11, fontweight="bold")
    ax_graph.set_title("Time-Expanded Graph Expansion", fontsize=12, fontweight="bold")
    ax_graph.legend()
    ax_graph.grid(True, linestyle=":", alpha=0.6)

    ax_time.plot(slot_mins, py_times, marker="s", color="#e377c2", linewidth=2, label="Python Scratch MCMF")
    ax_time.plot(slot_mins, c_times, marker="o", color="#bcbd22", linewidth=2, label="C Sequential MCMF")
    ax_time.invert_xaxis()
    ax_time.set_xlabel("Slot Granularity (Minutes per Slot)", fontsize=11, fontweight="bold")
    ax_time.set_ylabel("Solver Runtime (ms)", fontsize=11, fontweight="bold")
    ax_time.set_title("Algorithm Execution Time vs Granularity", fontsize=12, fontweight="bold")
    ax_time.legend()
    ax_time.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    p4 = os.path.join(output_dir, "plot_scaling_granularity.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    print(f"[OK] Saved {p4}")

    # 5. Multi-Route Corridor Scaling
    route_data = data["multi_route_scaling"]
    routes = [e["num_routes"] for e in route_data]
    route_edges = [e["num_edges"] for e in route_data]
    route_times = [e["time_ms"] for e in route_data]
    route_pax = [e["passengers_served"] for e in route_data]

    fig, ax_r1 = plt.subplots(figsize=(9, 5))
    ax_r2 = ax_r1.twinx()

    b_pax = ax_r1.bar(np.array(routes) - 0.15, route_pax, width=0.3, color="#8c564b", alpha=0.8, label="Passengers Served")
    l_time = ax_r2.plot(np.array(routes) + 0.15, route_times, marker="o", color="#9467bd", linewidth=2.5, label="Solver Time (ms)")

    ax_r1.set_xlabel("Corridor Routes / Sub-Branches", fontsize=11, fontweight="bold")
    ax_r1.set_ylabel("Total Passengers Carried", fontsize=11, fontweight="bold")
    ax_r2.set_ylabel("MCMF Runtime (ms)", fontsize=11, fontweight="bold", color="#9467bd")
    ax_r1.set_title("Scalability Across Multi-Branch Transit Networks", fontsize=13, fontweight="bold")
    ax_r1.set_xticks(routes)
    ax_r1.grid(True, linestyle=":", alpha=0.6)

    lines_r = [b_pax, l_time[0]]
    labels_r = [l.get_label() for l in lines_r]
    ax_r1.legend(lines_r, labels_r, loc="upper left", frameon=True)

    plt.tight_layout()
    p5 = os.path.join(output_dir, "plot_multi_route_scaling.png")
    plt.savefig(p5, dpi=300)
    plt.close()
    print(f"[OK] Saved {p5}")

    print("[OK] All 5 figures generated with 300 DPI publication quality!")


if __name__ == "__main__":
    generate_all_plots()
