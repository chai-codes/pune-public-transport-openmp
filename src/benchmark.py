"""
benchmark.py
Comprehensive experimental benchmark suite:
1. Scaling with Time Slots (discretization granularity: 60m, 30m, 20m, 15m, 10m, 5m).
2. Scaling with Fleet Size (F = 10, 20, 30, 40, 50, 60, 80, 100).
3. Scaling across Multi-Routes / Expanded Corridors (1, 2, 3, 4 parallel sub-branches).
4. Optimization Gains: Directional MCMF vs Uniform Baseline.
5. Runtime Performance Comparison:
   - Python From-Scratch MCMF
   - C Sequential
   - C OpenMP Parallel (1, 2, 4 threads)
Saves results into benchmark_results/benchmark_data.json.
"""

import os
import json
import time
import subprocess
from typing import Dict, List
from src.dataset import generate_demand_profile
from src.graph_builder import TimeExpandedCorridorGraph
from src.baseline import simulate_baseline_schedule
from src.export_graph import export_corridor_to_file


def run_c_seq(graph_path: str) -> Dict:
    proc = subprocess.run([r"c_src\mcmf_seq.exe", graph_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr}
    return json.loads(proc.stdout.strip())


def run_c_par(graph_path: str, threads: int = 4) -> Dict:
    proc = subprocess.run([r"c_src\mcmf_par.exe", graph_path, str(threads)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        return {"error": proc.stderr}
    return json.loads(proc.stdout.strip())


def benchmark_granularity_scaling():
    print("\n--- [1/4] Benchmarking Discretization Granularity (Time Slots) ---")
    granularities = [60, 45, 30, 20, 15, 10] # slot minutes
    results = []

    for slot_m in granularities:
        data = generate_demand_profile(slot_minutes=slot_m)
        num_slots = data["num_slots"]
        fleet = 40
        graph_file = f"benchmark_results/graph_slot_{slot_m}m.txt"
        export_corridor_to_file(graph_file, slot_minutes=slot_m, fleet_size=fleet, initial_buses=(fleet//2, fleet//2))

        # 1. Python solver
        corridor = TimeExpandedCorridorGraph(data)
        corridor.build_graph(initial_buses=(fleet//2, fleet//2))
        t0 = time.time()
        py_res = corridor.solve(fleet)
        py_time_ms = (time.time() - t0) * 1000.0

        # 2. C Sequential
        c_seq_res = run_c_seq(graph_file)
        c_seq_time = c_seq_res.get("time_ms", 0.0)

        # 3. C OpenMP (4 threads)
        c_par_res = run_c_par(graph_file, threads=4)
        c_par_time = c_par_res.get("time_ms", 0.0)

        entry = {
            "slot_minutes": slot_m,
            "num_slots": num_slots,
            "num_nodes": corridor.num_nodes,
            "num_edges": corridor.mcmf.edge_count,
            "passengers_served": py_res["total_passengers_served"],
            "py_time_ms": round(py_time_ms, 2),
            "c_seq_time_ms": round(c_seq_time, 2),
            "c_par_time_ms": round(c_par_time, 2)
        }
        print(f"Slot: {slot_m:2d}m | Slots: {num_slots:2d} | Nodes: {entry['num_nodes']:3d} | Edges: {entry['num_edges']:4d} | Pax: {entry['passengers_served']} | Py: {py_time_ms:6.1f}ms | C Seq: {c_seq_time:5.1f}ms | C Par: {c_par_time:5.1f}ms")
        results.append(entry)

    return results


def benchmark_fleet_scaling():
    print("\n--- [2/4] Benchmarking Fleet Size Scaling (F) & Baseline Comparison ---")
    fleets = [10, 20, 30, 40, 50, 60, 75, 90, 110]
    slot_m = 30
    data = generate_demand_profile(slot_minutes=slot_m)
    results = []

    for f in fleets:
        init_s = f // 2
        init_h = f - init_s

        # Baseline
        base_res = simulate_baseline_schedule(data, total_buses=f)

        # Python MCMF
        corridor = TimeExpandedCorridorGraph(data)
        corridor.build_graph(initial_buses=(init_s, init_h))
        t0 = time.time()
        mcmf_res = corridor.solve(f)
        py_time_ms = (time.time() - t0) * 1000.0

        pax_mcmf = mcmf_res["total_passengers_served"]
        pax_base = base_res["total_passengers_served"]
        gain_pct = ((pax_mcmf - pax_base) / pax_base * 100.0) if pax_base > 0 else 0.0

        entry = {
            "fleet_size": f,
            "mcmf_pax": pax_mcmf,
            "baseline_pax": pax_base,
            "gain_pct": round(gain_pct, 2),
            "mcmf_satisfaction_pct": round(mcmf_res["demand_satisfaction_pct"], 2),
            "base_satisfaction_pct": round(base_res["demand_satisfaction_pct"], 2),
            "py_time_ms": round(py_time_ms, 2)
        }
        print(f"Fleet: {f:3d} | Baseline Pax: {pax_base:5d} ({entry['base_satisfaction_pct']:4.1f}%) | MCMF Pax: {pax_mcmf:5d} ({entry['mcmf_satisfaction_pct']:4.1f}%) | Gain: +{gain_pct:5.1f}% | Time: {py_time_ms:5.1f}ms")
        results.append(entry)

    return results


def benchmark_openmp_speedup():
    print("\n--- [3/4] Benchmarking OpenMP Multi-Threaded Scalability ---")
    slot_m = 10
    data = generate_demand_profile(slot_minutes=slot_m)
    fleet = 60
    graph_file = "benchmark_results/graph_omp_benchmark.txt"
    export_corridor_to_file(graph_file, slot_minutes=slot_m, fleet_size=fleet, initial_buses=(30, 30))

    c_seq = run_c_seq(graph_file)
    seq_time = max(c_seq.get("time_ms", 1.0), 1.0)

    threads_list = [1, 2, 4]
    results = []
    print(f"Base Sequential C Time: {seq_time:.2f} ms")

    for th in threads_list:
        c_par = run_c_par(graph_file, threads=th)
        par_time = c_par.get("time_ms", 0.0)
        speedup = seq_time / par_time if par_time > 0 else 1.0
        entry = {
            "threads": th,
            "time_ms": round(par_time, 2),
            "speedup": round(speedup, 2)
        }
        print(f"OpenMP Threads: {th} | Time: {par_time:6.2f} ms | Speedup: {speedup:4.2f}x")
        results.append(entry)

    return {
        "seq_time_ms": round(seq_time, 2),
        "thread_results": results
    }


def benchmark_multi_route_scaling():
    print("\n--- [4/4] Benchmarking Multi-Route Corridor Scaling (Expanded Network) ---")
    route_multipliers = [1, 2, 3, 4]
    slot_m = 30
    results = []

    for routes in route_multipliers:
        # Multi-route scaled demand
        data = generate_demand_profile(slot_minutes=slot_m, scale_factor=routes)
        fleet = 40 * routes
        corridor = TimeExpandedCorridorGraph(data)
        corridor.build_graph(initial_buses=(fleet//2, fleet//2))

        t0 = time.time()
        res = corridor.solve(fleet)
        elapsed = (time.time() - t0) * 1000.0

        entry = {
            "num_routes": routes,
            "fleet": fleet,
            "total_demand": res["total_demand"],
            "passengers_served": res["total_passengers_served"],
            "num_nodes": corridor.num_nodes,
            "num_edges": corridor.mcmf.edge_count,
            "time_ms": round(elapsed, 2)
        }
        print(f"Routes: {routes} | Fleet: {fleet:3d} | Edges: {entry['num_edges']:4d} | Demand: {entry['total_demand']:6d} | Solved Pax: {entry['passengers_served']:6d} | Time: {elapsed:5.1f}ms")
        results.append(entry)

    return results


def run_full_benchmarks():
    all_data = {
        "granularity_scaling": benchmark_granularity_scaling(),
        "fleet_scaling": benchmark_fleet_scaling(),
        "openmp_speedup": benchmark_openmp_speedup(),
        "multi_route_scaling": benchmark_multi_route_scaling()
    }
    with open("benchmark_results/benchmark_data.json", "w") as f:
        json.dump(all_data, f, indent=2)
    print("\n[OK] All benchmark experiments completed and saved to benchmark_results/benchmark_data.json")
    return all_data


if __name__ == "__main__":
    run_full_benchmarks()
