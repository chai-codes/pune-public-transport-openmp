"""
compare_solvers.py
Interactive, user-friendly benchmarking and comparison between Serial (Sequential C)
and Parallel OpenMP solvers across customizable thread counts and workload sizes.

Features:
- Auto-detects system logical CPU cores.
- Interactive mode (prompts user for threads and fleet/resolution options) or CLI arguments.
- Formatted summary table displaying:
  * Solver Mode & Thread Count
  * Flow Delivered (Buses)
  * Passengers Carried
  * Total Net Cost
  * Execution Time (ms)
  * Speedup Factor (T_serial / T_parallel)
  * Parallel Efficiency (%)
- Automated optimality verdict highlighting trade-offs (Amdahl's law & thread synchronization overhead).
"""

import os
import sys
import json
import time
import subprocess
import multiprocessing
from typing import Dict, List, Tuple
from src.dataset import generate_demand_profile
from src.graph_builder import TimeExpandedCorridorGraph
from src.export_graph import export_corridor_to_file


def run_sequential_c(graph_path: str, runs: int = 5) -> Dict:
    times = []
    last_res = {}
    for _ in range(runs):
        proc = subprocess.run([r"c_src\mcmf_seq.exe", graph_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Error running sequential C: {proc.stderr}")
        res = json.loads(proc.stdout.strip())
        times.append(res.get("time_ms", 0.0))
        last_res = res

    # Use median time to avoid OS scheduler outliers
    times.sort()
    median_time = times[len(times) // 2]
    last_res["time_ms"] = median_time
    return last_res


def run_parallel_c(graph_path: str, threads: int, runs: int = 3) -> Dict:
    times = []
    last_res = {}
    for _ in range(runs):
        proc = subprocess.run([r"c_src\mcmf_par.exe", graph_path, str(threads)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Error running parallel C with {threads} threads: {proc.stderr}")
        res = json.loads(proc.stdout.strip())
        times.append(res.get("time_ms", 0.0))
        last_res = res

    times.sort()
    median_time = times[len(times) // 2]
    last_res["time_ms"] = median_time
    return last_res


def format_table_row(cols: List[str], widths: List[int], align: List[str] = None) -> str:
    res = []
    for i, col in enumerate(cols):
        w = widths[i]
        a = align[i] if align else '<'
        if a == '>':
            res.append(f"{str(col):>{w}}")
        elif a == '^':
            res.append(f"{str(col):^{w}}")
        else:
            res.append(f"{str(col):<{w}}")
    return " | ".join(res)


def run_solver_comparison(fleet: int = 50, slot_minutes: int = 30, thread_counts: List[int] = None) -> Dict:
    available_cores = multiprocessing.cpu_count()
    if not thread_counts:
        thread_counts = [1, 2, 4, min(8, available_cores)]
        thread_counts = sorted(list(set(thread_counts)))

    print("=" * 82)
    print("      PMPML FLEET DISPATCH: SERIAL VS PARALLEL OPENMP COMPARISON")
    print("=" * 82)
    print(f"System Hardware:           {available_cores} Logical Processors")
    print(f"Workload Configuration:    Fleet F = {fleet} buses | Slot Granularity = {slot_minutes} mins")

    # Generate graph
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    graph_file = os.path.join(data_dir, f"graph_compare_f{fleet}_s{slot_minutes}.txt")
    
    init_s = fleet // 2
    init_h = fleet - init_s
    meta = export_corridor_to_file(graph_file, slot_minutes=slot_minutes, fleet_size=fleet, initial_buses=(init_s, init_h))
    print(f"Network Dimension:         {meta['num_nodes']} Nodes | {meta['num_edges']} Edges")
    print("-" * 82)

    # 1. Run Serial Baseline
    print("Running Serial (Sequential C) solver...", end="", flush=True)
    t_start = time.time()
    seq_res = run_sequential_c(graph_file, runs=5)
    t_seq_ms = max(seq_res["time_ms"], 0.01)
    print(f" Completed! [{t_seq_ms:.3f} ms]")

    # 2. Run Parallel OpenMP for each thread count
    par_results = []
    for th in thread_counts:
        print(f"Running Parallel OpenMP ({th:2d} threads)...", end="", flush=True)
        res = run_parallel_c(graph_file, threads=th, runs=3)
        t_par_ms = res["time_ms"]
        speedup = t_seq_ms / t_par_ms if t_par_ms > 0 else 1.0
        efficiency = (speedup / th) * 100.0 if th > 0 else 0.0
        res["speedup"] = round(speedup, 3)
        res["efficiency_pct"] = round(efficiency, 1)
        par_results.append(res)
        print(f" Completed! [{t_par_ms:.3f} ms]")

    print("\n" + "=" * 82)
    print("                         EMPIRICAL COMPARISON RESULTS")
    print("=" * 82)

    headers = ["Implementation", "Threads", "Flow (Buses)", "Passengers", "Cost", "Time (ms)", "Speedup", "Efficiency"]
    widths =  [16,               7,         12,             11,           7,      10,          8,         10]
    aligns =  ['<',              '^',       '>',            '>',          '>',    '>',         '>',       '>']

    print(format_table_row(headers, widths, aligns))
    print("-" * 82)

    # Serial row
    serial_row = [
        "Serial (C)",
        "1",
        str(seq_res["flow"]),
        str(seq_res["passengers"]),
        str(seq_res["cost"]),
        f"{t_seq_ms:.3f}",
        "1.00x",
        "100.0%"
    ]
    print(format_table_row(serial_row, widths, aligns))

    # Parallel rows
    best_par_time = float("inf")
    best_threads = 1
    for p in par_results:
        p_row = [
            "Parallel (OpenMP)",
            str(p["threads"]),
            str(p["flow"]),
            str(p["passengers"]),
            str(p["cost"]),
            f"{p['time_ms']:.3f}",
            f"{p['speedup']:.2f}x",
            f"{p['efficiency_pct']:.1f}%"
        ]
        print(format_table_row(p_row, widths, aligns))
        if p["time_ms"] < best_par_time:
            best_par_time = p["time_ms"]
            best_threads = p["threads"]

    print("=" * 82)

    # Check numerical parity
    all_same_flow = all(p["flow"] == seq_res["flow"] for p in par_results)
    all_same_pax = all(p["passengers"] == seq_res["passengers"] for p in par_results)

    print("\n[VALIDATION VERDICT]")
    if all_same_flow and all_same_pax:
        print("  [OK] EXACT PARITY: All Serial and Parallel implementations delivered identical optimal flow and passengers.")
    else:
        print("  [FAIL] WARNING: Discrepancy observed between serial and parallel solutions.")

    print("\n[OPTIMALITY & SCALABILITY ANALYSIS]")
    print(f"  * Fastest Single Execution: {'Serial C' if t_seq_ms <= best_par_time else f'OpenMP C ({best_threads} threads)'} ({min(t_seq_ms, best_par_time):.3f} ms)")
    if t_seq_ms < 5.0 and best_par_time > t_seq_ms:
        print("  * Performance Insight: Because the corridor graph solves extremely fast (< 5 ms in Serial C),")
        print("    thread synchronization and barrier overhead in OpenMP dominate computation time for this problem size.")
        print("    Parallel scaling benefits become pronounced when scaling to multi-route networks or fine sub-minute discretization.")
    else:
        print(f"  * Optimal Thread Allocation: {best_threads} threads achieved highest performance.")

    print("-" * 82)

    return {
        "sequential": seq_res,
        "parallel": par_results,
        "optimal_threads": best_threads
    }


def interactive_cli():
    print("\n=======================================================")
    print(" PMPML FLEET DISPATCH: SERIAL VS PARALLEL OPENMP TOOL")
    print("=======================================================")
    max_cores = multiprocessing.cpu_count()
    print(f"Detected System Logical CPU Processors: {max_cores}\n")

    # 1. Ask for Fleet Size
    try:
        f_input = input(f"Enter Total Fleet Size F (press Enter for default [50]): ").strip()
        fleet = int(f_input) if f_input else 50
    except ValueError:
        fleet = 50

    # 2. Ask for Slot Granularity
    try:
        s_input = input(f"Enter Slot Duration in minutes (press Enter for default [30]): ").strip()
        slot_minutes = int(s_input) if s_input else 30
    except ValueError:
        slot_minutes = 30

    # 3. Ask for Threads
    th_input = input(f"Enter thread counts to compare, separated by spaces or commas\n(e.g., '1 2 4 8' or press Enter for default [1, 2, 4, {min(8, max_cores)}]): ").strip()
    if th_input:
        import re
        tokens = re.split(r"[\s,]+", th_input)
        thread_counts = []
        for tok in tokens:
            if tok.isdigit():
                val = int(tok)
                if 1 <= val <= max_cores * 2:
                    thread_counts.append(val)
        thread_counts = sorted(list(set(thread_counts)))
        if not thread_counts:
            thread_counts = [1, 2, 4]
    else:
        thread_counts = sorted(list(set([1, 2, 4, min(8, max_cores)])))

    run_solver_comparison(fleet=fleet, slot_minutes=slot_minutes, thread_counts=thread_counts)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] != "interactive":
        # Argument mode: python compare_solvers.py --threads 1 2 4 --fleet 50 --slot 30
        import argparse
        parser = argparse.ArgumentParser(description="Compare Serial vs Parallel OpenMP MCMF Solvers")
        parser.add_argument("--fleet", type=int, default=50, help="Fleet size F")
        parser.add_argument("--slot", type=int, default=30, help="Slot duration in minutes")
        parser.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4], help="List of thread counts")
        args = parser.parse_args()
        run_solver_comparison(fleet=args.fleet, slot_minutes=args.slot, thread_counts=args.threads)
    else:
        interactive_cli()
