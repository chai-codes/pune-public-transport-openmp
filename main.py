"""
main.py
CLI entry point for the DAA project:
Directional Fleet Dispatch Optimization for PMPML Hinjewadi <-> Swargate Corridor.

Commands:
  python main.py run           : Runs single corridor optimization with default config
  python main.py compare       : Runs MCMF vs Uniform baseline comparison
  python main.py benchmark     : Runs full scaling benchmark suite and saves json
  python main.py plot          : Generates publication-ready plots
  python main.py verify        : Executes cross-check validation tests against OR-Tools & C solvers
"""

import sys
import argparse
from src.dataset import generate_demand_profile, save_demand_profile
from src.graph_builder import TimeExpandedCorridorGraph
from src.baseline import simulate_baseline_schedule
from src.benchmark import run_full_benchmarks
from src.plotter import generate_all_plots


def cmd_run(args):
    print(f"=== PMPML Fleet Dispatch Optimization (Fleet = {args.fleet}) ===")
    profile = generate_demand_profile(slot_minutes=args.slot)
    corridor = TimeExpandedCorridorGraph(profile)
    init_s = args.fleet // 2
    init_h = args.fleet - init_s
    corridor.build_graph(initial_buses=(init_s, init_h))
    res = corridor.solve(args.fleet)

    print(f"Discretization:      {args.slot} min slots ({corridor.num_slots} time steps)")
    print(f"Network Size:        {corridor.num_nodes} nodes, {corridor.mcmf.edge_count} edges")
    print(f"Total Daily Demand:  {res['total_demand']} passengers")
    print(f"Passengers Carried:  {res['total_passengers_served']} ({res['demand_satisfaction_pct']:.2f}% satisfied)")
    print(f"Buses Dispatched S->H: {sum(res['departures']['S->H'])}")
    print(f"Buses Dispatched H->S: {sum(res['departures']['H->S'])}")
    print("=" * 60)


def cmd_compare(args):
    print(f"=== Optimization vs Baseline Comparison (Fleet = {args.fleet}) ===")
    profile = generate_demand_profile(slot_minutes=args.slot)
    
    # 1. Baseline
    base_res = simulate_baseline_schedule(profile, total_buses=args.fleet)
    
    # 2. MCMF
    corridor = TimeExpandedCorridorGraph(profile)
    corridor.build_graph(initial_buses=(args.fleet // 2, args.fleet - args.fleet // 2))
    mcmf_res = corridor.solve(args.fleet)

    gain = mcmf_res['total_passengers_served'] - base_res['total_passengers_served']
    gain_pct = (gain / base_res['total_passengers_served'] * 100.0) if base_res['total_passengers_served'] > 0 else 0.0

    print(f"Baseline Uniform Passengers: {base_res['total_passengers_served']} ({base_res['demand_satisfaction_pct']:.2f}%)")
    print(f"MCMF Optimal Passengers:     {mcmf_res['total_passengers_served']} ({mcmf_res['demand_satisfaction_pct']:.2f}%)")
    print(f"Passenger Improvement:       +{gain} passengers (+{gain_pct:.2f}%)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PMPML Fleet Dispatch Optimizer using MCMF on Time-Expanded Graphs")
    subparsers = parser.add_subparsers(dest="command")

    p_run = subparsers.add_parser("run", help="Run single corridor dispatch")
    p_run.add_argument("--fleet", type=int, default=50, help="Total fleet size F (default: 50)")
    p_run.add_argument("--slot", type=int, default=30, help="Slot duration in minutes (default: 30)")

    p_comp = subparsers.add_parser("compare", help="Compare MCMF with symmetric baseline")
    p_comp.add_argument("--fleet", type=int, default=50, help="Fleet size F (default: 50)")
    p_comp.add_argument("--slot", type=int, default=30, help="Slot duration in minutes (default: 30)")

    subparsers.add_parser("benchmark", help="Run full scaling benchmarks")
    subparsers.add_parser("plot", help="Generate all analysis plots")

    p_solvers = subparsers.add_parser("compare-solvers", help="Interactive or parameterized comparison of Serial vs Parallel OpenMP")
    p_solvers.add_argument("--fleet", type=int, default=50, help="Fleet size F (default: 50)")
    p_solvers.add_argument("--slot", type=int, default=30, help="Slot duration in minutes (default: 30)")
    p_solvers.add_argument("--threads", type=int, nargs="+", default=None, help="List of thread counts (e.g. --threads 1 2 4 8)")
    p_solvers.add_argument("--interactive", action="store_true", help="Prompt interactively for thread counts and parameters")

    args = parser.parse_args()
    if args.command == "run":
        cmd_run(args)
    elif args.command == "compare":
        cmd_compare(args)
    elif args.command == "benchmark":
        run_full_benchmarks()
    elif args.command == "plot":
        generate_all_plots()
    elif args.command == "compare-solvers":
        from compare_solvers import interactive_cli, run_solver_comparison
        if args.interactive or (args.threads is None and len(sys.argv) == 2):
            interactive_cli()
        else:
            run_solver_comparison(fleet=args.fleet, slot_minutes=args.slot, thread_counts=args.threads)
    else:
        # Default behavior: run comparison
        args.fleet = 50
        args.slot = 30
        cmd_run(args)
        cmd_compare(args)


if __name__ == "__main__":
    main()
