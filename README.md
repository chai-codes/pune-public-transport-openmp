# Directional Fleet Dispatch Optimization for PMPML Hinjewadi <-> Swargate Corridor
**Design and Analysis of Algorithms (DAA) Laboratory & Capstone Project**  
*Pune Mahanagar Parivahan Mahamandal Limited (PMPML) Route Analysis*

---

## 📌 Executive Summary
Urban public transit corridors connecting residential hubs to IT industrial corridors often experience severe directional peak imbalance. On the **Swargate $\leftrightarrow$ Hinjewadi IT Park** corridor (24 km), morning demand is heavily inbound towards Hinjewadi, while evening peak demand reverses towards Swargate.

Standard transit operations dispatch symmetric timetables (equal headways both ways), causing severe crowding in peak directions and running empty or near-empty buses in reverse. 

This project formulates dynamic fleet dispatch as a **Min-Cost Max-Flow (MCMF)** optimization over a **Time-Expanded Directed Graph** with non-linear convex passenger revenue piecewise discretization and empty deadheading support. The optimization algorithm is implemented **completely from scratch** in pure Python (no external solver dependency) as well as **Sequential C** and **Parallel OpenMP C**, achieving exact mathematical parity cross-checked against Google OR-Tools.

---

## 🗂 Project Structure
```text
.
├── c_src/
│   ├── mcmf_sequential.c          # Pure C sequential MCMF (SPFA + Min-Heap Dijkstra)
│   ├── mcmf_parallel.c            # Multi-threaded OpenMP C MCMF solver
│   ├── mcmf_seq.exe               # Compiled C sequential binary
│   └── mcmf_par.exe               # Compiled OpenMP C binary
├── src/
│   ├── dataset.py                 # Hinjewadi-Swargate demand & congestion generator
│   ├── mcmf.py                    # Pure Python from-scratch MCMF implementation
│   ├── graph_builder.py           # Time-expanded corridor network builder
│   ├── baseline.py                # Uniform symmetric headway simulation baseline
│   ├── export_graph.py            # Graph exporter for C solver benchmarks
│   ├── benchmark.py               # Empirical scalability and performance harness
│   └── plotter.py                 # Publication-ready visualization generator
├── tests/
│   ├── test_mcmf_crosscheck.py    # Cross-check suite vs Google OR-Tools
│   └── test_c_openmp_crosscheck.py# Cross-check suite between Python, C & OpenMP
├── data/
│   └── hinjewadi_swargate_demand.json
├── plots/
│   ├── plot_demand_profile.png
│   ├── plot_dispatch_schedule.png
│   ├── plot_optimization_vs_baseline.png
│   ├── plot_scaling_granularity.png
│   └── plot_multi_route_scaling.png
├── docs/
│   └── PROJECT_REPORT.md          # Comprehensive DAA academic report
├── requirements.txt               # Python package dependencies
├── main.py                        # Central CLI application entrypoint
└── README.md
```

---

## 🚀 Quick Start Guide

### 1. Prerequisites
- **Python 3.10+** (with `numpy`, `matplotlib`, `ortools`)
- **GCC compiler** with OpenMP support (`gcc -fopenmp`)

### 2. Setup Environment
```bash
pip install -r requirements.txt
```

### 3. Compile C Binaries (Sequential & OpenMP)
```bash
# Sequential C
gcc -O3 c_src/mcmf_sequential.c -o c_src/mcmf_seq.exe

# Parallel OpenMP C
gcc -O3 -fopenmp c_src/mcmf_parallel.c -o c_src/mcmf_par.exe
```

---

## 💻 Running the Optimizer & Tests

### Single Optimization Run
```bash
python main.py run --fleet 50 --slot 30
```

### MCMF vs Baseline Comparison
```bash
python main.py compare --fleet 50 --slot 30
```
*Sample Output:*
```text
=== Optimization vs Baseline Comparison (Fleet = 50) ===
Baseline Uniform Passengers: 8630 (86.78%)
MCMF Optimal Passengers:     9803 (98.57%)
Passenger Improvement:       +1173 passengers (+13.59%)
============================================================
```

### Run Cross-Check Unit Tests
```bash
# Verify against Google OR-Tools
python -m tests.test_mcmf_crosscheck

# Verify C Sequential and OpenMP Parity
python -m tests.test_c_openmp_crosscheck
```

### Run Benchmarks & Generate Plots
```bash
# Run benchmarks across slots, fleets, threads, and routes
python main.py benchmark

# Generate 300 DPI publication plots
python main.py plot
```

---

## 🔬 Algorithm & Mathematical Model

### 1. Time-Expanded Directed Graph
- **Time discretization**: Operating day $[06:00, 22:30]$ partitioned into discrete time slots $t \in [0, T-1]$ (default 30 min, down to 5 min).
- **Nodes**: $(u, t)$ for terminal $u \in \{\text{Swargate}, \text{Hinjewadi}\}$ and slot $t$. Plus super-source $S$ and super-sink $T_{\text{sink}}$.
- **Super-Source edges**: $S \to (u, 0)$ with capacity $F_u$ (initial depot allocation) and cost 0. Total flow target $= F$.
- **Waiting edges**: $(u, t) \to (u, t+1)$ with capacity $F$ and cost $0$. Represents buses parked idling at the terminal.
- **Trip edges**: $(u, t) \to (v, t + \Delta t)$ where:
  $$\Delta t = \left\lceil \frac{\text{travel\_time}(t) + \text{layover}}{\text{slot\_duration}} \right\rceil$$
  Travel time dynamically accounts for peak traffic congestion.

### 2. Concave Revenue Piecewise Linearization
- Passenger capacity per bus: $C = 60$.
- For slot $t$ with demand $D_t$, the marginal passengers carried by the $k$-th dispatched bus is:
  $$\Delta P_k = \min(C, \max(0, D_t - (k-1)C))$$
- Modeled as parallel directed arcs with capacity $1$ and cost $c_k = -\Delta P_k$.
- Because $\Delta P_1 \ge \Delta P_2 \ge \dots \ge \Delta P_k$, cost sequence is convex non-decreasing:
  $$- \Delta P_1 \le - \Delta P_2 \le \dots \le - \Delta P_k$$
  This guarantees that Min-Cost Flow naturally saturates lower-cost (higher passenger yield) edges first, retaining exactness.
- **Deadheading / Empty running**: Edge with capacity $F$ and cost $0$ allows repositioning buses without passenger constraints.

### 3. Min-Cost Max-Flow Complexity
1. **Initial Potentials**: Computed once via SPFA / Bellman-Ford in $O(|V| \cdot |E|)$ to eliminate negative edge weights and establish dual feasible potentials $\pi$.
2. **Successive Augmentation**: $F$ flow augmentations using Dijkstra with reduced costs $c^\pi(u, v) = c(u, v) + \pi(u) - \pi(v) \ge 0$, running in $O(F \cdot (|E| + |V| \log |V|))$.

---

## 📊 Summary of Experimental Results
1. **Passenger Throughput Gain**: At constrained fleet sizes ($F = 40$ to $50$ buses), directional dynamic MCMF delivers **$+11.2\%$ to $+13.6\%$ more passengers** ($>1,170$ additional commuters daily) compared to uniform symmetric scheduling.
2. **Runtime Efficiency**:
   - Sequential C solves the full day corridor problem in **$<1$ millisecond**.
   - Pure Python solves within **$5$ to $15$ milliseconds**.
3. **Cross-Language Parity**: Scratch Python, Sequential C, and Parallel OpenMP achieve **100% exact numerical match** with Google OR-Tools.

---

## 👥 Authors & Academic Submission
- **Course**: Design and Analysis of Algorithms (DAA) Laboratory
- **Target Route**: PMPML Route 100 / Corridor Hinjewadi Phase 3 $\leftrightarrow$ Swargate
- **License**: MIT
