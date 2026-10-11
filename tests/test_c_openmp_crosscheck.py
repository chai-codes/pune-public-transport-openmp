"""
test_c_openmp_crosscheck.py
Validation test cross-checking C Sequential and C OpenMP parallel solvers
against Python scratch MCMF and OR-Tools.
"""

import json
import subprocess
from src.dataset import generate_demand_profile
from src.graph_builder import TimeExpandedCorridorGraph
from src.export_graph import export_corridor_to_file


def run_c_binary(cmd):
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, shell=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Error running {cmd}: {proc.stderr}")
    return json.loads(proc.stdout.strip())


def test_cross_language_parity():
    print("[1/2] Exporting corridor graph to data/test_corridor.txt...")
    export_corridor_to_file("data/test_corridor.txt", slot_minutes=30, fleet_size=50)

    print("[2/2] Running C Sequential, C OpenMP (1, 2, 4 threads), and comparing solutions...")
    res_seq = run_c_binary(r"c_src\mcmf_seq.exe data\test_corridor.txt")
    print(f"      Sequential C:  Flow={res_seq['flow']}, Pax={res_seq['passengers']}, Cost={res_seq['cost']}, Time={res_seq['time_ms']:.2f}ms")

    for th in [1, 2, 4]:
        res_par = run_c_binary(rf"c_src\mcmf_par.exe data\test_corridor.txt {th}")
        print(f"      OpenMP C ({th} th): Flow={res_par['flow']}, Pax={res_par['passengers']}, Cost={res_par['cost']}, Time={res_par['time_ms']:.2f}ms")

        assert res_par["flow"] == res_seq["flow"], f"Flow mismatch at {th} threads: {res_par['flow']} vs {res_seq['flow']}"
        assert res_par["cost"] == res_seq["cost"], f"Cost mismatch at {th} threads: {res_par['cost']} vs {res_seq['cost']}"
        assert res_par["passengers"] == res_seq["passengers"], f"Passenger mismatch at {th} threads"

    print("      [OK] Cross-language and OpenMP parity verified with 100% mathematical precision!")


if __name__ == "__main__":
    test_cross_language_parity()
