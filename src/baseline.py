"""
baseline.py
Baseline dispatch model: Equal / uniform frequency dispatching both ways.
Simulates conventional PMPML transit operation where buses are evenly scheduled
in both directions (symmetric headway) regardless of dynamic directional demand imbalance.
"""

import math
from typing import Dict, List, Tuple


def simulate_baseline_schedule(config: Dict, total_buses: int = 50) -> Dict:
    """
    Simulates a standard fixed-headway symmetric dispatch policy.
    At each terminal, buses depart at fixed intervals as long as buses are available.
    Arriving buses become available after (travel_time + layover).
    """
    num_slots = config["num_slots"]
    demand = config["demand"]
    slots = config["slots"]
    slot_minutes = config["slot_minutes"]
    layover_minutes = config.get("layover_minutes", 10.0)
    cap = config.get("bus_capacity", 60)

    # Initial allocation: half at Swargate (0), half at Hinjewadi (1)
    buses_swargate = total_buses // 2
    buses_hinjewadi = total_buses - buses_swargate

    # Buses available at each terminal at slot t
    avail_S = [0] * (num_slots + 20)
    avail_H = [0] * (num_slots + 20)
    avail_S[0] = buses_swargate
    avail_H[0] = buses_hinjewadi

    departures = {"S->H": [0] * num_slots, "H->S": [0] * num_slots}
    pax_carried = {"S->H": [0] * num_slots, "H->S": [0] * num_slots}

    # In symmetric timetable, depot targets equal departures per hour
    # With F buses and ~2.5h cycle time, roughly F / 5 buses depart per hour total (i.e. half per terminal per slot)
    target_dep_per_slot = max(1, int(round((total_buses / 5.0) * (slot_minutes / 60.0))))

    for t in range(num_slots):
        # Accumulate carried over buses
        if t > 0:
            avail_S[t] += avail_S[t - 1]
            avail_H[t] += avail_H[t - 1]

        tt = slots[t]["travel_time_min"]
        transit_slots = int(math.ceil((tt + layover_minutes) / slot_minutes))
        arr_t = t + transit_slots

        # Swargate -> Hinjewadi dispatch
        dep_S = min(avail_S[t], target_dep_per_slot)
        avail_S[t] -= dep_S
        avail_H[arr_t] += dep_S
        departures["S->H"][t] = dep_S

        # Hinjewadi -> Swargate dispatch
        dep_H = min(avail_H[t], target_dep_per_slot)
        avail_H[t] -= dep_H
        avail_S[arr_t] += dep_H
        departures["H->S"][t] = dep_H

        # Calculate passengers served
        pax_carried["S->H"][t] = min(demand["S->H"][t], dep_S * cap)
        pax_carried["H->S"][t] = min(demand["H->S"][t], dep_H * cap)

    total_passengers = sum(pax_carried["S->H"]) + sum(pax_carried["H->S"])
    total_demand = sum(demand["S->H"]) + sum(demand["H->S"])

    return {
        "policy": "Symmetric Uniform Baseline",
        "fleet_size": total_buses,
        "total_passengers_served": total_passengers,
        "total_demand": total_demand,
        "demand_satisfaction_pct": (total_passengers / total_demand * 100.0) if total_demand > 0 else 0.0,
        "departures": departures,
        "passengers_carried": pax_carried
    }
