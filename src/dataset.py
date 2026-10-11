"""
dataset.py
Realistic demand and travel time profile generator for the PMPML Hinjewadi <-> Swargate corridor.
Corridor features:
- Hinjewadi (Terminal H): Major IT hub (Rajiv Gandhi Infotech Park Phase 1/2/3).
- Swargate (Terminal S): Major central residential and transport interchange in Pune.
- Distance: ~24 km.
- Nominal off-peak travel time: ~70-75 mins. Peak congestion travel time: 95-120 mins.
- Layover required at terminal: 10 mins.
- Asymmetry: Heavy morning commute (Swargate -> Hinjewadi), heavy evening commute (Hinjewadi -> Swargate).
"""

import json
import math
from typing import Dict, List, Tuple


def get_slot_times(start_hour: float = 6.0, end_hour: float = 22.5, slot_minutes: int = 30) -> List[Tuple[int, float, str]]:
    """
    Generate slot indices, decimal hour, and formatted string 'HH:MM'.
    Default: 06:00 to 22:30 (10:30 PM).
    """
    slots = []
    total_slots = int(round((end_hour - start_hour) * 60 / slot_minutes)) + 1
    for t in range(total_slots):
        cur_min = start_hour * 60 + t * slot_minutes
        h = int(cur_min // 60)
        m = int(cur_min % 60)
        slots.append((t, cur_min / 60.0, f"{h:02d}:{m:02d}"))
    return slots


def get_peak_factor(hour: float) -> float:
    """
    Traffic congestion multiplier depending on time of day.
    Morning peak: 08:30 - 10:30 (peak factor up to 1.55)
    Evening peak: 17:30 - 20:00 (peak factor up to 1.60)
    Off peak: ~1.0
    """
    # Gaussian-like bumps for traffic peaks
    # Morning peak centered at 9.25 (09:15 AM)
    m_peak = 0.55 * math.exp(-((hour - 9.25) ** 2) / (2 * 1.0 ** 2))
    # Evening peak centered at 18.75 (06:45 PM)
    e_peak = 0.60 * math.exp(-((hour - 18.75) ** 2) / (2 * 1.2 ** 2))
    return 1.0 + m_peak + e_peak


def calculate_travel_time(departure_hour: float, base_time: float = 75.0) -> float:
    """
    Return one-way travel time in minutes for a departure at `departure_hour`.
    """
    return base_time * get_peak_factor(departure_hour)


def generate_demand_profile(slot_minutes: int = 30, scale_factor: float = 1.0) -> Dict:
    """
    Generate realistic directional passenger demand per time slot.
    Returns:
    {
      "slot_minutes": slot_minutes,
      "slots": [{"index": t, "hour": h, "time_str": "HH:MM", "travel_time_min": tt}],
      "demand": {
         "S->H": [passengers in slot t], # Swargate to Hinjewadi
         "H->S": [passengers in slot t]  # Hinjewadi to Swargate
      }
    }
    """
    slot_info = get_slot_times(start_hour=6.0, end_hour=22.5, slot_minutes=slot_minutes)
    
    s_to_h_demand = []
    h_to_s_demand = []
    slot_details = []

    for t, hour, time_str in slot_info:
        tt = calculate_travel_time(hour, base_time=75.0)
        slot_details.append({
            "index": t,
            "hour": hour,
            "time_str": time_str,
            "travel_time_min": round(tt, 1)
        })

        # Scaled per slot duration (if slot is 30 mins vs 15 mins)
        duration_ratio = slot_minutes / 30.0

        # Swargate -> Hinjewadi: Massive morning inbound peak to Hinjewadi IT park
        # Off-peak base: 40-70 passengers/30min
        # Morning peak (07:30 - 10:30): up to 480-550 passengers/30min
        # Evening reverse: modest ~100-140
        s_h_base = 45.0
        s_h_morning = 450.0 * math.exp(-((hour - 9.0) ** 2) / (2 * 1.1 ** 2))
        s_h_midday = 40.0 * math.exp(-((hour - 13.5) ** 2) / (2 * 1.5 ** 2))
        s_h_evening = 80.0 * math.exp(-((hour - 18.5) ** 2) / (2 * 1.5 ** 2))
        total_s_h = (s_h_base + s_h_morning + s_h_midday + s_h_evening) * duration_ratio * scale_factor

        # Hinjewadi -> Swargate: Massive evening outbound rush from IT Park
        # Morning reverse: modest ~60-90
        # Evening peak (17:00 - 20:30): up to 500-580 passengers/30min
        h_s_base = 40.0
        h_s_morning = 70.0 * math.exp(-((hour - 8.5) ** 2) / (2 * 1.2 ** 2))
        h_s_midday = 50.0 * math.exp(-((hour - 13.5) ** 2) / (2 * 1.5 ** 2))
        h_s_evening = 480.0 * math.exp(-((hour - 18.75) ** 2) / (2 * 1.2 ** 2))
        total_h_s = (h_s_base + h_s_morning + h_s_midday + h_s_evening) * duration_ratio * scale_factor

        s_to_h_demand.append(int(round(total_s_h)))
        h_to_s_demand.append(int(round(total_h_s)))

    return {
        "corridor": "Hinjewadi <-> Swargate (PMPML)",
        "slot_minutes": slot_minutes,
        "layover_minutes": 10.0,
        "bus_capacity": 60,  # Standard PMPML standard/electric bus capacity
        "num_slots": len(slot_info),
        "slots": slot_details,
        "demand": {
            "S->H": s_to_h_demand,
            "H->S": h_to_s_demand
        }
    }


def save_demand_profile(filepath: str, slot_minutes: int = 30, scale_factor: float = 1.0):
    data = generate_demand_profile(slot_minutes=slot_minutes, scale_factor=scale_factor)
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
    return data


if __name__ == "__main__":
    profile = save_demand_profile("data/hinjewadi_swargate_demand.json", slot_minutes=30)
    print(f"Generated demand profile: {profile['num_slots']} slots.")
    print(f"Total S->H demand: {sum(profile['demand']['S->H'])} passengers.")
    print(f"Total H->S demand: {sum(profile['demand']['H->S'])} passengers.")
