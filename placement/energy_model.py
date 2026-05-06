"""
energy_model.py
Linear power model for datacenter servers.
Pure CPU/numpy — no GPU usage.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


def compute_power(utilization: float,
                  p_idle: float = None,
                  p_max: float = None) -> float:
    """
    Linear power model:  P = P_idle + U × (P_max - P_idle)

    Args:
        utilization: CPU utilization ratio in [0, 1]
        p_idle: idle power (Watts)
        p_max:  max power (Watts)

    Returns:
        Power consumption in Watts
    """
    if p_idle is None:
        p_idle = config.P_IDLE
    if p_max is None:
        p_max = config.P_MAX

    u = np.clip(utilization, 0.0, 1.0)
    return p_idle + u * (p_max - p_idle)


def compute_incremental_power(current_util: float,
                              added_cpu: float,
                              server_capacity: float = None) -> float:
    """
    Compute the power increase from adding a VM to a server.

    Args:
        current_util: current CPU utilization ratio of the server [0, 1]
        added_cpu: CPU demand of the VM (normalized [0, 100])
        server_capacity: total server CPU capacity

    Returns:
        delta_power: incremental power in Watts
    """
    if server_capacity is None:
        server_capacity = config.SERVER_CPU_CAPACITY

    new_util = current_util + added_cpu / server_capacity
    new_util = min(new_util, 1.0)
    return compute_power(new_util) - compute_power(current_util)


def compute_datacenter_power(server_utilizations: np.ndarray) -> float:
    """
    Compute total datacenter power from all active server utilizations.

    Args:
        server_utilizations: (N_servers,) array of CPU utilizations [0, 1]

    Returns:
        Total power in Watts
    """
    total = 0.0
    for u in server_utilizations:
        if u > 0:
            total += compute_power(u)
    return total
