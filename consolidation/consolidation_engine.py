"""
consolidation_engine.py
Dynamic server consolidation: migrate VMs off underutilized servers
and power them down to save energy.
Pure CPU/numpy — no GPU usage.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from placement.energy_model import compute_power


def identify_underutilized(servers: list,
                           threshold: float = None) -> list:
    """
    Identify servers whose average utilization is below the threshold.

    Args:
        servers: list of Server objects
        threshold: consolidation threshold (default from config)

    Returns:
        list of server indices to consolidate
    """
    if threshold is None:
        threshold = config.CONSOLIDATION_THRESHOLD

    candidates = []
    for s in servers:
        if s.is_active and s.avg_utilization < threshold:
            candidates.append(s.server_id)
    return candidates


def consolidate_servers(servers: list, allocated: np.ndarray) -> dict:
    """
    Consolidate underutilized servers by migrating their VMs elsewhere.

    Steps:
        1. Identify underutilized servers (avg util < threshold)
        2. For each candidate, try to migrate all VMs to other servers
        3. If all VMs migrated → power down the server
        4. Track migrations and energy savings

    Args:
        servers: list of Server objects
        allocated: (N, 3) — current allocation per VM [cpu, mem, storage]

    Returns:
        dict with consolidation statistics
    """
    from placement.placement_engine import place_vm

    candidates = identify_underutilized(servers)
    migrations = 0
    servers_powered_down = 0
    energy_before = sum(compute_power(s.cpu_util) for s in servers if s.is_active)

    for sid in candidates:
        server = servers[sid]
        vm_list = list(server.vm_ids)  # copy since we'll modify
        all_migrated = True

        for vm_id in vm_list:
            if vm_id >= len(allocated):
                continue
            cpu, mem, storage = allocated[vm_id]

            # Remove VM from current server
            server.remove_vm(vm_id, cpu, mem, storage)

            # Try to place on another server
            placed = False
            for target in servers:
                if target.server_id == sid:
                    continue
                if target.can_fit(cpu, mem, storage):
                    target.add_vm(vm_id, cpu, mem, storage)
                    migrations += 1
                    placed = True
                    break

            if not placed:
                # Put it back if we couldn't migrate
                server.add_vm(vm_id, cpu, mem, storage)
                all_migrated = False

        if all_migrated and not server.is_active:
            servers_powered_down += 1

    energy_after = sum(compute_power(s.cpu_util) for s in servers if s.is_active)
    energy_saved = max(0.0, energy_before - energy_after)
    active_servers = sum(1 for s in servers if s.is_active)

    print(f"[Consolidation] Candidates: {len(candidates)}, "
          f"Migrations: {migrations}, "
          f"Servers powered down: {servers_powered_down}")
    print(f"[Consolidation] Active servers: {active_servers}/{len(servers)}, "
          f"Energy saved: {energy_saved:.1f} W")

    return {
        "candidates": len(candidates),
        "migrations": migrations,
        "servers_powered_down": servers_powered_down,
        "energy_saved": energy_saved,
        "active_servers": active_servers,
        "energy_before": energy_before,
        "energy_after": energy_after,
    }
