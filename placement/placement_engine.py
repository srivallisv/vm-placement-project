"""
placement_engine.py
Energy-aware VM placement: selects the server with minimum incremental
power consumption that has sufficient capacity.
Pure CPU/numpy — no GPU usage.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from placement.energy_model import compute_power


class Server:
    """Represents a physical server with CPU, Memory, Storage capacity."""

    def __init__(self, server_id: int):
        self.server_id = server_id
        self.cpu_capacity = config.SERVER_CPU_CAPACITY
        self.mem_capacity = config.SERVER_MEM_CAPACITY
        self.storage_capacity = config.SERVER_STORAGE_CAPACITY
        self.cpu_used = 0.0
        self.mem_used = 0.0
        self.storage_used = 0.0
        self.vm_ids = []

    @property
    def cpu_util(self) -> float:
        return self.cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    @property
    def avg_utilization(self) -> float:
        utils = [
            self.cpu_used / self.cpu_capacity,
            self.mem_used / self.mem_capacity,
            self.storage_used / self.storage_capacity,
        ]
        return np.mean(utils)

    def can_fit(self, cpu: float, mem: float, storage: float) -> bool:
        return (self.cpu_used + cpu <= self.cpu_capacity and
                self.mem_used + mem <= self.mem_capacity and
                self.storage_used + storage <= self.storage_capacity)

    def add_vm(self, vm_id: int, cpu: float, mem: float, storage: float):
        self.cpu_used += cpu
        self.mem_used += mem
        self.storage_used += storage
        self.vm_ids.append(vm_id)

    def remove_vm(self, vm_id: int, cpu: float, mem: float, storage: float):
        self.cpu_used = max(0.0, self.cpu_used - cpu)
        self.mem_used = max(0.0, self.mem_used - mem)
        self.storage_used = max(0.0, self.storage_used - storage)
        if vm_id in self.vm_ids:
            self.vm_ids.remove(vm_id)

    @property
    def is_active(self) -> bool:
        return len(self.vm_ids) > 0


def create_server_fleet(n_servers: int = None) -> list:
    """Initialize a fleet of empty servers."""
    if n_servers is None:
        n_servers = config.NUM_SERVERS
    return [Server(i) for i in range(n_servers)]


def place_vm(servers: list, vm_id: int,
             cpu: float, mem: float, storage: float) -> int:
    """
    Energy-aware placement: choose the server with minimum incremental
    power that has enough capacity.

    Args:
        servers: list of Server objects
        vm_id: identifier for the VM
        cpu, mem, storage: resource demands

    Returns:
        server_id if placed successfully, -1 if no server fits
    """
    best_server = None
    best_delta_p = float("inf")

    for server in servers:
        if not server.can_fit(cpu, mem, storage):
            continue

        # Incremental power from adding this VM
        old_power = compute_power(server.cpu_util)
        new_util = (server.cpu_used + cpu) / server.cpu_capacity
        new_power = compute_power(min(new_util, 1.0))
        delta_p = new_power - old_power

        if delta_p < best_delta_p:
            best_delta_p = delta_p
            best_server = server

    if best_server is not None:
        best_server.add_vm(vm_id, cpu, mem, storage)
        return best_server.server_id

    return -1  # Placement failed


def run_placement(servers: list,
                  allocated: np.ndarray,
                  labels: np.ndarray) -> dict:
    """
    Place VMs that are flagged for migration.

    Args:
        servers: list of Server objects
        allocated: (N, 3) — [cpu, mem, storage] allocated per VM
        labels: (N,) — 1 = needs placement/migration

    Returns:
        dict with placement results and statistics
    """
    placements = {}  # vm_id -> server_id
    failures = []

    for vm_id in range(len(labels)):
        if labels[vm_id] == 0:
            continue  # VM is fine, no action needed

        cpu, mem, storage = allocated[vm_id]
        sid = place_vm(servers, vm_id, cpu, mem, storage)
        if sid >= 0:
            placements[vm_id] = sid
        else:
            failures.append(vm_id)

    active = sum(1 for s in servers if s.is_active)
    print(f"[Placement] Placed: {len(placements)}, "
          f"Failed: {len(failures)}, Active servers: {active}/{len(servers)}")

    return {
        "placements": placements,
        "failures": failures,
        "active_servers": active,
    }
