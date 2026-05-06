"""
failure_handler.py
Adaptive confidence decay on placement failure with retry logic.
Pure CPU/numpy — no GPU usage.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from placement.placement_engine import place_vm
from allocation.allocation_engine import allocate_resources


def handle_failures(servers: list,
                    failed_vm_ids: list,
                    predicted: np.ndarray,
                    current: np.ndarray,
                    confidence: np.ndarray) -> dict:
    """
    Retry failed placements with adaptive confidence decay.

    Strategy:
        1. Reduce confidence by decay factor (0.8)
        2. Recalculate allocation with reduced confidence
        3. Retry placement
        4. Repeat up to MAX_RETRIES
        5. If still fails → Best Fit heuristic (most-full server that fits)

    Args:
        servers: list of Server objects
        failed_vm_ids: list of VM IDs that failed initial placement
        predicted: (N, features) predicted resource usage
        current:   (N, features) current resource usage
        confidence: (N, features) original confidence scores

    Returns:
        dict with recovery stats
    """
    recovered = []
    still_failed = []
    total_retries = 0
    retry_log = []

    for vm_id in failed_vm_ids:
        placed = False
        vm_conf = confidence[vm_id].copy()

        for attempt in range(1, config.MAX_RETRIES + 1):
            # Decay confidence
            vm_conf *= config.CONFIDENCE_DECAY
            total_retries += 1

            # Recalculate allocation
            alloc = current[vm_id] + vm_conf * (predicted[vm_id] - current[vm_id])
            alloc = np.clip(alloc, 0.0, None)

            cpu, mem, storage = alloc
            sid = place_vm(servers, vm_id, cpu, mem, storage)
            retry_log.append({
                "vm_id": vm_id,
                "attempt": attempt,
                "confidence": vm_conf.mean(),
                "success": sid >= 0,
            })

            if sid >= 0:
                recovered.append(vm_id)
                placed = True
                break

        # Fallback: Best Fit heuristic
        if not placed:
            alloc = current[vm_id] + vm_conf * (predicted[vm_id] - current[vm_id])
            alloc = np.clip(alloc, 0.0, None)
            cpu, mem, storage = alloc

            best_server = None
            best_util = -1.0
            for server in servers:
                if server.can_fit(cpu, mem, storage):
                    if server.avg_utilization > best_util:
                        best_util = server.avg_utilization
                        best_server = server

            if best_server is not None:
                best_server.add_vm(vm_id, cpu, mem, storage)
                recovered.append(vm_id)
            else:
                still_failed.append(vm_id)

    recovery_rate = (len(recovered) / max(len(failed_vm_ids), 1)) * 100
    print(f"[Failure] Retries: {total_retries}, Recovered: {len(recovered)}, "
          f"Still failed: {len(still_failed)}, Recovery rate: {recovery_rate:.1f}%")

    return {
        "recovered": recovered,
        "still_failed": still_failed,
        "total_retries": total_retries,
        "recovery_rate": recovery_rate,
        "retry_log": retry_log,
    }
