"""
cloudsim_exporter.py
Exports prediction results, VM workloads, server configurations, and
placement decisions into formats consumable by CloudSim Plus (Java).

Generates:
  - JSON/CSV files for VM workload traces
  - Server (Host) configuration files
  - Placement mapping files
  - CloudSim Plus simulation runner template (Java)

CloudSim Plus docs: https://cloudsimplus.org/
"""

import os
import sys
import csv
import json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


CLOUDSIM_OUTPUT_DIR = os.path.join(config.OUTPUT_DIR, "cloudsim")


def export_vm_workloads(predictions: np.ndarray,
                        actuals: np.ndarray,
                        confidence: np.ndarray,
                        scaler_mean: np.ndarray = None,
                        scaler_std: np.ndarray = None,
                        output_dir: str = None) -> str:
    """
    Export VM workload traces as CSV for CloudSim Plus UtilizationModelDynamic.

    Each row represents a VM at a timestep with CPU, Memory, Storage utilization.
    Values are denormalized back to [0, 100] percentage scale.

    Args:
        predictions: (N, horizon, 3) predicted utilization
        actuals:     (N, horizon, 3) actual utilization
        confidence:  (N, horizon, 3) confidence scores
        scaler_mean: (3,) normalization mean (optional, for denormalization)
        scaler_std:  (3,) normalization std (optional, for denormalization)
        output_dir:  output directory

    Returns:
        Path to the generated CSV file
    """
    if output_dir is None:
        output_dir = CLOUDSIM_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # Denormalize if scaler params provided
    pred = predictions[:, 0, :].copy()  # first step predictions
    actual = actuals[:, 0, :].copy()
    conf = confidence[:, 0, :].copy() if confidence.ndim == 3 else confidence.copy()

    if scaler_mean is not None and scaler_std is not None:
        pred = pred * scaler_std + scaler_mean
        actual = actual * scaler_std + scaler_mean

    pred = np.clip(pred, 0, 100)
    actual = np.clip(actual, 0, 100)

    path = os.path.join(output_dir, "vm_workloads.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "vm_id", "predicted_cpu", "predicted_mem", "predicted_storage",
            "actual_cpu", "actual_mem", "actual_storage",
            "confidence_cpu", "confidence_mem", "confidence_storage",
        ])
        for i in range(len(pred)):
            w.writerow([
                i,
                f"{pred[i, 0]:.2f}", f"{pred[i, 1]:.2f}", f"{pred[i, 2]:.2f}",
                f"{actual[i, 0]:.2f}", f"{actual[i, 1]:.2f}", f"{actual[i, 2]:.2f}",
                f"{conf[i, 0]:.4f}", f"{conf[i, 1]:.4f}", f"{conf[i, 2]:.4f}",
            ])

    print(f"[CloudSim] Exported {len(pred)} VM workloads to {path}")
    return path


def export_host_config(n_servers: int = None,
                       output_dir: str = None) -> str:
    """
    Export server (Host) configurations as JSON for CloudSim Plus.

    Each host has: CPU capacity (MIPS), RAM (MB), Storage (MB), Bandwidth (Mbps).

    Args:
        n_servers: number of servers
        output_dir: output directory

    Returns:
        Path to the generated JSON file
    """
    if n_servers is None:
        n_servers = config.NUM_SERVERS
    if output_dir is None:
        output_dir = CLOUDSIM_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    hosts = []
    for i in range(n_servers):
        hosts.append({
            "host_id": i,
            "pes": 8,                # Processing elements (CPU cores)
            "mips_per_pe": 10000,     # MIPS per core
            "ram_mb": 32768,          # 32 GB RAM
            "storage_mb": 1048576,    # 1 TB storage
            "bandwidth_mbps": 10000,  # 10 Gbps
            "cpu_capacity_percent": config.SERVER_CPU_CAPACITY,
            "mem_capacity_percent": config.SERVER_MEM_CAPACITY,
            "storage_capacity_percent": config.SERVER_STORAGE_CAPACITY,
            "power_idle_watts": config.P_IDLE,
            "power_max_watts": config.P_MAX,
        })

    path = os.path.join(output_dir, "host_config.json")
    with open(path, "w") as f:
        json.dump({"hosts": hosts}, f, indent=2)

    print(f"[CloudSim] Exported {n_servers} host configs to {path}")
    return path


def export_placement_map(placements: dict,
                         allocated: np.ndarray,
                         output_dir: str = None) -> str:
    """
    Export VM-to-Host placement mapping as CSV.

    Args:
        placements: dict {vm_id: server_id}
        allocated: (N, 3) allocated resources per VM
        output_dir: output directory

    Returns:
        Path to the generated CSV
    """
    if output_dir is None:
        output_dir = CLOUDSIM_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, "placement_map.csv")
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["vm_id", "host_id", "allocated_cpu", "allocated_mem",
                     "allocated_storage"])
        for vm_id, host_id in sorted(placements.items()):
            if vm_id < len(allocated):
                cpu, mem, stor = allocated[vm_id]
                w.writerow([vm_id, host_id, f"{cpu:.2f}", f"{mem:.2f}",
                            f"{stor:.2f}"])

    print(f"[CloudSim] Exported {len(placements)} placements to {path}")
    return path


def export_consolidation_events(consolidation_result: dict,
                                output_dir: str = None) -> str:
    """
    Export consolidation/migration events as JSON for CloudSim Plus simulation.

    Args:
        consolidation_result: dict from consolidation_engine
        output_dir: output directory

    Returns:
        Path to the generated JSON
    """
    if output_dir is None:
        output_dir = CLOUDSIM_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    event = {
        "consolidation_candidates": consolidation_result.get("candidates", 0),
        "total_migrations": consolidation_result.get("migrations", 0),
        "servers_powered_down": consolidation_result.get("servers_powered_down", 0),
        "energy_before_watts": consolidation_result.get("energy_before", 0),
        "energy_after_watts": consolidation_result.get("energy_after", 0),
        "energy_saved_watts": consolidation_result.get("energy_saved", 0),
        "active_servers": consolidation_result.get("active_servers", 0),
    }

    path = os.path.join(output_dir, "consolidation_events.json")
    with open(path, "w") as f:
        json.dump(event, f, indent=2)

    print(f"[CloudSim] Exported consolidation events to {path}")
    return path


def generate_cloudsim_java_template(output_dir: str = None) -> str:
    """
    Generate a CloudSim Plus Java simulation template that loads the
    exported CSV/JSON files and runs a placement simulation.

    This is a starter template — users can extend it for their specific
    CloudSim Plus version and requirements.

    Returns:
        Path to the generated Java file
    """
    if output_dir is None:
        output_dir = CLOUDSIM_OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    java_code = '''
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.cloudlet.CloudletSchedulerTimeShared;
import org.cloudsimplus.schedulers.vm.VmSchedulerTimeShared;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;

import java.io.*;
import java.util.*;

/**
 * CloudSim Plus simulation template for FP-PC-CA-E project.
 *
 * This template loads VM workloads and host configurations exported by
 * the Python pipeline and runs an energy-aware placement simulation.
 *
 * Usage:
 *   1. Run the Python pipeline (python main.py) to generate export files
 *   2. Copy outputs/cloudsim/ files to this project's resources directory
 *   3. Compile and run this Java class with CloudSim Plus on classpath
 *
 * Requires: CloudSim Plus 7.x+ (https://cloudsimplus.org/)
 */
public class VmPlacementSimulation {

    private static final int NUM_HOSTS = 20;
    private static final int HOST_PES = 8;
    private static final long HOST_MIPS = 10000;
    private static final long HOST_RAM = 32768;   // MB
    private static final long HOST_STORAGE = 1048576; // MB
    private static final long HOST_BW = 10000;    // Mbps
    private static final double POWER_IDLE = 150.0;  // Watts
    private static final double POWER_MAX = 400.0;   // Watts

    public static void main(String[] args) {
        CloudSimPlus simulation = new CloudSimPlus();

        // Create Datacenter with Hosts
        List<Host> hostList = createHosts();
        DatacenterSimple datacenter = new DatacenterSimple(simulation, hostList);

        // Create Broker
        DatacenterBrokerSimple broker = new DatacenterBrokerSimple(simulation);

        // Load VM workloads from CSV (exported by Python pipeline)
        List<Vm> vmList = createVmsFromCsv("vm_workloads.csv");
        List<Cloudlet> cloudletList = createCloudlets(vmList.size());

        broker.submitVmList(vmList);
        broker.submitCloudletList(cloudletList);

        simulation.start();

        // Print results
        List<Cloudlet> finishedCloudlets = broker.getCloudletFinishedList();
        System.out.println("\\n===== Simulation Results =====");
        System.out.printf("Finished Cloudlets: %d%n", finishedCloudlets.size());
        System.out.printf("Active Hosts: %d / %d%n",
            hostList.stream().filter(h -> !h.getVmList().isEmpty()).count(),
            hostList.size());

        // Calculate total energy
        double totalEnergy = hostList.stream()
            .mapToDouble(h -> {
                double util = h.getCpuPercentUtilization();
                return POWER_IDLE + util * (POWER_MAX - POWER_IDLE);
            })
            .sum();
        System.out.printf("Total Energy: %.2f Watts%n", totalEnergy);
    }

    private static List<Host> createHosts() {
        List<Host> hosts = new ArrayList<>();
        for (int i = 0; i < NUM_HOSTS; i++) {
            List<Pe> peList = new ArrayList<>();
            for (int j = 0; j < HOST_PES; j++) {
                peList.add(new PeSimple(HOST_MIPS));
            }
            Host host = new HostSimple(HOST_RAM, HOST_BW, HOST_STORAGE, peList)
                .setVmScheduler(new VmSchedulerTimeShared());

            PowerModelHostSimple powerModel =
                new PowerModelHostSimple(POWER_MAX, POWER_IDLE);
            host.setPowerModel(powerModel);
            hosts.add(host);
        }
        return hosts;
    }

    private static List<Vm> createVmsFromCsv(String csvPath) {
        List<Vm> vms = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(csvPath))) {
            String header = br.readLine(); // skip header
            String line;
            int vmId = 0;
            while ((line = br.readLine()) != null) {
                String[] parts = line.split(",");
                double cpuUtil = Double.parseDouble(parts[1]) / 100.0;

                Vm vm = new VmSimple(HOST_MIPS, HOST_PES / 2)
                    .setRam(HOST_RAM / 4)
                    .setBw(HOST_BW / 4)
                    .setSize(HOST_STORAGE / 8)
                    .setCloudletScheduler(new CloudletSchedulerTimeShared());

                vms.add(vm);
                vmId++;
                if (vmId >= 100) break; // Limit for demo
            }
        } catch (IOException e) {
            System.err.println("Could not read VM workloads CSV: " + e.getMessage());
            // Create default VMs
            for (int i = 0; i < 20; i++) {
                vms.add(new VmSimple(HOST_MIPS, HOST_PES / 2)
                    .setRam(HOST_RAM / 4).setBw(HOST_BW / 4)
                    .setSize(HOST_STORAGE / 8)
                    .setCloudletScheduler(new CloudletSchedulerTimeShared()));
            }
        }
        return vms;
    }

    private static List<Cloudlet> createCloudlets(int count) {
        List<Cloudlet> cloudlets = new ArrayList<>();
        for (int i = 0; i < count; i++) {
            Cloudlet cloudlet = new CloudletSimple(40000, HOST_PES / 2)
                .setFileSize(1024)
                .setOutputSize(1024)
                .setUtilizationModelCpu(new UtilizationModelDynamic(0.5))
                .setUtilizationModelRam(new UtilizationModelDynamic(0.4))
                .setUtilizationModelBw(new UtilizationModelDynamic(0.3));
            cloudlets.add(cloudlet);
        }
        return cloudlets;
    }
}
'''.strip()

    path = os.path.join(output_dir, "VmPlacementSimulation.java")
    with open(path, "w") as f:
        f.write(java_code)

    print(f"[CloudSim] Generated Java simulation template at {path}")
    return path


def run_cloudsim_export(predictions: np.ndarray,
                        actuals: np.ndarray,
                        confidence: np.ndarray,
                        placements: dict,
                        allocated: np.ndarray,
                        consolidation_result: dict) -> None:
    """
    Run the full CloudSim Plus export pipeline.

    Generates all files needed for CloudSim Plus integration:
      - vm_workloads.csv
      - host_config.json
      - placement_map.csv
      - consolidation_events.json
      - VmPlacementSimulation.java (template)

    Args:
        predictions: (N, horizon, 3)
        actuals: (N, horizon, 3)
        confidence: (N, horizon, 3)
        placements: {vm_id: server_id}
        allocated: (N, 3)
        consolidation_result: dict from consolidation engine
    """
    print("\n[CloudSim] Exporting data for CloudSim Plus integration ...")

    # Load scaler params if available for denormalization
    scaler_path = os.path.join(config.PROCESSED_DATA_DIR, "scaler_params.npz")
    scaler_mean, scaler_std = None, None
    if os.path.exists(scaler_path):
        params = np.load(scaler_path)
        scaler_mean = params["mean"]
        scaler_std = params["std"]

    export_vm_workloads(predictions, actuals, confidence,
                        scaler_mean, scaler_std)
    export_host_config()
    export_placement_map(placements, allocated)
    export_consolidation_events(consolidation_result)
    generate_cloudsim_java_template()

    print(f"[CloudSim] All exports saved to {CLOUDSIM_OUTPUT_DIR}")
    print("[CloudSim] To run in CloudSim Plus:")
    print("  1. Copy outputs/cloudsim/ to your CloudSim Plus project")
    print("  2. Add VmPlacementSimulation.java to your source")
    print("  3. Compile and run with CloudSim Plus 7.x+ on classpath")
