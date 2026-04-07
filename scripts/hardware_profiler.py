#!/usr/bin/env python3
"""
Hardware Profiler for Reproducibility

Collects detailed hardware specifications for experiment reproducibility.
Outputs JSON file with CPU, RAM, GPU, disk, and network specs.

Usage: python3 hardware_profiler.py --output hardware_specs.json
"""

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path


def get_cpu_info():
    """Get detailed CPU information."""
    info = {
        "architecture": platform.machine(),
        "processor": platform.processor(),
        "cores_physical": None,
        "cores_logical": None,
        "frequency_mhz": None,
        "cache_l1": None,
        "cache_l2": None,
        "cache_l3": None,
        "model": None,
        "vendor": None
    }
    
    try:
        if sys.platform == "darwin":
            # macOS
            result = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], 
                                  capture_output=True, text=True)
            info["model"] = result.stdout.strip()
            
            result = subprocess.run(["sysctl", "-n", "hw.physicalcpu"], 
                                  capture_output=True, text=True)
            info["cores_physical"] = int(result.stdout.strip())
            
            result = subprocess.run(["sysctl", "-n", "hw.logicalcpu"], 
                                  capture_output=True, text=True)
            info["cores_logical"] = int(result.stdout.strip())
            
            result = subprocess.run(["sysctl", "-n", "hw.cpufrequency"], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                info["frequency_mhz"] = int(result.stdout.strip()) / 1_000_000
                
        elif sys.platform.startswith("linux"):
            # Linux
            with open("/proc/cpuinfo") as f:
                cpuinfo = f.read()
            
            for line in cpuinfo.split("\n"):
                if "model name" in line:
                    info["model"] = line.split(":")[1].strip()
                    break
            
            result = subprocess.run(["nproc", "--all"], 
                                  capture_output=True, text=True)
            info["cores_logical"] = int(result.stdout.strip())
            
            # Try to get physical cores
            result = subprocess.run(["lscpu"], capture_output=True, text=True)
            for line in result.stdout.split("\n"):
                if "Core(s) per socket" in line:
                    cores = int(line.split(":")[1].strip())
                if "Socket(s)" in line:
                    sockets = int(line.split(":")[1].strip())
                    info["cores_physical"] = cores * sockets
                    
    except Exception as e:
        print(f"Warning: Could not get full CPU info: {e}", file=sys.stderr)
    
    return info


def get_memory_info():
    """Get RAM specifications."""
    info = {
        "total_gb": None,
        "type": None,
        "speed_mhz": None,
        "channels": None
    }
    
    try:
        if sys.platform == "darwin":
            result = subprocess.run(["sysctl", "-n", "hw.memsize"], 
                                  capture_output=True, text=True)
            info["total_gb"] = int(result.stdout.strip()) / (1024**3)
            
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        info["total_gb"] = kb / (1024**2)
                        break
            
            # Try to get RAM type and speed
            try:
                result = subprocess.run(["dmidecode", "-t", "memory"], 
                                      capture_output=True, text=True)
                for line in result.stdout.split("\n"):
                    if "Type:" in line and "DDR" in line:
                        info["type"] = line.split(":")[1].strip()
                    if "Speed:" in line and "MHz" in line:
                        speed_str = line.split(":")[1].strip()
                        info["speed_mhz"] = int(speed_str.split()[0])
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Could not get full memory info: {e}", file=sys.stderr)
    
    return info


def get_disk_info():
    """Get disk specifications."""
    info = {
        "type": None,  # SSD, HDD, NVMe
        "total_gb": None,
        "free_gb": None,
        "filesystem": None
    }
    
    try:
        # Get disk space
        result = subprocess.run(["df", "-h", "."], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split("\n")
        if len(lines) > 1:
            parts = lines[1].split()
            info["filesystem"] = parts[0]
            info["total_gb"] = parts[1]
            info["free_gb"] = parts[3]
        
        # Try to detect SSD vs HDD
        if sys.platform.startswith("linux"):
            try:
                result = subprocess.run(["lsblk", "-d", "-o", "name,rota"], 
                                      capture_output=True, text=True)
                # rota=0 means SSD, rota=1 means HDD
                if "0" in result.stdout:
                    info["type"] = "SSD/NVMe"
                else:
                    info["type"] = "HDD"
            except:
                pass
                
    except Exception as e:
        print(f"Warning: Could not get disk info: {e}", file=sys.stderr)
    
    return info


def get_gpu_info():
    """Get GPU specifications (if available)."""
    info = {
        "available": False,
        "model": None,
        "memory_gb": None,
        "driver_version": None
    }
    
    try:
        # Try NVIDIA
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", 
                               "--format=csv,noheader"], 
                              capture_output=True, text=True)
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            info["available"] = True
            info["model"] = parts[0].strip()
            info["memory_gb"] = parts[1].strip()
            info["driver_version"] = parts[2].strip()
    except:
        pass
    
    return info


def get_network_info():
    """Get network specifications."""
    info = {
        "interfaces": [],
        "bandwidth_test": None
    }
    
    try:
        if sys.platform == "darwin":
            result = subprocess.run(["ifconfig"], capture_output=True, text=True)
        else:
            result = subprocess.run(["ip", "addr"], capture_output=True, text=True)
        
        # Parse interfaces (simplified)
        for line in result.stdout.split("\n"):
            if "inet " in line and "127.0.0.1" not in line:
                info["interfaces"].append(line.strip())
                
    except Exception as e:
        print(f"Warning: Could not get network info: {e}", file=sys.stderr)
    
    return info


def get_os_info():
    """Get operating system information."""
    return {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "machine": platform.machine(),
        "python_version": platform.python_version()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Profile hardware for reproducibility"
    )
    parser.add_argument("--output", default="hardware_specs.json",
                       help="Output JSON file")
    args = parser.parse_args()
    
    print("Collecting hardware specifications...")
    
    specs = {
        "timestamp": subprocess.run(["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], 
                                   capture_output=True, text=True).stdout.strip(),
        "hostname": platform.node(),
        "os": get_os_info(),
        "cpu": get_cpu_info(),
        "memory": get_memory_info(),
        "disk": get_disk_info(),
        "gpu": get_gpu_info(),
        "network": get_network_info()
    }
    
    # Write to file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(specs, f, indent=2)
    
    print(f"\n✅ Hardware specs saved to: {output_path}")
    print("\nSummary:")
    print(f"  CPU: {specs['cpu']['model']}")
    print(f"  Cores: {specs['cpu']['cores_physical']} physical, "
          f"{specs['cpu']['cores_logical']} logical")
    print(f"  RAM: {specs['memory']['total_gb']:.1f} GB")
    print(f"  Disk: {specs['disk']['type']} ({specs['disk']['free_gb']} free)")
    if specs['gpu']['available']:
        print(f"  GPU: {specs['gpu']['model']} ({specs['gpu']['memory_gb']})")
    else:
        print(f"  GPU: Not available")


if __name__ == "__main__":
    main()
