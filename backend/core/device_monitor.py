"""
Carbon Shadow v2 — Live Device Monitor
Reads real system metrics via psutil (cross-platform).
Falls back gracefully if permissions are limited.
"""

import time
import uuid
import platform
import datetime
import os

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

# Country/region to grid emission factor lookup
GRID_EF_BY_COUNTRY = {
    "IN": 0.71, "US": 0.42, "GB": 0.23, "DE": 0.35,
    "FR": 0.06, "AU": 0.71, "CN": 0.62, "JP": 0.47,
    "CA": 0.13, "SG": 0.49, "BR": 0.08, "ZA": 0.93,
    "SE": 0.08, "NO": 0.03, "NL": 0.32, "IT": 0.33,
}

DEVICE_TDP_PROFILES = {
    "laptop": {"idle": 8, "tdp": 45},
    "desktop": {"idle": 40, "tdp": 125},
    "workstation": {"idle": 80, "tdp": 250},
    "server": {"idle": 80, "tdp": 200},
    "mini_pc": {"idle": 5, "tdp": 35},
}


def get_platform_info():
    info = {
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "node": platform.node(),
        "python": platform.python_version(),
    }
    return info


def get_live_metrics():
    """
    Collect real-time system metrics.
    Returns dict of all available metrics with fallbacks.
    """
    if not PSUTIL_AVAILABLE:
        return _get_mock_metrics()

    metrics = {}

    # ── CPU
    try:
        metrics["cpu_util_pct"] = psutil.cpu_percent(interval=0.5)
        metrics["cpu_count_logical"] = psutil.cpu_count(logical=True)
        metrics["cpu_count_physical"] = psutil.cpu_count(logical=False)
        metrics["cpu_freq_mhz"] = getattr(psutil.cpu_freq(), "current", 0) if psutil.cpu_freq() else 0
    except Exception:
        metrics["cpu_util_pct"] = 20.0
        metrics["cpu_count_logical"] = 4
        metrics["cpu_count_physical"] = 2
        metrics["cpu_freq_mhz"] = 2400

    # ── Memory
    try:
        mem = psutil.virtual_memory()
        metrics["memory_total_gb"] = round(mem.total / (1024**3), 2)
        metrics["memory_used_gb"] = round(mem.used / (1024**3), 2)
        metrics["memory_util_pct"] = mem.percent
        metrics["memory_available_gb"] = round(mem.available / (1024**3), 2)
    except Exception:
        metrics["memory_total_gb"] = 8.0
        metrics["memory_used_gb"] = 4.0
        metrics["memory_util_pct"] = 50.0
        metrics["memory_available_gb"] = 4.0

    # ── Disk
    try:
        disk = psutil.disk_usage("/")
        metrics["storage_total_gb"] = round(disk.total / (1024**3), 1)
        metrics["storage_used_gb"] = round(disk.used / (1024**3), 1)
        metrics["storage_util_pct"] = disk.percent
        # Estimate storage type from speed
        io = psutil.disk_io_counters()
        metrics["disk_read_mb"] = round((io.read_bytes if io else 0) / (1024**2), 1)
        metrics["disk_write_mb"] = round((io.write_bytes if io else 0) / (1024**2), 1)
    except Exception:
        metrics["storage_total_gb"] = 256.0
        metrics["storage_used_gb"] = 128.0
        metrics["storage_util_pct"] = 50.0

    # ── Network
    try:
        net = psutil.net_io_counters()
        metrics["net_bytes_sent_mb"] = round(net.bytes_sent / (1024**2), 2)
        metrics["net_bytes_recv_mb"] = round(net.bytes_recv / (1024**2), 2)
        # Active connections
        metrics["net_connections"] = len(psutil.net_connections(kind="inet"))
    except PermissionError:
        metrics["net_bytes_sent_mb"] = 0
        metrics["net_bytes_recv_mb"] = 0
        metrics["net_connections"] = 0
    except Exception:
        metrics["net_bytes_sent_mb"] = 0
        metrics["net_bytes_recv_mb"] = 0
        metrics["net_connections"] = 0

    # ── Battery (laptops)
    try:
        bat = psutil.sensors_battery()
        if bat:
            metrics["battery_pct"] = bat.percent
            metrics["battery_charging"] = bat.power_plugged
            metrics["battery_secsleft"] = bat.secsleft if bat.secsleft != psutil.POWER_TIME_UNLIMITED else -1
        else:
            metrics["battery_pct"] = None
            metrics["battery_charging"] = True
    except Exception:
        metrics["battery_pct"] = None
        metrics["battery_charging"] = True

    # ── Temperature (Linux/macOS only)
    try:
        temps = psutil.sensors_temperatures()
        if temps:
            all_temps = []
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current:
                        all_temps.append(entry.current)
            metrics["cpu_temp_c"] = round(sum(all_temps) / len(all_temps), 1) if all_temps else None
        else:
            metrics["cpu_temp_c"] = None
    except Exception:
        metrics["cpu_temp_c"] = None

    # ── Process count
    try:
        metrics["process_count"] = len(list(psutil.process_iter()))
    except Exception:
        metrics["process_count"] = 0

    # ── Uptime
    try:
        metrics["uptime_seconds"] = time.time() - psutil.boot_time()
        metrics["uptime_hours"] = round(metrics["uptime_seconds"] / 3600, 2)
    except Exception:
        metrics["uptime_hours"] = 8.0
        metrics["uptime_seconds"] = 28800

    # ── OS info
    metrics.update(get_platform_info())
    metrics["timestamp"] = datetime.datetime.utcnow().isoformat()

    return metrics


def get_top_processes(limit=10):
    """Return top CPU-consuming processes"""
    if not PSUTIL_AVAILABLE:
        return []
    try:
        procs = []
        for p in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent", "status"]):
            try:
                info = p.info
                if info["cpu_percent"] is not None:
                    procs.append(info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda x: x.get("cpu_percent", 0), reverse=True)
        return procs[:limit]
    except Exception:
        return []


def get_network_interfaces():
    """Return active network interfaces and IPs for lab discovery"""
    if not PSUTIL_AVAILABLE:
        return []
    try:
        interfaces = []
        for iface, addrs in psutil.net_if_addrs().items():
            ips = [a.address for a in addrs if a.family == 2]  # AF_INET = IPv4
            if ips:
                stats = psutil.net_if_stats().get(iface)
                interfaces.append({
                    "name": iface,
                    "ips": ips,
                    "speed_mbps": getattr(stats, "speed", 0) if stats else 0,
                    "is_up": getattr(stats, "isup", False) if stats else False,
                })
        return interfaces
    except Exception:
        return []


def _get_mock_metrics():
    """Fallback metrics when psutil is unavailable"""
    import random
    return {
        "cpu_util_pct": round(random.uniform(15, 65), 1),
        "cpu_count_logical": 8,
        "cpu_count_physical": 4,
        "cpu_freq_mhz": 2400,
        "memory_total_gb": 16.0,
        "memory_used_gb": round(random.uniform(4, 12), 1),
        "memory_util_pct": round(random.uniform(30, 75), 1),
        "memory_available_gb": 6.0,
        "storage_total_gb": 512.0,
        "storage_used_gb": 256.0,
        "storage_util_pct": 50.0,
        "battery_pct": None,
        "battery_charging": True,
        "cpu_temp_c": round(random.uniform(45, 72), 1),
        "process_count": 142,
        "uptime_hours": round(random.uniform(1, 72), 2),
        "uptime_seconds": 3600,
        "net_bytes_sent_mb": 120.5,
        "net_bytes_recv_mb": 980.2,
        "net_connections": 18,
        "os": platform.system(),
        "os_version": platform.version(),
        "machine": platform.machine(),
        "processor": platform.processor() or "Unknown CPU",
        "node": platform.node(),
        "python": platform.python_version(),
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }


def compute_device_emission(metrics, region="ap-south-1", device_type="laptop", cooling_pue=1.0):
    """Compute real-time carbon emission from live metrics"""
    from .carbon_calculator import REGION_FACTORS, DEVICE_TDP_PROFILES as CALC_PROFILES

    ef = REGION_FACTORS.get(region, {"factor": 0.42})["factor"]
    # Map to calculator profiles
    dt_map = {
        "laptop": "laptop", "desktop": "desktop", "workstation": "workstation",
        "server": "server_1u", "mini_pc": "mini_pc"
    }
    profile_name = dt_map.get(device_type, "laptop")

    from .carbon_calculator import DEVICE_PROFILES
    profile = DEVICE_PROFILES.get(profile_name, DEVICE_PROFILES["laptop"])

    cpu_util = metrics.get("cpu_util_pct", 30) / 100
    cpu_watts = profile["idle_w"] + (profile["cpu_tdp"] - profile["idle_w"]) * cpu_util
    memory_gb = metrics.get("memory_total_gb", 8)
    mem_watts = memory_gb * 0.3725
    uptime_h = min(metrics.get("uptime_hours", 8), 24)

    cpu_kwh = (cpu_watts * uptime_h) / 1000
    mem_kwh = (mem_watts * uptime_h) / 1000
    total_kwh = (cpu_kwh + mem_kwh) * cooling_pue
    total_kg = total_kwh * ef

    return {
        "total_daily_kg": round(total_kg, 6),
        "total_kwh": round(total_kwh, 4),
        "cpu_watts_current": round(cpu_watts, 1),
        "memory_watts_current": round(mem_watts, 1),
        "emission_factor": ef,
        "region": region,
    }