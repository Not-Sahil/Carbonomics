"""
Carbon Shadow v2 — Device Agent
Runs on each machine in the lab/company.
Collects real system metrics, computes carbon, pushes to central backend.

Usage:
  python agent.py --server http://SERVER_IP:5000 --region ap-south-1 --device-type laptop
"""

import argparse
import json
import sys
import os
import time
import uuid
import platform
import signal
import threading
import socket
from datetime import datetime

# Allow running standalone or as installed package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠ 'requests' not installed. Install with: pip install requests")

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    print("⚠ 'psutil' not installed. Install with: pip install psutil")


AGENT_VERSION = "2.0.0"
DEFAULT_SERVER = "http://localhost:5000"
HEARTBEAT_INTERVAL = 30   # seconds
FULL_REPORT_INTERVAL = 300  # seconds (5 min)

# Grid emission factors per country
GRID_EF = {
    "IN": 0.71, "US": 0.42, "GB": 0.23, "DE": 0.35,
    "EU": 0.28, "SG": 0.49, "JP": 0.47, "AU": 0.71,
}

DEVICE_TDP = {
    "laptop": {"idle": 8, "tdp": 45},
    "desktop": {"idle": 40, "tdp": 125},
    "workstation": {"idle": 80, "tdp": 250},
    "server": {"idle": 80, "tdp": 200},
    "mini_pc": {"idle": 5, "tdp": 35},
}


class DeviceAgent:
    def __init__(self, server_url, region, device_type, device_name=None, interval=HEARTBEAT_INTERVAL):
        self.server_url = server_url.rstrip("/")
        self.region = region
        self.device_type = device_type
        self.device_name = device_name or platform.node() or socket.gethostname()
        self.interval = interval
        self.device_id = self._load_or_create_id()
        self._running = False
        self._thread = None
        self._ef = GRID_EF.get(region.split("-")[0].upper(), 0.42)

        # Map region codes to EF
        EF_MAP = {
            "ap-south-1": 0.71, "us-east-1": 0.42, "us-west-2": 0.15,
            "eu-west-1": 0.28, "eu-central-1": 0.35, "eu-north-1": 0.08,
            "ap-southeast-1": 0.49, "ap-northeast-1": 0.47,
        }
        self._ef = EF_MAP.get(region, 0.42)

    def _load_or_create_id(self):
        """Persist device ID across restarts"""
        id_file = os.path.join(os.path.expanduser("~"), ".carbon_shadow_id")
        if os.path.exists(id_file):
            with open(id_file) as f:
                return f.read().strip()
        new_id = str(uuid.uuid4())[:12]
        with open(id_file, "w") as f:
            f.write(new_id)
        return new_id

    def _collect(self):
        """Collect system metrics"""
        data = {
            "device_id": self.device_id,
            "device_name": self.device_name,
            "device_type": self.device_type,
            "region": self.region,
            "os": platform.system(),
            "os_version": platform.version()[:80],
            "processor": platform.processor()[:100] or "Unknown",
            "node": platform.node(),
            "agent_version": AGENT_VERSION,
        }

        if PSUTIL_AVAILABLE:
            try:
                data["cpu_util_pct"] = psutil.cpu_percent(interval=0.3)
                data["cpu_count"] = psutil.cpu_count(logical=True)
                mem = psutil.virtual_memory()
                data["memory_total_gb"] = round(mem.total / (1024**3), 2)
                data["memory_used_gb"] = round(mem.used / (1024**3), 2)
                data["memory_util_pct"] = mem.percent
                disk = psutil.disk_usage("/")
                data["storage_total_gb"] = round(disk.total / (1024**3), 1)
                data["storage_used_gb"] = round(disk.used / (1024**3), 1)
                data["uptime_hours"] = round((time.time() - psutil.boot_time()) / 3600, 2)
                bat = psutil.sensors_battery()
                if bat:
                    data["battery_pct"] = bat.percent
                    data["battery_charging"] = bat.power_plugged
                net = psutil.net_io_counters()
                data["net_bytes_sent_mb"] = round(net.bytes_sent / (1024**2), 1)
                data["net_bytes_recv_mb"] = round(net.bytes_recv / (1024**2), 1)
            except Exception as e:
                data["psutil_error"] = str(e)
        else:
            data["cpu_util_pct"] = 30.0
            data["memory_total_gb"] = 8.0
            data["memory_used_gb"] = 4.0
            data["uptime_hours"] = 8.0

        # ── Compute emission locally
        profile = DEVICE_TDP.get(self.device_type, DEVICE_TDP["laptop"])
        cpu_util = data.get("cpu_util_pct", 30) / 100
        cpu_w = profile["idle"] + (profile["tdp"] - profile["idle"]) * cpu_util
        mem_gb = data.get("memory_total_gb", 8)
        mem_w = mem_gb * 0.3725
        uptime = min(data.get("uptime_hours", 8), 24)
        total_kwh = ((cpu_w + mem_w) * uptime) / 1000
        total_kg = total_kwh * self._ef

        data["total_daily_kg"] = round(total_kg, 6)
        data["total_kwh_daily"] = round(total_kwh, 4)
        data["emission_factor"] = self._ef
        data["status"] = "online"
        data["timestamp"] = datetime.utcnow().isoformat()

        return data

    def _register(self):
        if not REQUESTS_AVAILABLE:
            return False
        try:
            data = self._collect()
            r = requests.post(f"{self.server_url}/api/fleet/register", json=data, timeout=5)
            if r.ok:
                print(f"✅ Registered as {self.device_id} at {self.server_url}")
                return True
            print(f"❌ Registration failed: {r.text[:100]}")
            return False
        except Exception as e:
            print(f"❌ Cannot reach server: {e}")
            return False

    def _heartbeat(self):
        if not REQUESTS_AVAILABLE:
            return
        try:
            data = self._collect()
            requests.post(f"{self.server_url}/api/fleet/heartbeat", json=data, timeout=5)
            print(f"💚 [{datetime.now().strftime('%H:%M:%S')}] Heartbeat — CPU:{data.get('cpu_util_pct',0):.1f}% | CO₂:{data.get('total_daily_kg',0):.4f} kg/day")
        except Exception as e:
            print(f"⚠ Heartbeat failed: {e}")

    def start(self):
        self._running = True
        self._register()

        def loop():
            while self._running:
                self._heartbeat()
                time.sleep(self.interval)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        print(f"🟢 Agent running — Device: {self.device_name} [{self.device_id}] | Interval: {self.interval}s")

    def stop(self):
        self._running = False
        print("\n🔴 Agent stopped.")

    def run_forever(self):
        self.start()

        def handle_signal(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self.stop()


def main():
    parser = argparse.ArgumentParser(description="Carbon Shadow Device Agent v2")
    parser.add_argument("--server", default=DEFAULT_SERVER, help="Backend URL (default: http://localhost:5000)")
    parser.add_argument("--region", default="ap-south-1", help="Grid region code (default: ap-south-1)")
    parser.add_argument("--device-type", default="laptop", choices=list(DEVICE_TDP.keys()), help="Device type")
    parser.add_argument("--name", default=None, help="Device friendly name")
    parser.add_argument("--interval", type=int, default=HEARTBEAT_INTERVAL, help="Heartbeat interval (seconds)")
    parser.add_argument("--once", action="store_true", help="Collect once and print (don't run daemon)")
    args = parser.parse_args()

    agent = DeviceAgent(
        server_url=args.server,
        region=args.region,
        device_type=args.device_type,
        device_name=args.name,
        interval=args.interval,
    )

    if args.once:
        data = agent._collect()
        print(json.dumps(data, indent=2))
    else:
        agent.run_forever()


if __name__ == "__main__":
    main()