"""
Carbon Shadow v2 — Main Flask Application
REST API + WebSocket for real-time device monitoring
"""

import os
import sys
import json
import threading
import time
import uuid
from datetime import datetime, timedelta
from flask import Flask, request, jsonify, Response
from flask_cors import CORS

# Adjust sys.path so core/ is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core.carbon_calculator import CarbonCalculator, REGION_FACTORS, DEVICE_PROFILES
from core.ml_engine import CarbonMLEngine
from core.device_monitor import (
    get_live_metrics, get_top_processes, get_network_interfaces,
    compute_device_emission, PSUTIL_AVAILABLE
)
import pandas as pd

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# ── Singletons
calculator = CarbonCalculator()
ml_engine = CarbonMLEngine()

# ── In-memory fleet registry (device_id → latest snapshot)
_fleet: dict = {}
_fleet_lock = threading.Lock()

# ── SSE subscribers (for real-time push)
_sse_clients: list = []
_sse_lock = threading.Lock()


def _push_event(event_type: str, data: dict):
    payload = f"event: {event_type}\ndata: {json.dumps(data)}\n\n"
    with _sse_lock:
        for q in _sse_clients:
            try:
                q.append(payload)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════
# HEALTH
# ═══════════════════════════════════════════════════════════════
@app.route("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ml_ready": ml_engine.is_trained,
        "psutil": PSUTIL_AVAILABLE,
        "fleet_size": len(_fleet),
        "version": "2.0.0",
    })


# ═══════════════════════════════════════════════════════════════
# REFERENCE DATA
# ═══════════════════════════════════════════════════════════════
@app.route("/api/regions")
def regions():
    return jsonify(REGION_FACTORS)

@app.route("/api/device-profiles")
def device_profiles():
    return jsonify(DEVICE_PROFILES)


# ═══════════════════════════════════════════════════════════════
# LIVE DEVICE METRICS  (this device)
# ═══════════════════════════════════════════════════════════════
@app.route("/api/device/live")
def device_live():
    """Returns live metrics for the machine running the backend"""
    region = request.args.get("region", "ap-south-1")
    device_type = request.args.get("device_type", "laptop")
    metrics = get_live_metrics()
    emission = compute_device_emission(metrics, region=region, device_type=device_type)
    return jsonify({**metrics, **emission})


@app.route("/api/device/processes")
def device_processes():
    return jsonify({"processes": get_top_processes(15)})


@app.route("/api/device/interfaces")
def device_interfaces():
    return jsonify({"interfaces": get_network_interfaces()})


# ═══════════════════════════════════════════════════════════════
# REAL-TIME SSE  (Server-Sent Events for live chart streaming)
# ═══════════════════════════════════════════════════════════════
@app.route("/api/device/stream")
def device_stream():
    """SSE endpoint — browser subscribes, gets metrics every 3s"""
    region = request.args.get("region", "ap-south-1")
    device_type = request.args.get("device_type", "laptop")
    client_queue = []
    with _sse_lock:
        _sse_clients.append(client_queue)

    def generate():
        try:
            while True:
                try:
                    metrics = get_live_metrics()
                    emission = compute_device_emission(metrics, region=region, device_type=device_type)
                    payload = json.dumps({**metrics, **emission,
                                          "ts": datetime.utcnow().isoformat()})
                    yield f"data: {payload}\n\n"
                    time.sleep(3)
                except GeneratorExit:
                    break
                except Exception as e:
                    yield f"data: {json.dumps({'error': str(e)})}\n\n"
                    time.sleep(3)
        finally:
            with _sse_lock:
                try:
                    _sse_clients.remove(client_queue)
                except ValueError:
                    pass

    return Response(generate(),
                    mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Access-Control-Allow-Origin": "*",
                    })


# ═══════════════════════════════════════════════════════════════
# FLEET / LAB MANAGEMENT
# ═══════════════════════════════════════════════════════════════
@app.route("/api/fleet/register", methods=["POST"])
def fleet_register():
    """Device agent calls this to register/heartbeat"""
    data = request.json or {}
    device_id = data.get("device_id") or str(uuid.uuid4())[:8]
    with _fleet_lock:
        _fleet[device_id] = {
            **data,
            "device_id": device_id,
            "status": "online",
            "last_seen": datetime.utcnow().isoformat(),
        }
    _push_event("fleet_update", {"device_id": device_id, "action": "register"})
    return jsonify({"device_id": device_id, "registered": True})


@app.route("/api/fleet/heartbeat", methods=["POST"])
def fleet_heartbeat():
    """Lightweight heartbeat from device agents"""
    data = request.json or {}
    device_id = data.get("device_id")
    if not device_id or device_id not in _fleet:
        return jsonify({"error": "unknown device"}), 404
    with _fleet_lock:
        _fleet[device_id].update({
            **data,
            "status": "online",
            "last_seen": datetime.utcnow().isoformat(),
        })
    _push_event("fleet_update", {"device_id": device_id, "action": "heartbeat"})
    return jsonify({"ok": True})


@app.route("/api/fleet/devices")
def fleet_devices():
    """Return all registered devices with their latest metrics"""
    cutoff = datetime.utcnow() - timedelta(minutes=5)
    with _fleet_lock:
        devices = []
        for d in _fleet.values():
            try:
                last = datetime.fromisoformat(d.get("last_seen", "2000-01-01"))
                d["status"] = "online" if last > cutoff else "offline"
            except Exception:
                d["status"] = "offline"
            devices.append(dict(d))
    summary = calculator.compute_fleet_summary(devices)
    return jsonify({"devices": devices, "summary": summary})


@app.route("/api/fleet/summary")
def fleet_summary():
    with _fleet_lock:
        devices = list(_fleet.values())
    return jsonify(calculator.compute_fleet_summary(devices))


@app.route("/api/fleet/remove/<device_id>", methods=["DELETE"])
def fleet_remove(device_id):
    with _fleet_lock:
        _fleet.pop(device_id, None)
    return jsonify({"removed": True})


# ═══════════════════════════════════════════════════════════════
# CARBON ESTIMATION — Manual (single service/device)
# ═══════════════════════════════════════════════════════════════
@app.route("/api/estimate/service", methods=["POST"])
def estimate_service():
    d = request.json or {}
    result = calculator.estimate_service(
        cpu_time_ms=d.get("cpu_time_ms", 100),
        memory_mb=d.get("memory_mb", 512),
        daily_requests=d.get("daily_requests", 1000),
        region=d.get("region", "ap-south-1"),
        network_latency_ms=d.get("network_latency_ms", 50),
        storage_gb=d.get("storage_gb", 0),
        gpu_time_ms=d.get("gpu_time_ms", 0),
        idle_cpu_percent=d.get("idle_cpu_percent", 20),
        cooling_pue=d.get("cooling_pue", 1.4),
        service_name=d.get("service_name", "my-service"),
    )
    result["ml_prediction"] = ml_engine.predict_service({**d, "emission_factor": result["emission_factor"]})
    return jsonify(result)


@app.route("/api/estimate/device", methods=["POST"])
def estimate_device():
    d = request.json or {}
    result = calculator.estimate_device(
        device_type=d.get("device_type", "laptop"),
        cpu_util_pct=d.get("cpu_util_pct", 40),
        memory_gb=d.get("memory_gb", 16),
        storage_gb=d.get("storage_gb", 512),
        storage_type=d.get("storage_type", "ssd"),
        uptime_hours_per_day=d.get("uptime_hours_per_day", 8),
        region=d.get("region", "ap-south-1"),
        gpu_util_pct=d.get("gpu_util_pct", 0),
        display_count=d.get("display_count", 1),
        cooling_pue=d.get("cooling_pue", 1.0),
        device_name=d.get("device_name", "My Device"),
    )
    result["ml_prediction"] = ml_engine.predict_device({**d, "emission_factor": result["emission_factor"]})
    return jsonify(result)


# ═══════════════════════════════════════════════════════════════
# CSV BULK UPLOAD
# ═══════════════════════════════════════════════════════════════
@app.route("/api/estimate/csv", methods=["POST"])
def estimate_csv():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".csv"):
        return jsonify({"error": "Only CSV files accepted"}), 400
    try:
        df = pd.read_csv(f)
        result = calculator.estimate_bulk_csv(df)
        ml_engine.train_on_real_data(result["services"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ═══════════════════════════════════════════════════════════════
# ANALYTICS
# ═══════════════════════════════════════════════════════════════
@app.route("/api/attribution", methods=["POST"])
def attribution():
    items = (request.json or {}).get("items", [])
    total = sum(s.get("total_daily_kg", 0) for s in items)
    result = []
    for s in items:
        name = s.get("service_name") or s.get("device_name") or "unknown"
        result.append({**s, "name": name,
                        "percentage": round(s.get("total_daily_kg",0)/max(total,1e-9)*100, 2),
                        "weekly_kg": round(s.get("total_daily_kg",0)*7, 6)})
    return jsonify({"items": result, "total_daily_kg": total,
                    "total_weekly_kg": total * 7, "count": len(result)})


@app.route("/api/forecast", methods=["POST"])
def forecast():
    d = request.json or {}
    return jsonify(ml_engine.forecast(
        d.get("items", []),
        d.get("horizon_months", 6),
        d.get("growth_rate", 0.05),
        d.get("scenario", "baseline"),
    ))


@app.route("/api/optimize", methods=["POST"])
def optimize():
    items = (request.json or {}).get("items", [])
    return jsonify(ml_engine.get_recommendations(items))


@app.route("/api/report", methods=["POST"])
def report():
    d = request.json or {}
    return jsonify(calculator.generate_report(d.get("items", []), d.get("period", "H1 2025")))


@app.route("/api/carbon-score", methods=["POST"])
def carbon_score():
    """Carbon intensity score: A+ to F based on emission per unit"""
    items = (request.json or {}).get("items", [])
    if not items:
        return jsonify({"score": "N/A", "grade": "N/A"})
    total = sum(s.get("total_daily_kg", 0) for s in items)
    total_units = sum(s.get("daily_requests", s.get("uptime_hours", 8)) for s in items)
    epu = total / max(total_units, 1)
    if epu < 0.000001:   grade = "A+"
    elif epu < 0.00001:  grade = "A"
    elif epu < 0.0001:   grade = "B+"
    elif epu < 0.001:    grade = "B"
    elif epu < 0.01:     grade = "C"
    elif epu < 0.1:      grade = "D"
    else:                grade = "F"
    return jsonify({"score": round(epu, 8), "grade": grade,
                    "total_daily_kg": total, "unit_count": total_units})


if __name__ == "__main__":
    print("🌿 Carbon Shadow v2 Backend — starting on :5000")
    print(f"   psutil available: {PSUTIL_AVAILABLE}")
    print(f"   ML pretrained: {ml_engine.is_trained}")
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)