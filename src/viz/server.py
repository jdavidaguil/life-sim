"""Server module: serves simulation state to a browser-based renderer."""

from __future__ import annotations
import json
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
import numpy as np

PORT: int = 7777

_state: dict[str, Any] = {}
_lock: threading.Lock = threading.Lock()


def update_state(simulation, step: int) -> None:
    snapshot = {
        "step": step,
        "population": simulation.agent_count(),
        "width": simulation.width,
        "height": simulation.height,
        "agents": [
            [
                a.x,
                a.y,
                float(a.policy.traits[0]),
                float(a.policy.traits[1]),
                float(a.policy.traits[2]),
                float(a.policy.traits[3]),
            ]
            for a in simulation.agents
        ],
        "resources": (
            simulation.grid.resources / simulation.grid.MAX_RESOURCE
        ).tolist(),
        "history": {
            "steps":      simulation.history["step"],
            "total":      simulation.history["total"],
            "mean_rw":    simulation.history["mean_resource_weight"],
            "mean_cs":    simulation.history["mean_crowd_sensitivity"],
            "mean_noise": simulation.history["mean_noise"],
            "mean_ea":    simulation.history["mean_energy_awareness"],
        },
    }
    traits_sample = snapshot["history"]["mean_rw"]
    if len(traits_sample) > 0:
        print(
            f"[server] step={snapshot['step']} "
            f"rw={traits_sample[-1]:.3f} "
            f"noise={snapshot['history']['mean_noise'][-1]:.3f} "
            f"cs={snapshot['history']['mean_cs'][-1]:.3f}"
        )
    with _lock:
        _state.clear()
        _state.update(snapshot)


class SimHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/":
            index = Path(__file__).parent / "static" / "index.html"
            try:
                body = index.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(body)
            except FileNotFoundError:
                self.send_response(404)
                self.end_headers()
        elif self.path == "/state":
            with _lock:
                body = json.dumps(_state).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args) -> None:
        pass


def start(open_browser: bool = True) -> None:
    server = HTTPServer(("", PORT), SimHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Visualiser running at http://localhost:{PORT}")
    if open_browser:
        time.sleep(0.5)
        webbrowser.open(f"http://localhost:{PORT}")
