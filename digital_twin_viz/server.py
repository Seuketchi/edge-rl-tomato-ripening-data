"""
WebSocket backend for the Digital Twin visualization — multi-agent edition.

Supports:
  - Single-agent mode:  any one policy (DQN/PPO/A2C/baselines) + any variant (A/B/C)
  - Compare mode:       multiple agents run on the same seed in lockstep

Usage:
    python digital_twin_viz/server.py
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from websockets.datastructures import Headers
from websockets.http11 import Response as WsResponse

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import websockets
from stable_baselines3 import DQN, PPO, A2C
import yaml

from ml_training.rl.environment import TomatoRipeningEnv

# ── Agent catalogue ───────────────────────────────────────────────────
_OUT = ROOT / "outputs"

AGENT_DEFS: dict[str, dict] = {
    "dqn": {
        "label": "DQN", "algo": "dqn",
        "path": _OUT / "rl_20260523_075926" / "best_model" / "best_model.zip",
        "variant": "B", "color": "#2a7a3b",
    },
    "ppo": {
        "label": "PPO", "algo": "ppo",
        "path": _OUT / "algo_comparison_20260523_063952" / "ppo" / "best_model" / "best_model.zip",
        "variant": "B", "color": "#c84b38",
    },
    "a2c": {
        "label": "A2C", "algo": "a2c",
        "path": _OUT / "algo_comparison_20260523_063952" / "a2c" / "best_model" / "best_model.zip",
        "variant": "B", "color": "#3a6abf",
    },
    "dqn_a": {
        "label": "DQN Var-A", "algo": "dqn",
        "path": _OUT / "rl_20260303_204723" / "best_model" / "best_model.zip",
        "variant": "A", "color": "#c8a838",
    },
    "dqn_c": {
        "label": "DQN Var-C", "algo": "dqn",
        "path": _OUT / "rl_20260303_212521" / "best_model" / "best_model.zip",
        "variant": "C", "color": "#7a4a8a",
    },
    "fixed_day": {
        "label": "Fixed-Day", "algo": "baseline", "policy": "fixed_day",
        "variant": "B", "color": "#888888",
    },
    "fixed_stage5": {
        "label": "Fixed-Stage5", "algo": "baseline", "policy": "fixed_stage5",
        "variant": "B", "color": "#e8883a",
    },
    "random": {
        "label": "Random", "algo": "baseline", "policy": "random",
        "variant": "B", "color": "#aaaaaa",
    },
}

# ── Load all ML models at startup ─────────────────────────────────────
_models: dict[str, Any] = {}
_CLS = {"dqn": DQN, "ppo": PPO, "a2c": A2C}

print("\nLoading models...")
for _aid, _defn in AGENT_DEFS.items():
    if _defn["algo"] == "baseline":
        continue
    _path = Path(_defn["path"])
    if not _path.exists():
        print(f"  WARNING: {_aid} — model not found ({_path.parent.parent.name})")
        continue
    _models[_aid] = _CLS[_defn["algo"]].load(str(_path).replace(".zip", ""))
    print(f"  ✅ {_aid} ({_defn['algo'].upper()} {_defn['variant']}): {_path.parent.parent.name}")
print(f"Models ready: {list(_models.keys())}\n")

with open(ROOT / "ml_training" / "config.yaml") as f:
    _BASE_CONFIG = yaml.safe_load(f)

ACTION_NAMES = ["maintain", "heat", "cool"]


# ── Per-connection session ────────────────────────────────────────────
class SimSession:
    """All simulation state for one WebSocket connection."""

    def __init__(self) -> None:
        self.running = False
        self.speed = 3
        self.compare_mode = False
        self.active_agent = "dqn"          # single mode
        self.compare_agents: list[str] = ["dqn", "ppo", "a2c"]  # compare mode
        self.manual_action = 0
        self.shared_seed = 42

        # Forced environment overrides (set by frontend scenarios)
        self.forced_target_day: float | None = None
        self.forced_ambient_temp: float | None = None
        self._default_ambient_temp: float = float(
            _BASE_CONFIG.get("simulation", {}).get("simulator", {}).get("ambient_temp_mean", 27.0)
        )

        # Single-agent state
        self.env: TomatoRipeningEnv | None = None
        self.obs: np.ndarray | None = None
        self.step_count = 0
        self.episode_history: list[dict] = []
        self.last_q_values: list[float] = [0.0, 0.0, 0.0]

        # Compare state
        self.cstates: dict[str, dict] = {}

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, seed: int | None = None) -> dict:
        seed = seed if seed is not None else random.randint(0, 99999)
        self.shared_seed = seed
        return self._reset_compare(seed) if self.compare_mode else self._reset_single(seed)

    def _patch_env(self, env: TomatoRipeningEnv) -> None:
        """Apply forced overrides to a freshly created (not yet reset) env."""
        if self.forced_target_day is not None:
            # Narrow range so rng.uniform returns exactly the forced value
            env.target_day_range = (self.forced_target_day, self.forced_target_day)
        if self.forced_ambient_temp is not None:
            env.simulator.config.ambient_temp_mean = float(self.forced_ambient_temp)

    def _reset_single(self, seed: int) -> dict:
        defn = AGENT_DEFS[self.active_agent]
        self.env = TomatoRipeningEnv(config=_BASE_CONFIG, state_variant=defn["variant"], seed=seed)
        self._patch_env(self.env)
        self.obs, _ = self.env.reset()
        self.step_count = 0
        self.episode_history = []
        self.last_q_values = [0.0, 0.0, 0.0]
        return self._single_msg("reset")

    def _reset_compare(self, seed: int) -> dict:
        self.cstates = {}
        for aid in self.compare_agents:
            defn = AGENT_DEFS[aid]
            env = TomatoRipeningEnv(config=_BASE_CONFIG, state_variant=defn["variant"], seed=seed)
            self._patch_env(env)
            obs, _ = env.reset()
            self.cstates[aid] = {
                "env": env, "obs": obs, "done": False,
                "step": 0, "total_reward": 0.0,
                "history": [], "final_info": None,
            }
        return self._compare_msg("compare_reset")

    # ── Step ──────────────────────────────────────────────────────────
    def step(self) -> dict:
        return self._step_compare() if self.compare_mode else self._step_single()

    def _get_action(self, aid: str, obs: np.ndarray) -> tuple[int, list[float]]:
        defn = AGENT_DEFS[aid]
        q_vals = [0.0, 0.0, 0.0]

        if defn["algo"] == "baseline":
            pol = defn["policy"]
            if pol == "fixed_day":
                action = 0
            elif pol == "fixed_stage5":
                action = 1 if float(obs[0]) > 0.3 else 0
            else:  # random
                action = random.randint(0, 2)
            return action, q_vals

        model = _models.get(aid)
        if model is None:
            return 0, q_vals
        act, _ = model.predict(obs, deterministic=True)
        action = int(act)
        if defn["algo"] == "dqn":
            with torch.no_grad():
                t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                q_vals = [float(v) for v in model.q_net(t).squeeze().tolist()]
        return action, q_vals

    def _step_single(self) -> dict:
        if self.env is None:
            return {"event": "error", "error": "No environment"}

        if self.active_agent == "manual":
            action, q_vals = self.manual_action, [0.0, 0.0, 0.0]
        else:
            action, q_vals = self._get_action(self.active_agent, self.obs)
        self.last_q_values = q_vals

        obs_new, reward, terminated, truncated, info = self.env.step(action)
        self.obs = obs_new
        self.step_count += 1
        sim = self.env.simulator
        self.episode_history.append({
            "hours": sim.hours_elapsed,
            "ripeness": float(sim.ripeness),
            "temperature": float(sim.temperature),
            "humidity": float(sim.humidity),
            "action": action,
            "reward": float(reward),
        })
        if terminated or truncated:
            self.running = False
            return self._single_msg("done", action=action, reward=reward, info=info)
        return self._single_msg("step", action=action, reward=reward, info=info)

    def _step_compare(self) -> dict:
        all_done = True
        for aid, cs in self.cstates.items():
            if cs["done"]:
                continue
            all_done = False
            action, _ = self._get_action(aid, cs["obs"])
            obs_new, reward, terminated, truncated, info = cs["env"].step(action)
            cs["obs"] = obs_new
            cs["step"] += 1
            cs["total_reward"] += float(reward)
            sim = cs["env"].simulator
            cs["history"].append({
                "hours": sim.hours_elapsed,
                "ripeness": float(sim.ripeness),
                "temperature": float(sim.temperature),
                "action": action,
                "reward": float(reward),
            })
            if terminated or truncated:
                cs["done"] = True
                cs["final_info"] = info

        if all_done:
            self.running = False
            return self._compare_msg("compare_done")
        return self._compare_msg("compare_step")

    # ── Message builders ──────────────────────────────────────────────
    def _single_msg(self, event: str, **kw) -> dict:
        if self.env is None:
            return {"event": event, "error": "no env"}
        sim = self.env.simulator
        defn = AGENT_DEFS.get(self.active_agent, {})
        hist = self.episode_history

        msg: dict = {
            "event": event, "mode": "single",
            "agentId": self.active_agent,
            "agentLabel": defn.get("label", self.active_agent),
            "agentColor": defn.get("color", "#888"),
            "seed": self.shared_seed,
            "step": self.step_count,
            "hours": round(float(sim.hours_elapsed), 2),
            "days": round(float(sim.hours_elapsed) / 24.0, 2),
            "targetDay": round(float(self.env.target_day), 1),
            "ripeness": round(float(sim.ripeness), 4),
            "ripenessStage": int(min(5, int((1.0 - float(sim.ripeness)) * 6))),
            "temperature": round(float(sim.temperature), 2),
            "humidity": round(float(sim.humidity), 2),
            "hourOfDay": round(sim.hours_elapsed % 24.0, 1),
            "quality": round(float(sim.compute_quality_score()), 4),
            "isOverripe": bool(float(sim.ripeness) < 0.05),
            "speed": self.speed,
            "forcedTargetDay": self.forced_target_day,
            "ambientTempMean": self.forced_ambient_temp if self.forced_ambient_temp is not None else self._default_ambient_temp,
            "totalReward": round(float(sum(h["reward"] for h in hist)), 2),
            "history": {
                "hours": [h["hours"] for h in hist[-200:]],
                "ripeness": [h["ripeness"] for h in hist[-200:]],
                "temperature": [h["temperature"] for h in hist[-200:]],
                "humidity": [h["humidity"] for h in hist[-200:]],
                "actions": [h["action"] for h in hist[-200:]],
                "rewards": [h["reward"] for h in hist[-200:]],
            },
            "observation": [float(v) for v in self.obs] if self.obs is not None else [],
            "qValues": self.last_q_values,
        }
        if "action" in kw:
            msg["action"] = ACTION_NAMES[int(kw["action"])]
            msg["actionId"] = int(kw["action"])
        if "reward" in kw:
            msg["reward"] = round(float(kw["reward"]), 4)
        if "info" in kw:
            info = kw["info"]
            if "harvest_quality" in info:
                msg["harvestQuality"] = round(float(info["harvest_quality"]), 4)
            if "timing_error" in info:
                msg["timingError"] = round(float(info["timing_error"]), 2)
        return msg

    def _compare_msg(self, event: str) -> dict:
        agents_out: dict[str, dict] = {}
        target_day = None

        for aid, cs in self.cstates.items():
            defn = AGENT_DEFS[aid]
            sim = cs["env"].simulator
            if target_day is None:
                target_day = round(float(cs["env"].target_day), 1)
            hist = cs["history"][-200:]
            last_action = int(hist[-1]["action"]) if hist else 0
            entry: dict = {
                "label": defn["label"],
                "color": defn["color"],
                "step": cs["step"],
                "days": round(sim.hours_elapsed / 24.0, 2),
                "ripeness": round(float(sim.ripeness), 4),
                "temperature": round(float(sim.temperature), 2),
                "action": last_action,
                "totalReward": round(cs["total_reward"], 2),
                "done": cs["done"],
                "history": {
                    "hours": [h["hours"] for h in hist],
                    "ripeness": [h["ripeness"] for h in hist],
                    "temperature": [h["temperature"] for h in hist],
                    "actions": [h["action"] for h in hist],
                },
            }
            if cs["done"] and cs["final_info"]:
                fi = cs["final_info"]
                entry["harvestQuality"] = round(float(fi.get("harvest_quality", 0)), 4)
                entry["timingError"] = round(float(fi.get("timing_error", 0)), 2)
                entry["harvestDay"] = round(sim.hours_elapsed / 24.0, 2)
            agents_out[aid] = entry

        return {
            "event": event, "mode": "compare",
            "targetDay": target_day,
            "seed": self.shared_seed,
            "agents": agents_out,
            "speed": self.speed,
        }

    # ── Catalogue helper (sent to frontend on connect) ─────────────────
    @staticmethod
    def catalogue_msg() -> dict:
        return {
            "event": "catalogue",
            "agents": {
                aid: {
                    "label": d["label"],
                    "color": d["color"],
                    "variant": d["variant"],
                    "available": d["algo"] == "baseline" or aid in _models,
                }
                for aid, d in AGENT_DEFS.items()
            },
        }


# ── WebSocket handler ─────────────────────────────────────────────────
async def handler(websocket):
    print(f"  Client connected: {websocket.remote_address}")
    session = SimSession()

    # Send catalogue + initial state
    await websocket.send(json.dumps(SimSession.catalogue_msg()))
    state = session.reset()
    await websocket.send(json.dumps(state))

    sim_task: asyncio.Task | None = None

    async def sim_loop():
        while session.running:
            for _ in range(session.speed):
                s = session.step()
                ev = s.get("event", "")
                if ev in ("done", "compare_done", "error"):
                    await websocket.send(json.dumps(s))
                    return
            await websocket.send(json.dumps(s))
            await asyncio.sleep(0.05)

    try:
        async for raw in websocket:
            msg = json.loads(raw)
            cmd = msg.get("cmd")

            if cmd == "start":
                if not session.running:
                    session.running = True
                    sim_task = asyncio.create_task(sim_loop())

            elif cmd == "pause":
                session.running = False
                if sim_task:
                    await sim_task
                    sim_task = None

            elif cmd == "reset":
                session.running = False
                if sim_task:
                    await sim_task
                    sim_task = None
                seed = msg.get("seed")  # optional pinned seed
                state = session.reset(seed)
                await websocket.send(json.dumps(state))

            elif cmd == "step":
                if not session.running:
                    s = session.step()
                    await websocket.send(json.dumps(s))

            elif cmd == "set_speed":
                session.speed = max(1, min(20, int(msg.get("speed", 3))))

            elif cmd == "set_action":
                session.manual_action = int(msg.get("action", 0))

            elif cmd == "set_single":
                session.running = False
                if sim_task:
                    sim_task.cancel()
                    sim_task = None
                session.compare_mode = False
                session.active_agent = msg.get("agent", "dqn")
                session.forced_target_day = msg.get("forced_target_day")
                session.forced_ambient_temp = msg.get("forced_ambient_temp")
                state = session.reset()
                await websocket.send(json.dumps(state))

            elif cmd == "set_compare":
                session.running = False
                if sim_task:
                    sim_task.cancel()
                    sim_task = None
                session.compare_mode = True
                session.compare_agents = msg.get("agents", ["dqn", "ppo", "a2c"])
                session.forced_target_day = msg.get("forced_target_day")
                session.forced_ambient_temp = msg.get("forced_ambient_temp")
                state = session.reset()
                await websocket.send(json.dumps(state))

            elif cmd == "set_forced_target":
                # Update target day and reset (used by the custom-target slider)
                session.forced_target_day = msg.get("target_day")
                session.running = False
                if sim_task:
                    sim_task.cancel()
                    sim_task = None
                seed = session.shared_seed if msg.get("same_seed") else None
                state = session.reset(seed)
                await websocket.send(json.dumps(state))
                if msg.get("auto_restart"):
                    session.running = True
                    sim_task = asyncio.create_task(sim_loop())

            elif cmd == "set_ambient_temp":
                # Live ambient temperature override — no reset needed
                new_temp = msg.get("temp")
                session.forced_ambient_temp = new_temp
                t = float(new_temp) if new_temp is not None else session._default_ambient_temp
                if session.env:
                    session.env.simulator.config.ambient_temp_mean = t
                for cs in session.cstates.values():
                    cs["env"].simulator.config.ambient_temp_mean = t

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        session.running = False
        if sim_task:
            sim_task.cancel()
        print(f"  Client disconnected")


_VIZ_DIR = Path(__file__).resolve().parent
_MIME = {
    ".html": "text/html; charset=utf-8",
    ".js":   "text/javascript; charset=utf-8",
    ".css":  "text/css; charset=utf-8",
    ".png":  "image/png",
    ".ico":  "image/x-icon",
    ".json": "application/json",
}


def _http_response(status: int, reason: str, ct: str, body: bytes) -> WsResponse:
    h = Headers([("Content-Type", ct), ("Content-Length", str(len(body)))])
    return WsResponse(status, reason, h, body)


def process_request(connection, request):
    """Serve static files on regular HTTP GET; let WebSocket upgrades through."""
    if "websocket" in request.headers.get("Upgrade", "").lower():
        return None
    path = request.path.split("?")[0]
    if path in ("/", ""):
        path = "/index.html"
    candidate = (_VIZ_DIR / path.lstrip("/")).resolve()
    # guard against path traversal
    if not str(candidate).startswith(str(_VIZ_DIR)):
        return _http_response(403, "Forbidden", "text/plain", b"Forbidden")
    if candidate.is_file():
        ct = _MIME.get(candidate.suffix.lower(), "application/octet-stream")
        return _http_response(200, "OK", ct, candidate.read_bytes())
    return _http_response(404, "Not Found", "text/plain", b"Not Found")


ACTION_NAMES_LIVE = ["maintain", "heat", "cool"]
ACTION_IDS_LIVE   = {"maintain": 0, "heat": 1, "cool": 2}


def _latest_sidecar(batch: int) -> Path:
    return ROOT / "outputs" / "experiments" / f"batch{batch}_latest.json"


async def live_handler(websocket, batch: int):
    """WebSocket handler for --live mode: streams ESP32 telemetry to the dashboard."""
    print(f"  [live] Client connected: {websocket.remote_address}")
    sidecar = _latest_sidecar(batch)
    last_mtime = 0.0

    try:
        while True:
            if not sidecar.exists():
                await asyncio.sleep(2)
                continue

            mtime = sidecar.stat().st_mtime
            if mtime <= last_mtime:
                await asyncio.sleep(2)
                continue
            last_mtime = mtime

            try:
                data = json.loads(sidecar.read_text())
            except (json.JSONDecodeError, OSError):
                await asyncio.sleep(2)
                continue

            action_str  = data.get("action", "maintain")
            action_id   = ACTION_IDS_LIVE.get(action_str, 0)
            chrom_x     = float(data.get("chromatic_x", 0.5))
            dx          = float(data.get("dx", 0.0))
            day         = float(data.get("day", 0))

            msg = {
                "event":        "step",
                "mode":         "live",
                "agentId":      "esp32",
                "agentLabel":   "ESP32-S3 (live)",
                "agentColor":   "#e67e22",
                "days":         day,
                "ripeness":     round(chrom_x, 4),
                "temperature":  round(float(data.get("temp", 0)), 2),
                "setpoint":     round(float(data.get("setpoint", 20)), 2),
                "humidity":     round(float(data.get("humidity", 0)), 2),
                "action":       action_str,
                "actionId":     action_id,
                "actionConf":   round(float(data.get("action_conf", 0)), 3),
                "classLabel":   data.get("class", "unknown"),
                "classConf":    round(float(data.get("class_conf", 0)), 2),
                "harvest":      data.get("harvest", False),
                "thermalFault": data.get("thermal_fault", False),
                "dx":           round(dx, 4),
                "uptime":       data.get("uptime", 0),
                "wallTime":     data.get("wall_time", ""),
            }
            await websocket.send(json.dumps(msg))
            await asyncio.sleep(2)

    except websockets.exceptions.ConnectionClosed:
        pass


async def main():
    import argparse as _ap
    p = _ap.ArgumentParser()
    p.add_argument("--live",  action="store_true", help="Stream live ESP32 data instead of simulation")
    p.add_argument("--batch", type=int, default=1, help="Batch number to read from (--live only)")
    p.add_argument("--port",  type=int, default=int(os.environ.get("PORT", 8080)))
    args = p.parse_args()

    port = args.port

    import functools as _functools
    if args.live:
        print(f"\n  [live] Streaming batch {args.batch} ESP32 data")
        print(f"  [live] Sidecar: {_latest_sidecar(args.batch)}")
        _handler = _functools.partial(live_handler, batch=args.batch)
    else:
        _handler = handler

    print(f"\n  Dashboard + WebSocket →  http://localhost:{port}\n")
    async with websockets.serve(_handler, "0.0.0.0", port, process_request=process_request):
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
