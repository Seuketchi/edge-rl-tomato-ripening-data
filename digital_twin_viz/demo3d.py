#!/usr/bin/env python3
"""
Edge-RL — Ripening Chamber Demo (Panda3D)

Two side-by-side chambers: DQN (left) vs a chosen baseline (right),
running on the same random seed so the comparison is fair.

Controls
  Space / P      Start / Pause
  R              New episode (random seed)
  UP / DOWN      Speed up / slow down
  B              Cycle opponent: Fixed-Day -> Random -> Fixed-Stage5 -> PPO -> A2C
  [ / ]          Target harvest day  -/+   (RANDOM <-> forced 3..7; applies on reset)
  - / =          Ambient temperature -/+   (LIVE -- perturb mid-run, policy reacts)
  . / N          Step one decision (while paused) -- narrate each action
  H              Heat shock: inject +6 C spike -- watch the policy recover
  L              Learning showdown: early DQN checkpoint vs final (needs a
                 --checkpoint-freq retrain; shows the agent before/after learning)
  S / F12        Save screenshot (PNG, venue-safe capture)
  0              Clear scenario overrides (back to RANDOM / default)
  Left-drag      Orbit camera
  Scroll wheel   Zoom in / out
  Q / Esc        Quit

Run:  python digital_twin_viz/demo3d.py [--size 1600x900]
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml_training.rl.distill import StudentPolicy
from ml_training.rl.environment import TomatoRipeningEnv

from direct.gui.OnscreenText import OnscreenText
from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import (
    AmbientLight, CardMaker, CullFaceAttrib,
    DirectionalLight, LColor, LineSegs,
    MouseButton, PointLight, TextNode, TransparencyAttrib, WindowProperties,
)

# ── Paths ─────────────────────────────────────────────────────────────
# Canonical distilled student (matches manuscript: 5443 params, 90.17% fidelity).
POLICY_PATH = ROOT / 'outputs' / 'distill_20260523_063646' / 'student_policy.pth'
CONFIG_PATH = ROOT / 'ml_training' / 'config.yaml'

ACTION_NAMES = ['MAINTAIN', 'HEAT', 'COOL']
ACTION_FG    = {
    0: (0.35, 1.00, 0.55, 1),
    1: (1.00, 0.58, 0.20, 1),
    2: (0.28, 0.62, 1.00, 1),
}
# Decision-strip colours (maintain=gray, heat=orange, cool=blue).
ACTION_STRIP = {
    0: (0.45, 0.47, 0.52),
    1: (0.95, 0.45, 0.20),
    2: (0.30, 0.58, 1.00),
}

# Baselines the right chamber can cycle through
BASELINES = [
    ('Fixed-Day',    lambda obs: 0),
    ('Random',       lambda obs: int(np.random.randint(0, 3))),
    ('Fixed-Stage5', lambda obs: 1 if float(obs[0]) > 0.3 else 0),
]
# Learned RL opponents (other algorithms) — appended to the B-cycle if their
# trained models are on disk. Shows DQN vs PPO / DQN vs A2C live.
LEARNED_OPPONENTS = [('PPO', 'ppo'), ('A2C', 'a2c')]


# ── Policy helpers ────────────────────────────────────────────────────
def load_policy() -> StudentPolicy:
    ckpt = torch.load(POLICY_PATH, map_location='cpu', weights_only=False)
    net  = StudentPolicy(ckpt['state_dim'], ckpt['action_dim'], ckpt['hidden_sizes'])
    net.load_state_dict(ckpt['model_state_dict'])
    net.eval()
    return net


@torch.no_grad()
def policy_act(net: StudentPolicy, obs: np.ndarray) -> int:
    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return int(net(t).argmax(dim=-1).item())


# ── Color mapping ─────────────────────────────────────────────────────
def ripen_color(x: float) -> tuple[float, float, float, float]:
    """X=1.0 → green (unripe),  X=0.15 → red (ripe)."""
    t = max(0.0, min(1.0, (x - 0.15) / 0.75))
    r = 0.80 * (0.25 + (1.0 - t) * 0.75)
    g = 0.75 * (0.12 + t * 0.88)
    b = 0.04
    return (r, g, b, 1.0)


# ── Main application ──────────────────────────────────────────────────
class EdgeRLDemo(ShowBase):
    CW, CH, CD  = 2.1, 1.7, 1.6   # chamber width, height, depth
    TICK_S      = 0.05             # wall-clock seconds per tick burst

    def __init__(self, win_size: tuple[int, int] = (1280, 720)) -> None:
        super().__init__()

        props = WindowProperties()
        props.setTitle('Edge-RL — Ripening Chamber Demo')
        props.setSize(int(win_size[0]), int(win_size[1]))
        self.win.requestProperties(props)
        self.setBackgroundColor(0.055, 0.060, 0.075, 1)
        self.disableMouse()

        # Camera: slightly elevated, looking between both chambers
        self.camera.setPos(0, -9.5, 2.8)
        self.camera.setHpr(0, -15, 0)

        print('Loading policy…')
        self._policy = load_policy()
        with open(CONFIG_PATH) as f:
            self._cfg = yaml.safe_load(f)
        print('Policy ready.')

        self._setup_lights()
        self._setup_environment()
        self._ch_l = self._make_chamber(-1.5, (0.25, 0.90, 0.45))
        self._ch_r = self._make_chamber( 1.5, (0.92, 0.48, 0.20))
        self._setup_hud()

        # Runtime state
        self._running       = False
        self._speed         = 3
        self._tick_acc      = 0.0
        self._env_l         = None
        self._env_r         = None
        self._obs_l         = None
        self._obs_r         = None
        self._done_l        = False
        self._done_r        = False
        self._last_act_l    = 0
        self._last_act_r    = 0
        self._target_day    = 5.0
        self._baseline_idx  = 0

        # Scenario overrides (None = default behaviour). Mirrors the
        # forced_target_day / forced_ambient_temp hooks in server.py so the
        # panel can pin any case live.
        self._forced_target_day: float | None = None
        self._forced_ambient: float | None = None
        _rl = self._cfg.get('rl', self._cfg)
        self._default_ambient = float(
            _rl.get('simulator', {}).get('ambient_temp_mean', 27.0))

        # Early-vs-late learning showdown (DQN checkpoints, loaded on demand)
        self._showdown    = False
        self._dqn_early   = None
        self._dqn_late    = None
        self._early_label = 'DQN early'
        self._late_label  = 'DQN final'

        # Learned opponents (PPO/A2C) loaded lazily when cycled to via B
        self._opp_models  = {}   # baseline_idx -> SB3 model

        # Graph history
        self._hist_l        = []   # list of (day, x, temp, action)
        self._hist_r        = []
        self._graph_node_l  = None
        self._graph_node_r  = None
        self._graph_tnode_l = None  # temperature/setpoint trace (right axis)
        self._graph_tnode_r = None
        self._graph_anode_l = None  # decision strip (HEAT/MAINTAIN/COOL)
        self._graph_anode_r = None
        self._graph_amb     = None  # ambient (real-life environment) line
        self._graph_tgt     = None  # target-day vertical line

        # Orbit camera state
        self._cam_h         = 0.0    # heading (azimuth)
        self._cam_p         = -15.0  # pitch (elevation)
        self._cam_dist      = 9.5
        self._orbit_last    = None   # (mx, my) of last drag frame

        self.accept('space',      self._toggle)
        self.accept('p',          self._toggle)
        self.accept('r',          self._reset)
        self.accept('b',          self._cycle_baseline)
        self.accept('[',          self._target_day_down)
        self.accept(']',          self._target_day_up)
        self.accept('-',          self._ambient_down)
        self.accept('=',          self._ambient_up)
        self.accept('0',          self._clear_scenario)
        self.accept('.',          self._step_paused)
        self.accept('n',          self._step_paused)
        self.accept('h',          self._heat_shock)
        self.accept('l',          self._toggle_showdown)
        self.accept('s',          self._screenshot)
        self.accept('f12',        self._screenshot)
        self.accept('arrow_up',   self._faster)
        self.accept('arrow_down', self._slower)
        self.accept('wheel_up',   self._zoom_in)
        self.accept('wheel_down', self._zoom_out)
        self.accept('escape',     sys.exit)
        self.accept('q',          sys.exit)

        self.taskMgr.add(self._tick, 'sim')
        self._setup_graph()
        self._reset()

    # ── Lighting ──────────────────────────────────────────────────────
    def _setup_lights(self) -> None:
        al = AmbientLight('amb')
        al.setColor(LColor(0.40, 0.46, 0.58, 1))
        self.render.setLight(self.render.attachNewNode(al))

        dl = DirectionalLight('sun')
        dl.setColor(LColor(0.90, 0.84, 0.72, 1))
        dlnp = self.render.attachNewNode(dl)
        dlnp.setHpr(25, -52, 0)
        self.render.setLight(dlnp)

    # ── Environment (bench + back wall, so chambers sit in a room) ─────
    def _setup_environment(self) -> None:
        floor_z = -self.CH / 2 - 0.02

        # Workbench surface under both chambers
        cm = CardMaker('bench')
        cm.setFrame(-7.0, 7.0, -5.5, 5.5)
        self._bench = self.render.attachNewNode(cm.generate())
        self._bench.setP(-90)           # lay flat
        self._bench.setZ(floor_z)

        # Back wall behind the chambers (gives depth / a "room")
        cm2 = CardMaker('backwall')
        cm2.setFrame(-7.0, 7.0, floor_z, 4.2)
        self._wall = self.render.attachNewNode(cm2.generate())
        self._wall.setY(2.6)            # behind chambers (depth half = 0.8)
        self._wall.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))

        # Initial tint (neutral ~25 °C); updated live by _update_backdrop().
        self._apply_backdrop(25.0)

    # Backdrop palette: effective ambient °C → (wall, bench, background) RGB.
    # Cold night-blue → neutral → hot sunset-orange.
    _BACKDROP_STOPS = [
        (12.0, (0.045, 0.065, 0.160), (0.060, 0.075, 0.115), (0.040, 0.050, 0.100)),
        (25.0, (0.070, 0.085, 0.110), (0.115, 0.100, 0.090), (0.055, 0.060, 0.075)),
        (38.0, (0.200, 0.100, 0.075), (0.170, 0.095, 0.070), (0.130, 0.070, 0.055)),
    ]

    @staticmethod
    def _lerp3(a, b, t):
        return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))

    def _apply_backdrop(self, temp: float) -> None:
        stops = self._BACKDROP_STOPS
        temp = max(stops[0][0], min(stops[-1][0], temp))
        wall, bench, bg = stops[0][1], stops[0][2], stops[0][3]
        for i in range(len(stops) - 1):
            t0, w0, b0, g0 = stops[i]
            t1, w1, b1, g1 = stops[i + 1]
            if temp <= t1:
                f = (temp - t0) / (t1 - t0) if t1 > t0 else 0.0
                wall  = self._lerp3(w0, w1, f)
                bench = self._lerp3(b0, b1, f)
                bg    = self._lerp3(g0, g1, f)
                break
        self._wall.setColor(*wall, 1)
        self._bench.setColor(*bench, 1)
        self.setBackgroundColor(*bg, 1)

    @staticmethod
    def _effective_ambient(sim) -> float:
        """Ambient the chamber feels: mean + day/night swing (no noise)."""
        cfg = sim.config
        hour = sim.hours_elapsed % 24.0
        diurnal = cfg.diurnal_amplitude * math.sin(
            2.0 * math.pi * (hour - cfg.diurnal_peak_hour + 6.0) / 24.0)
        return cfg.ambient_temp_mean + diurnal

    def _update_backdrop(self) -> None:
        """Tint the room by the effective ambient (mean + diurnal swing),
        mirroring the simulator's day/night cycle."""
        if self._env_l is None:
            return
        self._apply_backdrop(self._effective_ambient(self._env_l.simulator))

    # ── Chamber geometry ──────────────────────────────────────────────
    def _make_chamber(self, x: float, accent: tuple) -> dict:
        W, H, D = self.CW, self.CH, self.CD
        hy, hz  = H / 2, D / 2

        root = self.render.attachNewNode('chamber')
        root.setX(x)

        # Wireframe outline
        ls = LineSegs()
        ls.setColor(*accent, 0.70)
        ls.setThickness(1.6)
        pts = [
            (-W/2, -hz, -hy), ( W/2, -hz, -hy),
            ( W/2,  hz, -hy), (-W/2,  hz, -hy),
            (-W/2, -hz,  hy), ( W/2, -hz,  hy),
            ( W/2,  hz,  hy), (-W/2,  hz,  hy),
        ]
        for a, b in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),
                     (0,4),(1,5),(2,6),(3,7)]:
            ls.moveTo(*pts[a]); ls.drawTo(*pts[b])
        root.attachNewNode(ls.create())

        # Floor slab
        cm = CardMaker('floor')
        cm.setFrame(-W/2, W/2, -hz, hz)
        floor = root.attachNewNode(cm.generate())
        floor.setP(-90); floor.setZ(-hy)
        floor.setColor(0.055, 0.080, 0.150, 1)

        # Transparent walls (back + sides)
        def wall(w, h, pos, hpr, alpha):
            c = CardMaker('wall')
            c.setFrame(-w/2, w/2, -h/2, h/2)
            n = root.attachNewNode(c.generate())
            n.setPos(*pos); n.setHpr(*hpr)
            n.setColor(0.20, 0.30, 0.55, alpha)
            n.setTransparency(TransparencyAttrib.MAlpha)
            n.setAttrib(CullFaceAttrib.make(CullFaceAttrib.MCullNone))
            return n

        wall(W, H, (0,  hz, 0), (0,   0, 0), 0.08)   # back
        wall(D, H, (-W/2, 0, 0), (90, 0, 0), 0.06)   # left
        wall(D, H, ( W/2, 0, 0), (90, 0, 0), 0.06)   # right

        # Tomato sphere
        try:
            tomato = self.loader.loadModel('models/misc/sphere')
            tomato.setScale(0.30)
            tomato.reparentTo(root)
        except Exception:
            # Fallback: billboard quad
            cm2 = CardMaker('tomato')
            cm2.setFrame(-0.28, 0.28, -0.28, 0.28)
            tomato = root.attachNewNode(cm2.generate())
        tomato.setPos(0, 0, -hy + 0.34)
        tomato.setColor(*ripen_color(0.90))

        # Stem (short green line segment)
        ls2 = LineSegs()
        ls2.setColor(0.20, 0.58, 0.12, 1); ls2.setThickness(3.5)
        ls2.moveTo(0, 0, -hy + 0.64); ls2.drawTo(0, 0.02, -hy + 0.76)
        stem_np = root.attachNewNode(ls2.create())

        # Heater bar — bottom-back
        cm3 = CardMaker('heater')
        cm3.setFrame(-W * 0.40, W * 0.40, -0.03, 0.03)
        heater_np = root.attachNewNode(cm3.generate())
        heater_np.setPos(0, hz - 0.05, -hy + 0.05)
        heater_np.setColor(0.12, 0.12, 0.12, 1)

        # Cooler panel — top-back
        cm4 = CardMaker('cooler')
        cm4.setFrame(-0.42, 0.42, -0.04, 0.04)
        cooler_np = root.attachNewNode(cm4.generate())
        cooler_np.setPos(0, hz - 0.05, hy - 0.09)
        cooler_np.setColor(0.12, 0.12, 0.12, 1)

        # Glow point lights (heat=orange, cool=blue)
        hl = PointLight('hl'); hl.setColor(LColor(0, 0, 0, 1))
        hlnp = root.attachNewNode(hl)
        hlnp.setPos(0, hz - 0.18, -hy + 0.28)
        root.setLight(hlnp)

        cl = PointLight('cl'); cl.setColor(LColor(0, 0, 0, 1))
        clnp = root.attachNewNode(cl)
        clnp.setPos(0, hz - 0.18, hy - 0.28)
        root.setLight(clnp)

        return dict(
            root=root, tomato=tomato, stem=stem_np,
            heater=heater_np, cooler=cooler_np,
            hl=hl, cl=cl, base_z=-hy + 0.34,
        )

    def _update_chamber(self, ch: dict, x: float, action: int) -> None:
        ch['tomato'].setColor(*ripen_color(x))

        if action == 1:   # HEAT
            ch['heater'].setColor(0.88, 0.30, 0.04, 1)
            ch['hl'].setColor(LColor(1.5, 0.42, 0.0, 1))
        else:
            ch['heater'].setColor(0.10, 0.10, 0.10, 1)
            ch['hl'].setColor(LColor(0, 0, 0, 1))

        if action == 2:   # COOL
            ch['cooler'].setColor(0.10, 0.38, 0.95, 1)
            ch['cl'].setColor(LColor(0.28, 0.62, 1.5, 1))
        else:
            ch['cooler'].setColor(0.10, 0.10, 0.10, 1)
            ch['cl'].setColor(LColor(0, 0, 0, 1))

    # ── HUD ───────────────────────────────────────────────────────────
    def _setup_hud(self) -> None:
        def label(text, pos, scale=0.044, fg=(0.75, 0.85, 0.95, 1), align=TextNode.ALeft):
            return OnscreenText(text=text, pos=pos, scale=scale, fg=fg,
                                shadow=(0,0,0,0.75), shadowOffset=(0.002,-0.002),
                                align=align, mayChange=True)

        OnscreenText(
            text='Edge-RL -- Post-Harvest Ripening Control -- Chamber Demo',
            pos=(0, 0.93), scale=0.050,
            fg=(0.55, 0.82, 1.0, 1), shadow=(0,0,0,0.8), shadowOffset=(0.002,-0.002),
            align=TextNode.ACenter,
        )
        OnscreenText(
            text='[Space] Pause  [./N] Step  [H] Heat-shock  [L] Learning showdown  '
                 '[-/=] Ambient(LIVE)  [ [ ] ] Target  [B] Baseline  [S] Shot  [R] Reset  [Q] Quit',
            pos=(0, -0.980), scale=0.028, fg=(0.45, 0.60, 0.72, 1),
            align=TextNode.ACenter,
        )

        # Scenario readout (under the title) — shows the active case to the panel
        self._scenario_txt = OnscreenText(
            text='', pos=(0, 0.87), scale=0.033, fg=(0.95, 0.82, 0.45, 1),
            shadow=(0,0,0,0.8), shadowOffset=(0.002,-0.002),
            align=TextNode.ACenter, mayChange=True,
        )

        # Left (DQN) — far-left column
        self._txt_l_name = label('DQN', (-1.42, 0.74), 0.060, fg=(0.30, 1.00, 0.52, 1))
        self._lX   = label('X: —',    (-1.42, 0.64))
        self._lT   = label('T: —',    (-1.42, 0.55))
        self._lDay = label('Day: —',  (-1.42, 0.46))
        self._lAct = label('—',       (-1.42, 0.36), 0.052, fg=ACTION_FG[0])

        # Right (Baseline — label changes when B is pressed) — far-right column
        self._txt_r_name = label('Fixed-Day', (0.62, 0.74), 0.060, fg=(0.95, 0.52, 0.22, 1))
        self._rX   = label('X: —',    (0.62, 0.64))
        self._rT   = label('T: —',    (0.62, 0.55))
        self._rDay = label('Day: —',  (0.62, 0.46))
        self._rAct = label('—',       (0.62, 0.36), 0.052, fg=ACTION_FG[0])

        self._spd_txt = label('Speed: 3x',  (-1.55, -0.50), 0.038, fg=(0.55, 0.65, 0.75, 1))
        self._status  = label('READY',      (0, -0.50), 0.050,
                               fg=(0.50, 0.65, 0.55, 1), align=TextNode.ACenter)

    def _refresh_hud(self) -> None:
        if self._env_l:
            sim = self._env_l.simulator
            x   = float(sim.ripeness)
            d   = sim.hours_elapsed / 24.0
            a   = self._last_act_l
            self._lX.setText(f'X = {x:.3f}')
            self._lT.setText(f'T = {float(sim.temperature):.1f} °C    H = {float(sim.humidity):.0f}%')
            self._lDay.setText(f'Day {d:.1f} / {self._target_day:.1f}')
            self._lAct.setText(ACTION_NAMES[a])
            self._lAct['fg'] = ACTION_FG[a]

        if self._env_r:
            sim = self._env_r.simulator
            x   = float(sim.ripeness)
            d   = sim.hours_elapsed / 24.0
            a   = self._last_act_r
            self._rX.setText(f'X = {x:.3f}')
            self._rT.setText(f'T = {float(sim.temperature):.1f} °C    H = {float(sim.humidity):.0f}%')
            self._rDay.setText(f'Day {d:.1f} / {self._target_day:.1f}')
            self._rAct.setText(ACTION_NAMES[a])
            self._rAct['fg'] = ACTION_FG[a]

        self._spd_txt.setText(f'Speed: {self._speed}x')

    # ── Controls ──────────────────────────────────────────────────────
    def _reset(self) -> None:
        self._running  = False
        self._done_l   = self._done_r = False
        self._last_act_l = self._last_act_r = 0

        seed = np.random.randint(0, 99999)
        self._env_l = TomatoRipeningEnv(config=self._cfg, state_variant='B', seed=seed)
        self._apply_scenario(self._env_l)
        self._obs_l, _ = self._env_l.reset()
        self._target_day = float(self._env_l.target_day)

        self._env_r = TomatoRipeningEnv(config=self._cfg, state_variant='B', seed=seed)
        self._apply_scenario(self._env_r)
        self._obs_r, _ = self._env_r.reset()

        self._hist_l.clear()
        self._hist_r.clear()
        self._update_chamber(self._ch_l, float(self._env_l.simulator.ripeness), 0)
        self._update_chamber(self._ch_r, float(self._env_r.simulator.ripeness), 0)
        self._refresh_hud()
        self._refresh_scenario_hud()
        self._redraw_graph()
        self._update_backdrop()
        self._status.setText('READY    [Space] to start')
        self._status['fg'] = (0.50, 0.68, 0.56, 1)

    def _toggle(self) -> None:
        if self._done_l and self._done_r:
            self._reset(); return
        self._running = not self._running
        if self._running:
            self._status.setText('RUNNING')
            self._status['fg'] = (0.30, 1.00, 0.52, 1)
        else:
            self._status.setText('PAUSED')
            self._status['fg'] = (1.00, 0.78, 0.28, 1)

    def _faster(self) -> None: self._speed = min(12, self._speed + 1)
    def _slower(self) -> None: self._speed = max(1,  self._speed - 1)

    def _zoom_in(self)  -> None: self._cam_dist = max(3.5,  self._cam_dist - 0.6); self._apply_cam()
    def _zoom_out(self) -> None: self._cam_dist = min(20.0, self._cam_dist + 0.6); self._apply_cam()

    def _apply_cam(self) -> None:
        rh = math.radians(self._cam_h)
        rp = math.radians(self._cam_p)
        x  =  self._cam_dist * math.sin(rh) * math.cos(rp)
        y  = -self._cam_dist * math.cos(rh) * math.cos(rp)
        z  =  self._cam_dist * math.sin(-rp) + 0.4
        self.camera.setPos(x, y, z)
        self.camera.lookAt(0, 0, 0.3)

    def _opponent_name(self, idx: int) -> str:
        if idx < len(BASELINES):
            return BASELINES[idx][0]
        return LEARNED_OPPONENTS[idx - len(BASELINES)][0]

    @staticmethod
    def _algo_model_path(algo: str):
        for run in sorted(ROOT.glob('outputs/algo_comparison_*'), reverse=True):
            p = run / algo / 'best_model' / 'best_model.zip'
            if p.exists():
                return p
        return None

    def _ensure_opp_loaded(self, idx: int) -> bool:
        """Lazy-load a learned opponent's model; False if it's not on disk."""
        if idx in self._opp_models:
            return True
        algo = LEARNED_OPPONENTS[idx - len(BASELINES)][1]
        path = self._algo_model_path(algo)
        if path is None:
            return False
        from stable_baselines3 import PPO, A2C   # lazy import
        self._opp_models[idx] = {'ppo': PPO, 'a2c': A2C}[algo].load(str(path))
        return True

    def _cycle_baseline(self) -> None:
        if self._showdown:           # opponents are hidden during the showdown
            return
        total = len(BASELINES) + len(LEARNED_OPPONENTS)
        idx = self._baseline_idx
        for _ in range(total):       # advance, skipping learned models not on disk
            idx = (idx + 1) % total
            if idx < len(BASELINES) or self._ensure_opp_loaded(idx):
                break
        self._baseline_idx = idx
        name = self._opponent_name(idx)
        self._txt_r_name.setText(name)
        self._legend_baseline.setText(name)
        if not self._running:
            self._reset()

    # ── Early-vs-late learning showdown ───────────────────────────────
    def _discover_showdown_models(self):
        """Newest rl_* run with an early + late policy. Late = final_model if
        present, else the latest checkpoint (so it works mid-training).
        Returns (early_path, late_path, early_steps, late_tag) or (None,)*4."""
        runs = sorted(ROOT.glob('outputs/rl_*'), reverse=True)
        for run in runs:
            ckpts = sorted(run.glob('checkpoints/dqn_*_steps.zip'),
                           key=lambda p: int(p.stem.split('_')[1]))
            if not ckpts:
                continue
            early = ckpts[0]
            e_steps = int(early.stem.split('_')[1])
            final = run / 'final_model.zip'
            if final.exists():
                return early, final, e_steps, 'final'
            if len(ckpts) >= 2:
                l_steps = int(ckpts[-1].stem.split('_')[1])
                return early, ckpts[-1], e_steps, f'{l_steps // 1000}K'
        return None, None, 0, None

    def _toggle_showdown(self) -> None:
        if not self._showdown:
            early, late, steps, late_tag = self._discover_showdown_models()
            if early is None:
                self._status.setText('No checkpoints — train with --checkpoint-freq')
                self._status['fg'] = (1.0, 0.6, 0.3, 1)
                return
            from stable_baselines3 import DQN   # lazy: only when needed
            self._dqn_early = DQN.load(str(early))
            self._dqn_late = DQN.load(str(late))
            self._early_label = f'DQN @{steps // 1000}K'
            self._late_label = 'DQN @1M' if late_tag == 'final' else f'DQN @{late_tag}'
            self._showdown = True
        else:
            self._showdown = False
        self._apply_mode_labels()
        self._reset()

    def _apply_mode_labels(self) -> None:
        if self._showdown:
            self._txt_l_name.setText(self._early_label)
            self._txt_r_name.setText(self._late_label)
            self._legend_dqn.setText(self._early_label)
            self._legend_baseline.setText(self._late_label)
        else:
            self._txt_l_name.setText('DQN')
            name = self._opponent_name(self._baseline_idx)
            self._txt_r_name.setText(name)
            self._legend_dqn.setText('DQN')
            self._legend_baseline.setText(name)

    # ── Scenario overrides (let the panel pin any case) ───────────────
    def _apply_scenario(self, env) -> None:
        """Pin a freshly-built (not yet reset) env to the forced case."""
        if self._forced_target_day is not None:
            # Narrow the range so reset() draws exactly the forced value.
            env.target_day_range = (self._forced_target_day, self._forced_target_day)
        if self._forced_ambient is not None:
            env.simulator.config.ambient_temp_mean = float(self._forced_ambient)

    def _target_day_step(self, delta: float) -> None:
        base = self._forced_target_day if self._forced_target_day is not None else 5.0
        self._forced_target_day = float(min(7.0, max(3.0, round(base + delta))))
        self._scenario_changed()

    def _target_day_down(self) -> None: self._target_day_step(-1.0)
    def _target_day_up(self)   -> None: self._target_day_step(+1.0)

    def _ambient_step(self, delta: float) -> None:
        base = self._forced_ambient if self._forced_ambient is not None else self._default_ambient
        self._forced_ambient = float(min(38.0, max(15.0, base + delta)))
        # Apply LIVE to the running chambers (no reset) so the policy reacts in
        # real time — mirrors server.py's set_ambient_temp live override.
        for env in (self._env_l, self._env_r):
            if env is not None:
                env.simulator.config.ambient_temp_mean = self._forced_ambient
        self._refresh_scenario_hud()
        self._update_backdrop()

    def _ambient_down(self) -> None: self._ambient_step(-2.0)
    def _ambient_up(self)   -> None: self._ambient_step(+2.0)

    def _clear_scenario(self) -> None:
        self._forced_target_day = None
        self._forced_ambient = None
        self._scenario_changed()

    def _scenario_changed(self) -> None:
        # Re-init both chambers under the new case (pauses at READY) so the
        # override is applied from step 0 and visible to the panel.
        self._reset()

    def _refresh_scenario_hud(self) -> None:
        td = 'RANDOM' if self._forced_target_day is None else f'{self._forced_target_day:.0f}'
        if self._forced_ambient is None:
            amb = f'{self._default_ambient:.0f} °C (default)'
        else:
            amb = f'{self._forced_ambient:.0f} °C (forced)'
        self._scenario_txt.setText(f'Target day: {td}      Ambient: {amb}')

    # ── Stepping primitives ───────────────────────────────────────────
    def _act_left(self, obs) -> int:
        if self._showdown:
            return int(self._dqn_early.predict(obs, deterministic=True)[0])
        return policy_act(self._policy, obs)

    def _act_right(self, obs) -> int:
        if self._showdown:
            return int(self._dqn_late.predict(obs, deterministic=True)[0])
        idx = self._baseline_idx
        if idx < len(BASELINES):
            return BASELINES[idx][1](obs)
        model = self._opp_models.get(idx)          # learned opponent (PPO/A2C)
        if model is None:
            return 0
        return int(model.predict(obs, deterministic=True)[0])

    def _step_envs(self) -> None:
        """Advance both chambers by one decision step (no rendering)."""
        if not self._done_l:
            a = self._act_left(self._obs_l)
            self._obs_l, _, tl, ul, _ = self._env_l.step(a)
            self._last_act_l = a
            if tl or ul: self._done_l = True

        if not self._done_r:
            a_r = self._act_right(self._obs_r)
            self._obs_r, _, tr, ur, _ = self._env_r.step(a_r)
            self._last_act_r = a_r
            if tr or ur: self._done_r = True

    def _render_state(self) -> None:
        """Push current sim state to chambers, HUD and graph."""
        x_l = float(self._env_l.simulator.ripeness)
        x_r = float(self._env_r.simulator.ripeness)
        d_l = self._env_l.simulator.hours_elapsed / 24.0
        d_r = self._env_r.simulator.hours_elapsed / 24.0
        t_l = float(self._env_l.simulator.temperature)
        t_r = float(self._env_r.simulator.temperature)
        amb = self._effective_ambient(self._env_l.simulator)   # shared environment
        self._hist_l.append((d_l, x_l, t_l, self._last_act_l, amb))
        self._hist_r.append((d_r, x_r, t_r, self._last_act_r, amb))
        self._update_chamber(self._ch_l, x_l, self._last_act_l)
        self._update_chamber(self._ch_r, x_r, self._last_act_r)
        self._refresh_hud()
        self._redraw_graph()
        self._update_backdrop()

        if self._done_l and self._done_r:
            self._running = False
            self._status.setText('DONE    [Space] replay')
            self._status['fg'] = (0.95, 0.88, 0.30, 1)

    def _step_paused(self) -> None:
        """Advance exactly one decision step while paused (for narration)."""
        if self._running or (self._done_l and self._done_r):
            return
        self._step_envs()
        self._render_state()
        if not (self._done_l and self._done_r):
            self._status.setText('STEPPED  (paused)')
            self._status['fg'] = (0.65, 0.80, 1.00, 1)

    def _heat_shock(self) -> None:
        """Inject an instantaneous +6 °C chamber spike into both chambers
        (live disturbance) so the panel can watch the policy recover."""
        if self._env_l is None:
            return
        for env in (self._env_l, self._env_r):
            sim = env.simulator
            sim.temperature = float(min(40.0, sim.temperature + 6.0))
        self._refresh_hud()
        self._status.setText('HEAT SHOCK  +6 °C')
        self._status['fg'] = (1.00, 0.45, 0.30, 1)

    def _screenshot(self) -> None:
        """Save a PNG of the current frame (venue-safe capture / fallback)."""
        fn = self.screenshot('edge_rl_demo')
        if fn:
            self._status.setText(f'SAVED  {fn}')
            self._status['fg'] = (0.65, 0.95, 0.70, 1)

    # ── Simulation tick ───────────────────────────────────────────────
    def _tick(self, task) -> int:
        # Mouse orbit
        if self.mouseWatcherNode.hasMouse():
            md = self.win.getPointer(0)
            mx, my = md.getX(), md.getY()
            if self.mouseWatcherNode.isButtonDown(MouseButton.one()):
                if self._orbit_last is not None:
                    dx = mx - self._orbit_last[0]
                    dy = my - self._orbit_last[1]
                    self._cam_h  = (self._cam_h - dx * 0.35) % 360
                    self._cam_p  = max(-75, min(5, self._cam_p + dy * 0.28))
                    self._apply_cam()
                self._orbit_last = (mx, my)
            else:
                self._orbit_last = None

        # Gentle tomato bob (always running, regardless of sim state)
        t   = globalClock.getFrameTime()
        bob = math.sin(t * 1.15) * 0.013
        if self._env_l:
            self._ch_l['tomato'].setZ(self._ch_l['base_z'] + bob)
        if self._env_r:
            self._ch_r['tomato'].setZ(self._ch_r['base_z'] + bob)

        if not self._running:
            return Task.cont

        self._tick_acc += globalClock.getDt()
        if self._tick_acc < self.TICK_S:
            return Task.cont
        self._tick_acc = 0.0

        for _ in range(self._speed):
            if self._done_l and self._done_r:
                break
            self._step_envs()
        self._render_state()

        return Task.cont


    # ── Graph ─────────────────────────────────────────────────────────
    # Panel occupies bottom strip of screen (aspect2d coords)
    GX0, GX1 = -1.58, 1.58
    GZ0, GZ1 = -0.86, -0.58       # plot area; decision strips sit below GZ0
    STRIP_ZL, STRIP_ZR = -0.875, -0.900   # action strips: left chamber (upper), right (lower)

    def _setup_graph(self) -> None:
        # Dark background
        cm = CardMaker('graph_bg')
        cm.setFrame(self.GX0, self.GX1, self.GZ0, self.GZ1)
        bg = self.aspect2d.attachNewNode(cm.generate())
        bg.setColor(0.03, 0.05, 0.10, 0.92)
        bg.setTransparency(TransparencyAttrib.MAlpha)

        # Grid lines (horizontal, at X=0.25 / 0.5 / 0.75)
        ls = LineSegs(); ls.setThickness(1.0)
        for val, alpha in [(0.25, 0.18), (0.5, 0.22), (0.75, 0.18)]:
            ls.setColor(0.30, 0.40, 0.55, alpha)
            z = self._gz(val)
            ls.moveTo(self.GX0 + 0.06, 0, z)
            ls.drawTo(self.GX1 - 0.02, 0, z)
        # Axes
        ls.setColor(0.30, 0.42, 0.58, 0.55)
        ls.moveTo(self.GX0 + 0.06, 0, self.GZ0 + 0.05)
        ls.drawTo(self.GX0 + 0.06, 0, self.GZ1 - 0.02)  # Y axis
        ls.moveTo(self.GX0 + 0.06, 0, self.GZ0 + 0.05)
        ls.drawTo(self.GX1 - 0.02, 0, self.GZ0 + 0.05)  # X axis
        self.aspect2d.attachNewNode(ls.create())

        # Harvest threshold line (X = 0.15, red dashed look)
        ls2 = LineSegs(); ls2.setThickness(1.2)
        ls2.setColor(0.85, 0.28, 0.28, 0.65)
        z = self._gz(0.15)
        ls2.moveTo(self.GX0 + 0.06, 0, z)
        ls2.drawTo(self.GX1 - 0.02, 0, z)
        self.aspect2d.attachNewNode(ls2.create())

        # Labels
        def tiny(text, x, z, fg=(0.45, 0.58, 0.70, 1)):
            OnscreenText(text=text, pos=(x, z), scale=0.030,
                         fg=fg, shadow=(0,0,0,0.6), shadowOffset=(0.001,-0.001),
                         align=TextNode.ALeft)

        tiny('1.0', self.GX0 + 0.01, self._gz(1.0) - 0.01)
        tiny('0.5', self.GX0 + 0.01, self._gz(0.5) - 0.01)
        tiny('0.15', self.GX0 + 0.01, self._gz(0.15) - 0.01, fg=(0.85, 0.38, 0.38, 1))
        tiny('thick = X (chromatic index)     thin = T setpoint (°C)     faint = ambient (real-life)',
             self.GX0 + 0.08, self.GZ1 - 0.04)
        tiny('harvest threshold', self.GX0 + 0.45, self._gz(0.15) + 0.01, fg=(0.80, 0.35, 0.35, 0.8))

        # Right axis — temperature scale (matches the thin traces)
        tcol = (0.72, 0.82, 0.66, 0.95)
        for tv in (15, 25, 35):
            tiny(f'{tv}°', self.GX1 - 0.10, self._gz_temp(tv) - 0.01, fg=tcol)
        # Day axis labels drawn after target_day is known (in _redraw_graph)

        # Decision strips: left-margin accent ticks mark which chamber each row is
        for z, accent in ((self.STRIP_ZL, (0.30, 1.00, 0.52)),
                          (self.STRIP_ZR, (0.95, 0.52, 0.22))):
            tk = LineSegs(); tk.setThickness(7.0); tk.setColor(*accent, 1.0)
            tk.moveTo(self.GX0 + 0.005, 0, z); tk.drawTo(self.GX0 + 0.045, 0, z)
            self.aspect2d.attachNewNode(tk.create())

        # Action colour key on its own row, clear of the strips
        tiny('decisions per step:', self.GX0 + 0.08, -0.945, fg=(0.55, 0.62, 0.70, 0.9))
        keys = [('maintain', ACTION_STRIP[0]), ('heat', ACTION_STRIP[1]), ('cool', ACTION_STRIP[2])]
        kx = self.GX0 + 0.62
        for label_txt, col in keys:
            sw = LineSegs(); sw.setThickness(6.0); sw.setColor(*col, 1.0)
            sw.moveTo(kx, 0, -0.943); sw.drawTo(kx + 0.05, 0, -0.943)
            self.aspect2d.attachNewNode(sw.create())
            tiny(label_txt, kx + 0.065, -0.952, fg=(0.62, 0.66, 0.72, 0.95))
            kx += 0.27

        # Legend
        ls3 = LineSegs(); ls3.setThickness(2.5)
        ls3.setColor(0.30, 1.00, 0.52, 1)
        lx = self.GX1 - 0.50
        lz = self.GZ1 - 0.04
        ls3.moveTo(lx, 0, lz); ls3.drawTo(lx + 0.08, 0, lz)
        self.aspect2d.attachNewNode(ls3.create())
        self._legend_dqn = OnscreenText(text='DQN', pos=(lx + 0.10, lz), scale=0.030,
                                        fg=(0.30, 1.00, 0.52, 1), align=TextNode.ALeft,
                                        mayChange=True)

        ls4 = LineSegs(); ls4.setThickness(2.5)
        ls4.setColor(0.95, 0.52, 0.22, 1)
        lx2 = lx + 0.30
        ls4.moveTo(lx2, 0, lz); ls4.drawTo(lx2 + 0.08, 0, lz)
        self.aspect2d.attachNewNode(ls4.create())
        self._legend_baseline = OnscreenText(text='Fixed-Day', pos=(lx2 + 0.10, lz),
                                              scale=0.030, fg=(0.95, 0.52, 0.22, 1),
                                              align=TextNode.ALeft, mayChange=True)

    def _gx(self, day: float) -> float:
        inner = (self.GX1 - 0.02) - (self.GX0 + 0.06)
        return self.GX0 + 0.06 + (day / 8.0) * inner

    def _gz(self, x_val: float) -> float:
        inner = (self.GZ1 - 0.02) - (self.GZ0 + 0.05)
        return self.GZ0 + 0.05 + x_val * inner

    # Right-axis temperature scale: 12.5 °C (t_base) .. 40 °C (safety ceiling)
    TEMP_MIN, TEMP_MAX = 12.5, 40.0

    def _gz_temp(self, temp: float) -> float:
        inner = (self.GZ1 - 0.02) - (self.GZ0 + 0.05)
        frac = (temp - self.TEMP_MIN) / (self.TEMP_MAX - self.TEMP_MIN)
        frac = max(0.0, min(1.0, frac))
        return self.GZ0 + 0.05 + frac * inner

    def _redraw_graph(self) -> None:
        for attr in ('_graph_node_l', '_graph_node_r',
                     '_graph_tnode_l', '_graph_tnode_r',
                     '_graph_anode_l', '_graph_anode_r', '_graph_amb', '_graph_tgt'):
            node = getattr(self, attr)
            if node:
                node.removeNode(); setattr(self, attr, None)

        # Target-day vertical line
        ls0 = LineSegs(); ls0.setThickness(1.2)
        ls0.setColor(0.90, 0.80, 0.30, 0.60)
        gx = self._gx(self._target_day)
        ls0.moveTo(gx, 0, self.GZ0 + 0.05)
        ls0.drawTo(gx, 0, self.GZ1 - 0.02)
        self._graph_tgt = self.aspect2d.attachNewNode(ls0.create())

        def polyline(hist, value_fn, color, thick):
            if len(hist) < 2:
                return None
            ls = LineSegs(); ls.setThickness(thick)
            ls.setColor(*color)
            ls.moveTo(self._gx(hist[0][0]), 0, value_fn(hist[0]))
            for pt in hist[1:]:
                ls.drawTo(self._gx(pt[0]), 0, value_fn(pt))
            return self.aspect2d.attachNewNode(ls.create())

        # Chromatic index — thick, bright (left axis)
        self._graph_node_l = polyline(
            self._hist_l, lambda p: self._gz(p[1]), (0.30, 1.00, 0.52, 1), 2.2)
        self._graph_node_r = polyline(
            self._hist_r, lambda p: self._gz(p[1]), (0.95, 0.52, 0.22, 1), 2.2)
        # Ambient (real-life environment) — faint dashed-feel line on the °C axis
        self._graph_amb = polyline(
            self._hist_l, lambda p: self._gz_temp(p[4]), (0.85, 0.80, 0.45, 0.40), 1.1)

        # Temperature/setpoint — thin, dimmed (right axis)
        self._graph_tnode_l = polyline(
            self._hist_l, lambda p: self._gz_temp(p[2]), (0.45, 0.95, 0.62, 0.55), 1.3)
        self._graph_tnode_r = polyline(
            self._hist_r, lambda p: self._gz_temp(p[2]), (0.98, 0.66, 0.42, 0.55), 1.3)

        # Decision strips — per-step HEAT/MAINTAIN/COOL colour bars
        def action_strip(hist, z):
            if len(hist) < 2:
                return None
            ls = LineSegs(); ls.setThickness(7.0)
            for i in range(len(hist) - 1):
                ls.setColor(*ACTION_STRIP[int(hist[i][3])], 1.0)
                ls.moveTo(self._gx(hist[i][0]), 0, z)
                ls.drawTo(self._gx(hist[i + 1][0]), 0, z)
            return self.aspect2d.attachNewNode(ls.create())

        self._graph_anode_l = action_strip(self._hist_l, self.STRIP_ZL)
        self._graph_anode_r = action_strip(self._hist_r, self.STRIP_ZR)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Edge-RL 3D ripening chamber demo')
    parser.add_argument('--size', default='1280x720',
                        help='window size WxH (e.g. 1600x900) — match the projector')
    a = parser.parse_args()
    try:
        w, h = (int(v) for v in a.size.lower().split('x'))
    except ValueError:
        parser.error('--size must be WxH, e.g. 1280x720')

    demo = EdgeRLDemo(win_size=(w, h))
    demo.run()
