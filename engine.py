"""
Engine = modulation + spawner kinematics + simulation + post FX for one GL context.

The same class drives the live viewport and the offline renderer, so what you
see live is what gets rendered.
"""
from __future__ import annotations

import colorsys
import math

import moderngl

from .audio_analysis import FeatureBank
from .gl_common import GL
from .modulation import ModEngine
from .postfx import POST_PARAMS, PostFX
from .project import Project
from .sims import REGISTRY, SpawnerFrame


class Engine:
    def __init__(self, ctx: moderngl.Context, project: Project, bank: FeatureBank | None = None,
                 duration: float = 0.0, render_mode: bool = False):
        self.ctx = ctx
        self.gl = GL(ctx)
        self.project = project
        self.render_mode = render_mode
        self.mods = ModEngine(project, bank, duration)
        self.post = PostFX(self.gl)
        self.sim = None
        self.sim_name = None
        self._sig = None
        self._phase: dict = {}
        self.out_size = (1280, 720)
        self.last_state = None
        self._reset_pending = False

    # ------------------------------------------------------------------
    def set_bank(self, bank: FeatureBank | None, duration: float):
        self.mods.bank = bank
        self.mods.duration = duration
        self.mods.reset()

    def set_output_size(self, w: int, h: int):
        self.out_size = (max(16, int(w) // 2 * 2), max(16, int(h) // 2 * 2))

    def request_reset(self):
        self._reset_pending = True

    def sim_params(self):
        cls = REGISTRY.get(self.project.sim_name)
        return cls.params if cls else []

    def _ensure_sim(self):
        name = self.project.sim_name
        if name not in REGISTRY:
            name = self.project.sim_name = next(iter(REGISTRY))
        if self.sim_name != name:
            if self.sim:
                self.sim.release()
            self.sim = REGISTRY[name](self.gl)
            self.sim_name = name
            self._sig = None
        base = self.project.sim_values.setdefault(name, {})
        vals = {p.key: base.get(p.key, p.default) for p in self.sim.params}
        w, h = self.out_size
        sig = self.sim.reinit_signature(vals, w, h, self.render_mode)
        if sig != self._sig:
            self.sim.setup(w, h, vals, self.render_mode)
            self.sim.reset()
            self.post.ensure(w, h)
            self.post.clear()
            self._sig = sig
        if self.project.bg_image_path != self.post.bg_path:
            self.post.set_background(self.project.bg_image_path)

    # ------------------------------------------------------------------
    def _spawner_frames(self, t: float, dt: float, sp_vals: dict) -> list[SpawnerFrame]:
        frames = []
        w, h = self.out_size
        aspect = w / h
        for s in self.project.spawners:
            if not s.enabled:
                continue
            v = sp_vals[s.uid]
            ph = self._phase.setdefault(s.uid, [0.0, 0.0])
            ph[0] += v["spin"] * dt
            ph[1] += v["orbit_speed"] * dt * 2 * math.pi
            ang = v["angle"] + ph[0] + v["sway_amp"] * math.sin(2 * math.pi * v["sway_freq"] * t + s.uid * 1.7)
            x = s.x + v["offset_x"] + v["orbit_r"] * math.cos(ph[1]) / aspect
            # Invert Y so canvas top (0.0) matches simulation top (1.0 in GL coords)
            y = (1.0 - s.y) - v["offset_y"] - v["orbit_r"] * math.sin(ph[1])
            if v["jitter"] > 0:
                k = s.uid * 13.1
                x += v["jitter"] * (math.sin(t * 1.3 + k) + math.sin(t * 2.9 + k * 0.7)) * 0.5 / aspect
                y += v["jitter"] * (math.sin(t * 1.7 + k * 1.3) + math.sin(t * 3.1 + k * 0.3)) * 0.5
            hue = ((v["hue"] + v["hue_shift"]) % 360) / 360.0
            rgb = colorsys.hsv_to_rgb(hue, v["sat"], v["val"])
            a = math.radians(ang)
            # dx = sin(ang), dy = -cos(ang) in canvas (y-down) -> in simulation (y-up): dx = sin(ang), dy = -(-cos(ang)) = cos(ang)
            frames.append(SpawnerFrame(uid=s.uid, x=x, y=y, radius=v["radius"],
                                       dx=math.sin(a), dy=math.cos(a), force=v["force"], dye=v["dye"],
                                       color=rgb, angle=ang, dye_life=v.get("dye_life", 0.0)))
        return frames

    def frame(self, t: float, dt: float):
        """Advance one frame and render into the post frame buffer. Returns the frame Target."""
        self._ensure_sim()
        if self._reset_pending:
            self.sim.reset()
            self.post.clear()
            self.mods.reset()
            self._phase.clear()
            self._reset_pending = False
        sim_vals, post_vals, sp_vals, feats = self.mods.evaluate(t, dt, self.sim.params, POST_PARAMS)
        frames = self._spawner_frames(t, dt, sp_vals)
        self.sim.step(dt, t, sim_vals, frames)
        out = self.post.process(self.sim.output(), post_vals, dt if dt > 0 else 1 / 60)
        self.last_state = dict(t=t, sim=sim_vals, post=post_vals, spawners=sp_vals, frames=frames, features=feats)
        return out

    def read_rgb(self) -> bytes:
        """Last frame as packed RGB bytes (bottom-up rows)."""
        return self.post.frame.read.fbo.read(components=3, alignment=1)

    def release(self):
        if self.sim:
            self.sim.release()
        self.post.release()

