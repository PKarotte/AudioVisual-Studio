"""
Modulation engine: evaluates the routing matrix every frame.

    effective = clamp(base + sum(amount * span * envelope(shape(feature))))

Sources:
    "stem:<i>"  feature of a specific stem
    "own"       feature of the stem assigned to the spawner (per-spawner targets)
    "mix"       feature of the summed mix
    "global"    deterministic LFOs / ramps (work without audio)
"""
from __future__ import annotations

import math

import numpy as np

from .audio_analysis import FEATURE_INDEX, FEATURES, FeatureBank
from .params import Param, by_key
from .project import SPAWNER_PARAMS, Project

GLOBAL_FEATURES: list[tuple[str, str]] = [
    ("lfo_slow", "LFO slow (0.05 Hz)"),
    ("lfo_mid", "LFO mid (0.25 Hz)"),
    ("lfo_fast", "LFO fast (1 Hz)"),
    ("wander", "Random wander"),
    ("wander2", "Random wander 2"),
    ("ramp", "Song progress (0..1)"),
]
GLOBAL_INDEX = {k: i for i, (k, _) in enumerate(GLOBAL_FEATURES)}


def global_features(t: float, duration: float) -> np.ndarray:
    w1 = (math.sin(t * 0.31) + math.sin(t * 0.173 + 1.3) + math.sin(t * 0.077 + 4.0)) / 3
    w2 = (math.sin(t * 0.41 + 2.0) + math.sin(t * 0.137 + 0.4) + math.sin(t * 0.059 + 2.5)) / 3
    return np.array([
        0.5 + 0.5 * math.sin(2 * math.pi * 0.05 * t),
        0.5 + 0.5 * math.sin(2 * math.pi * 0.25 * t),
        0.5 + 0.5 * math.sin(2 * math.pi * 1.0 * t),
        0.5 + 0.5 * w1,
        0.5 + 0.5 * w2,
        min(1.0, t / duration) if duration > 0 else 0.0,
    ], dtype=np.float32)


def features_for_source(source: str) -> list[tuple[str, str]]:
    return GLOBAL_FEATURES if source == "global" else FEATURES


class ModEngine:
    def __init__(self, project: Project, bank: FeatureBank | None = None, duration: float = 0.0):
        self.project = project
        self.bank = bank
        self.duration = duration
        self.env: dict = {}
        self.spawner_defs = by_key(SPAWNER_PARAMS)

    def reset(self):
        self.env.clear()

    # ------------------------------------------------------------------
    def _source_vec(self, cache: dict, key, t: float) -> np.ndarray:
        v = cache.get(key)
        if v is None:
            if key == "global":
                v = global_features(t, self.duration)
            elif self.bank is None:
                v = np.zeros(len(FEATURES), np.float32)
            elif key == "mix":
                v = self.bank.sample(-1, t)
            else:  # int stem index
                v = self.bank.sample(int(key), t) if int(key) < len(self.bank.stems) else np.zeros(len(FEATURES), np.float32)
            cache[key] = v
        return v

    def _shape(self, r, x: float) -> float:
        span = max(1e-6, r.hi - r.lo)
        x = min(1.0, max(0.0, (x - r.lo) / span))
        if r.curve != 1.0:
            x = x ** max(0.05, r.curve)
        return 1.0 - x if r.invert else x

    def _envelope(self, key, x: float, r, dt: float) -> float:
        e = self.env.get(key)
        if e is None or dt <= 0:
            self.env[key] = x
            return x
        tau = r.attack if x > e else r.release
        a = 1.0 if tau <= 1e-4 else 1.0 - math.exp(-dt / tau)
        e += (x - e) * a
        self.env[key] = e
        return e

    # ------------------------------------------------------------------
    def evaluate(self, t: float, dt: float, sim_defs: list[Param], post_defs: list[Param]):
        p = self.project
        sim_d, post_d = by_key(sim_defs), by_key(post_defs)
        sim_base = p.sim_values.setdefault(p.sim_name, {})
        sim_vals = {k: sim_base.get(k, d.default) for k, d in sim_d.items()}
        post_vals = {k: p.post_values.get(k, d.default) for k, d in post_d.items()}
        sp_vals = {s.uid: dict(s.params) for s in p.spawners}
        cache: dict = {}

        for ri, r in enumerate(p.routes):
            if not r.enabled or r.amount == 0:
                continue
            parts = r.target.split(".")
            gfeat = r.source == "global"
            fidx = (GLOBAL_INDEX if gfeat else FEATURE_INDEX).get(r.feature)
            if fidx is None:
                continue

            def src_key(spawner=None):
                if r.source == "own":
                    return spawner.stem if spawner is not None else "mix"
                if r.source.startswith("stem:"):
                    return int(r.source[5:])
                return r.source

            if parts[0] in ("sim", "post"):
                defs, vals = (sim_d, sim_vals) if parts[0] == "sim" else (post_d, post_vals)
                d = defs.get(parts[1])
                if d is None or not d.modulatable:
                    continue
                x = self._source_vec(cache, src_key(), t)[fidx]
                e = self._envelope((ri, None), self._shape(r, float(x)), r, dt)
                vals[parts[1]] += r.amount * d.span * e
            elif parts[0] == "spawner" and len(parts) == 3:
                d = self.spawner_defs.get(parts[2])
                if d is None:
                    continue
                targets = p.spawners if parts[1] == "*" else [s for s in p.spawners if str(s.uid) == parts[1]]
                for s in targets:
                    x = self._source_vec(cache, src_key(s), t)[fidx]
                    e = self._envelope((ri, s.uid), self._shape(r, float(x)), r, dt)
                    sp_vals[s.uid][parts[2]] += r.amount * d.span * e

        for k, d in sim_d.items():
            sim_vals[k] = d.clamp(sim_vals[k])
        for k, d in post_d.items():
            post_vals[k] = d.clamp(post_vals[k])
        for vals in sp_vals.values():
            for k, d in self.spawner_defs.items():
                vals[k] = d.clamp(vals.get(k, d.default))
        return sim_vals, post_vals, sp_vals, cache

