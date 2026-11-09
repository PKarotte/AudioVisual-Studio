"""Project state: spawners, parameter values, modulation routes, stems, render settings."""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field

from .params import Param, defaults

# ----------------------------------------------------------------------------
# Spawner parameters (common to all simulations)
# ----------------------------------------------------------------------------
SPAWNER_PARAMS: list[Param] = [
    Param("force", "Force", 0, 100, 0.0, group="Motion", tooltip="Push strength of the jet"),
    Param("radius", "Radius", 0.005, 0.3, 0.035, group="Motion", tooltip="Jet radius (fraction of height)"),
    Param("angle", "Angle (deg)", -180, 180, 0, wrap=True, group="Motion", tooltip="0 = up, clockwise"),
    Param("sway_amp", "Sway amount (deg)", 0, 120, 0, group="Motion"),
    Param("sway_freq", "Sway speed (Hz)", 0, 4, 0.25, group="Motion"),
    Param("spin", "Spin (deg/s)", -720, 720, 0, group="Motion"),
    Param("orbit_r", "Orbit radius", 0, 0.4, 0, group="Position"),
    Param("orbit_speed", "Orbit speed (Hz)", -2, 2, 0.1, group="Position"),
    Param("jitter", "Position jitter", 0, 0.15, 0, group="Position"),
    Param("offset_x", "Offset X", -0.5, 0.5, 0, group="Position"),
    Param("offset_y", "Offset Y", -0.5, 0.5, 0, group="Position"),
    Param("dye", "Dye amount", 0, 4, 0.0, group="Color"),
    Param("dye_life", "Dye lifetime (s)", 0, 30, 0.0, step=0.1, group="Color",
          tooltip="Dye lifetime for this spawner in seconds (0 = use simulation default)"),
    Param("hue", "Hue", 0, 360, 190, wrap=True, hidden=True, group="Color"),
    Param("sat", "Saturation", 0, 1, 1, hidden=True, group="Color"),
    Param("val", "Brightness", 0, 1, 1, hidden=True, group="Color"),
    Param("hue_shift", "Hue shift", -180, 180, 0, group="Color"),
]

DEFAULT_COLORS = [(190, 1.0, 1.0), (0, 1.0, 0.9), (50, 0.9, 1.0), (125, 1.0, 0.8),
                  (25, 1.0, 1.0), (220, 0.9, 1.0), (315, 0.9, 1.0), (270, 1.0, 0.9)]


@dataclass
class Spawner:
    uid: int
    x: float = 0.5          # 0..1 left -> right
    y: float = 0.5          # 0..1 top -> bottom
    stem: int = 0           # index into project stems ("own stem" source)
    enabled: bool = True
    name: str = ""
    params: dict = field(default_factory=lambda: defaults(SPAWNER_PARAMS))

    def label(self) -> str:
        return self.name or f"Spawner {self.uid}"


@dataclass
class Route:
    source: str            # "own" | "mix" | "global" | "stem:<i>"
    feature: str
    target: str            # "sim.<k>" | "post.<k>" | "spawner.*.<k>" | "spawner.<uid>.<k>"
    amount: float = 0.5    # -1..1, fraction of the parameter range
    attack: float = 0.01   # seconds
    release: float = 0.15  # seconds
    curve: float = 1.0     # exponent applied to the normalised feature
    lo: float = 0.0        # input range (threshold)
    hi: float = 1.0
    invert: bool = False
    enabled: bool = True


@dataclass
class StemSpec:
    path: str
    kind: str = "file"     # "file" (played + analysed) | "band" (analysis only)
    lo: float = 0.0
    hi: float = 0.0
    volume: float = 1.0
    mute: bool = False
    name: str = ""

    def label(self) -> str:
        import os
        base = self.name or os.path.splitext(os.path.basename(self.path))[0]
        if self.kind == "band":
            return f"{base} [{self.lo:.0f}-{self.hi:.0f} Hz]"
        return base


RENDER_DEFAULTS = dict(width=1920, height=1080, fps=60, codec="H.264 (x264)", quality=18,
                       start=0.0, end=0.0, preroll=2.5, path="")


class Project:
    def __init__(self):
        self.sim_name = "Fluid (Stable Fluids)"
        self.sim_values: dict[str, dict] = {}     # sim name -> {key: value}
        self.post_values: dict = {}
        self.spawners: list[Spawner] = []
        self.routes: list[Route] = []
        self.stems: list[StemSpec] = []
        self.grid = dict(cols=16, rows=9, snap=True)
        self.aspect = (16, 9)
        self.master_volume = 0.8
        self.bg_image_path: str = ""
        self.render = dict(RENDER_DEFAULTS)
        self._next_uid = 1

    # ---------------- spawners ----------------
    def new_spawner(self, x=0.5, y=0.5, stem=None, with_default_routes=True) -> Spawner:
        uid = self._next_uid
        self._next_uid += 1
        h, s, v = DEFAULT_COLORS[(uid - 1) % len(DEFAULT_COLORS)]
        sp = Spawner(uid=uid, x=x, y=y)
        sp.params.update(hue=h, sat=s, val=v, force=20.0, dye=0.8)
        sp.stem = stem if stem is not None else max(0, (uid - 1) % max(1, len(self.stems)))
        self.spawners.append(sp)
        if with_default_routes and not any(r.target.startswith("spawner.*.") for r in self.routes):
            self.add_default_routes()
        return sp

    def add_default_routes(self):
        """Sensible starting point: every spawner reacts dynamically to its assigned stem."""
        self.routes += [
            Route("own", "rms", "spawner.*.force", 0.45, 0.005, 0.12, 1.2),
            Route("own", "rms", "spawner.*.dye", 0.60, 0.005, 0.15, 1.2),
            Route("own", "onset", "spawner.*.radius", 0.15, 0.002, 0.15, 1.5),
        ]

    def spawner_by_uid(self, uid) -> Spawner | None:
        for s in self.spawners:
            if s.uid == uid:
                return s
        return None

    def remove_spawner(self, uid):
        self.spawners = [s for s in self.spawners if s.uid != uid]
        self.routes = [r for r in self.routes if not r.target.startswith(f"spawner.{uid}.")]

    def duplicate_spawner(self, uid) -> Spawner | None:
        src = self.spawner_by_uid(uid)
        if not src:
            return None
        sp = copy.deepcopy(src)
        sp.uid = self._next_uid
        self._next_uid += 1
        sp.x = min(0.98, sp.x + 0.05)
        sp.name = ""
        self.spawners.append(sp)
        for r in list(self.routes):
            if r.target.startswith(f"spawner.{uid}."):
                nr = copy.deepcopy(r)
                nr.target = r.target.replace(f"spawner.{uid}.", f"spawner.{sp.uid}.", 1)
                self.routes.append(nr)
        return sp

    # ---------------- stems ----------------
    def remove_stem(self, idx):
        del self.stems[idx]
        for sp in self.spawners:
            if sp.stem == idx:
                sp.stem = 0
            elif sp.stem > idx:
                sp.stem -= 1
        keep = []
        for r in self.routes:
            if r.source.startswith("stem:"):
                i = int(r.source[5:])
                if i == idx:
                    continue
                if i > idx:
                    r.source = f"stem:{i - 1}"
            keep.append(r)
        self.routes = keep

    # ---------------- routes ----------------
    def find_route(self, source, feature, target) -> Route | None:
        for r in self.routes:
            if r.source == source and r.feature == feature and r.target == target:
                return r
        return None

    # ---------------- serialisation ----------------
    def to_dict(self) -> dict:
        return dict(
            version=1, sim_name=self.sim_name, sim_values=self.sim_values,
            post_values=self.post_values, spawners=[asdict(s) for s in self.spawners],
            routes=[asdict(r) for r in self.routes], stems=[asdict(s) for s in self.stems],
            grid=self.grid, aspect=list(self.aspect), master_volume=self.master_volume,
            bg_image_path=self.bg_image_path,
            render=self.render, next_uid=self._next_uid)

    @classmethod
    def from_dict(cls, d: dict) -> "Project":
        p = cls()
        p.sim_name = d.get("sim_name", p.sim_name)
        p.sim_values = copy.deepcopy(d.get("sim_values", {}))
        p.post_values = copy.deepcopy(d.get("post_values", {}))
        p.bg_image_path = d.get("bg_image_path", "")
        for sd in d.get("spawners", []):
            params = defaults(SPAWNER_PARAMS)
            params.update(sd.get("params", {}))
            sd = dict(sd, params=params)
            p.spawners.append(Spawner(**sd))
        p.routes = [Route(**r) for r in d.get("routes", [])]
        p.stems = [StemSpec(**s) for s in d.get("stems", [])]
        p.grid = dict(p.grid, **d.get("grid", {}))
        p.aspect = tuple(d.get("aspect", p.aspect))
        p.master_volume = d.get("master_volume", p.master_volume)
        p.render = dict(RENDER_DEFAULTS, **d.get("render", {}))
        p._next_uid = d.get("next_uid", max([s.uid for s in p.spawners], default=0) + 1)
        return p

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path) -> "Project":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))

