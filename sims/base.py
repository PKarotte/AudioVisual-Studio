"""
Simulation plugin interface.

To add a new algorithm:
    1. create a module in this folder (e.g. `physarum.py`)
    2. subclass `Simulation`, set `name`, `description`, `params`
    3. implement setup / reset / step / output
    4. decorate the class with `@register`

The GUI picks it up automatically: sliders are generated from `params`, every
modulatable param becomes a column in the modulation matrix, and spawners are
passed in every frame as `SpawnerFrame`s (position, direction, force, dye,
colour) which each simulation interprets in its own way.
"""
from __future__ import annotations

from dataclasses import dataclass

import moderngl

from ..gl_common import GL
from ..params import Param

REGISTRY: dict[str, type["Simulation"]] = {}


def register(cls):
    REGISTRY[cls.name] = cls
    return cls


@dataclass
class SpawnerFrame:
    uid: int
    x: float          # 0..1, GL convention (y up)
    y: float
    radius: float     # fraction of height
    dx: float         # unit direction
    dy: float
    force: float      # 0..100 (UI units)
    dye: float        # 0..4
    color: tuple      # linear-ish rgb 0..1
    angle: float = 0.0
    dye_life: float = 0.0


class Simulation:
    name = "Base"
    description = ""
    params: list[Param] = []

    def __init__(self, gl: GL):
        self.gl = gl
        self.ctx: moderngl.Context = gl.ctx
        self.width = self.height = 0

    def reinit_signature(self, values: dict, out_w: int, out_h: int, render_mode: bool):
        """Anything returned here that changes triggers a new setup()."""
        return tuple(values.get(p.key) for p in self.params if p.reinit) + (out_w, out_h, render_mode)

    def setup(self, out_w: int, out_h: int, values: dict, render_mode: bool):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError

    def step(self, dt: float, t: float, values: dict, spawners: list[SpawnerFrame]):
        raise NotImplementedError

    def output(self) -> moderngl.Texture:
        """HDR-ish RGB texture; the post-processing chain tonemaps it."""
        raise NotImplementedError

    def release(self):
        pass

