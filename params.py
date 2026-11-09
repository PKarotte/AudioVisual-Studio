"""
Parameter definitions.

Every tweakable value in the app (simulation, post FX, spawner) is described by
a `Param`.  The GUI builds sliders from these definitions automatically and the
modulation matrix uses them as targets, so a new simulation only has to declare
its parameter list to get full UI + audio-modulation support.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    key: str
    label: str
    min: float
    max: float
    default: float
    step: float = 0.0            # 0 => continuous
    group: str = "General"
    integer: bool = False
    modulatable: bool = True     # shows up as a column in the mod matrix
    reinit: bool = False         # changing it requires re-allocating the simulation
    wrap: bool = False           # value wraps around (e.g. hue) instead of clamping
    hidden: bool = False         # not shown as a slider (edited by a custom widget)
    tooltip: str = ""

    @property
    def span(self) -> float:
        return self.max - self.min

    def clamp(self, v: float) -> float:
        if self.wrap:
            v = self.min + (v - self.min) % self.span
        else:
            v = min(self.max, max(self.min, v))
        if self.integer:
            v = int(round(v))
        return v


def defaults(params: list[Param]) -> dict[str, float]:
    return {p.key: p.default for p in params}


def by_key(params: list[Param]) -> dict[str, Param]:
    return {p.key: p for p in params}

