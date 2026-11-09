"""Simulation plugins. Every module in this package is imported automatically."""
import importlib
import pkgutil

from .base import REGISTRY, Simulation, SpawnerFrame, register  # noqa: F401

for _m in pkgutil.iter_modules(__path__):
    if _m.name != "base":
        importlib.import_module(f"{__name__}.{_m.name}")

