"""Reusable widgets: auto-generated parameter sliders and an HSV colour wheel."""
from __future__ import annotations

import math

import numpy as np
from PySide6.QtCore import QPointF, Qt, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPen
from PySide6.QtWidgets import (QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QSlider, QVBoxLayout, QWidget)

from ..audio_analysis import FEATURES
from ..modulation import GLOBAL_FEATURES
from ..params import Param
from ..project import Project, Route, SPAWNER_PARAMS

SLIDER_STEPS = 1000


class ParamSlider(QWidget):
    valueChanged = Signal(str, float)
    routeChanged = Signal()

    def __init__(self, p: Param, parent=None, project: Project | None = None,
                 target_type: str = "sim", spawner_uid: int | None = None):
        super().__init__(parent)
        self.p = p
        self.project = project
        self.target_type = target_type
        self.spawner_uid = spawner_uid
        self._updating_mod = False

        main_lay = QVBoxLayout(self)
        main_lay.setContentsMargins(0, 2, 0, 4)
        main_lay.setSpacing(2)

        # Row 1: Parameter Slider & Spinner
        row1 = QHBoxLayout()
        row1.setContentsMargins(0, 0, 0, 0)
        row1.setSpacing(6)

        self.label = QLabel(p.label)
        self.label.setMinimumWidth(125)
        self.label.setToolTip((p.tooltip + "\n" if p.tooltip else "") + "Double-click to reset")
        self.label.mouseDoubleClickEvent = lambda e: self._set_from_user(p.default)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, SLIDER_STEPS)

        self.spin = QDoubleSpinBox()
        dec = 0 if p.integer else (3 if p.span <= 2 else (2 if p.span <= 20 else 1))
        if p.span <= 0.2:
            dec = 4
        self.spin.setDecimals(dec)
        self.spin.setRange(p.min, p.max)
        self.spin.setSingleStep(p.step or p.span / 100)
        self.spin.setFixedWidth(78)

        self.live = QLabel("")
        self.live.setFixedWidth(58)
        self.live.setStyleSheet("color:#f0a040; font-size:10px;")

        row1.addWidget(self.label)
        row1.addWidget(self.slider, 1)
        row1.addWidget(self.spin)
        row1.addWidget(self.live)
        main_lay.addLayout(row1)

        self.slider.valueChanged.connect(self._slider_moved)
        self.spin.valueChanged.connect(self._spin_changed)
        self.set_value(p.default)

        # Row 2: Inline modulation row (if modulatable)
        if p.modulatable and self.project is not None:
            self.mod_row = QWidget()
            mod_lay = QHBoxLayout(self.mod_row)
            mod_lay.setContentsMargins(12, 0, 0, 0)
            mod_lay.setSpacing(5)

            self.lbl_mod = QLabel("⚡ Mod:")
            self.lbl_mod.setStyleSheet("color: #778899; font-size: 11px; font-weight: bold;")
            self.lbl_mod.setToolTip("Audio or dynamic modulation for this parameter")

            self.combo_source = QComboBox()
            self.combo_source.setMinimumWidth(140)
            self.combo_source.setSizeAdjustPolicy(QComboBox.AdjustToContents)
            self.combo_source.setStyleSheet("font-size: 11px;")
            self._populate_sources()

            self.lbl_strength = QLabel("Strength:")
            self.lbl_strength.setStyleSheet("color: #778899; font-size: 11px;")

            self.spin_strength = QDoubleSpinBox()
            self.spin_strength.setRange(-1.0, 1.0)
            self.spin_strength.setSingleStep(0.05)
            self.spin_strength.setValue(0.50)
            self.spin_strength.setFixedWidth(64)
            self.spin_strength.setStyleSheet("font-size: 11px;")
            self.spin_strength.setToolTip("Modulation strength (-1.0 to +1.0). Negative inverts effect.")

            mod_lay.addWidget(self.lbl_mod)
            mod_lay.addWidget(self.combo_source, 1)
            mod_lay.addWidget(self.lbl_strength)
            mod_lay.addWidget(self.spin_strength)

            if self.target_type == "spawner":
                self.combo_scope = QComboBox()
                self.combo_scope.addItems(["This Spawner", "All Spawners"])
                self.combo_scope.setFixedWidth(98)
                self.combo_scope.setStyleSheet("font-size: 11px;")
                self.combo_scope.setToolTip("Scope: modulate only this spawner or all spawners")
                self.combo_scope.currentTextChanged.connect(self._on_scope_changed)
                mod_lay.addWidget(self.combo_scope)

            self.btn_detail = QPushButton("⚙")
            self.btn_detail.setFixedSize(24, 22)
            self.btn_detail.setStyleSheet("font-size: 11px; padding: 0px;")
            self.btn_detail.setToolTip("Envelope & Curve settings (Attack, Release, Curve, Invert)")
            self.btn_detail.clicked.connect(self._on_detail_clicked)
            mod_lay.addWidget(self.btn_detail)

            main_lay.addWidget(self.mod_row)

            self.combo_source.currentIndexChanged.connect(self._on_source_changed)
            self.spin_strength.valueChanged.connect(self._on_strength_changed)

            self.refresh_modulation()

    def _populate_sources(self):
        self.combo_source.blockSignals(True)
        self.combo_source.clear()
        self.combo_source.addItem("None (No Mod)", "")

        # 1. Spawner stem (if target is a spawner)
        if self.target_type == "spawner":
            for f_key, f_label in FEATURES:
                self.combo_source.addItem(f"[Spawner Stem] {f_label}", f"own::{f_key}")

        # 2. Master Mix
        for f_key, f_label in FEATURES:
            self.combo_source.addItem(f"[Mix Audio] {f_label}", f"mix::{f_key}")

        # 3. Global sources
        for f_key, f_label in GLOBAL_FEATURES:
            self.combo_source.addItem(f"[Global] {f_label}", f"global::{f_key}")

        # 4. Individual stems
        if self.project and self.project.stems:
            for idx, stem in enumerate(self.project.stems):
                s_name = stem.label()
                if len(s_name) > 22:
                    s_name = s_name[:10] + "…" + s_name[-10:]
                for f_key, f_label in FEATURES:
                    self.combo_source.addItem(f"[{s_name}] {f_label}", f"stem:{idx}::{f_key}")

        self.combo_source.blockSignals(False)

    def find_existing_route(self) -> tuple[Route | None, str]:
        if not self.project:
            return None, ""
        if self.target_type == "sim":
            tgt = f"sim.{self.p.key}"
            for r in self.project.routes:
                if r.target == tgt:
                    return r, ""
        elif self.target_type == "post":
            tgt = f"post.{self.p.key}"
            for r in self.project.routes:
                if r.target == tgt:
                    return r, ""
        elif self.target_type == "spawner":
            tgt_this = f"spawner.{self.spawner_uid}.{self.p.key}"
            for r in self.project.routes:
                if r.target == tgt_this:
                    return r, "This Spawner"
            tgt_all = f"spawner.*.{self.p.key}"
            for r in self.project.routes:
                if r.target == tgt_all:
                    return r, "All Spawners"
        return None, ""

    def get_target_key(self) -> str:
        if self.target_type == "sim":
            return f"sim.{self.p.key}"
        elif self.target_type == "post":
            return f"post.{self.p.key}"
        elif self.target_type == "spawner":
            scope = self.combo_scope.currentText() if hasattr(self, "combo_scope") else "This Spawner"
            if scope == "All Spawners":
                return f"spawner.*.{self.p.key}"
            else:
                return f"spawner.{self.spawner_uid}.{self.p.key}"
        return ""

    def refresh_modulation(self):
        if not hasattr(self, "combo_source"):
            return
        self._updating_mod = True
        route, scope = self.find_existing_route()
        if route is not None:
            if hasattr(self, "combo_scope") and scope:
                self.combo_scope.blockSignals(True)
                self.combo_scope.setCurrentText(scope)
                self.combo_scope.blockSignals(False)
            data_key = f"{route.source}::{route.feature}"
            idx = self.combo_source.findData(data_key)
            self.combo_source.blockSignals(True)
            if idx >= 0:
                self.combo_source.setCurrentIndex(idx)
            else:
                self.combo_source.setCurrentIndex(0)
            self.combo_source.blockSignals(False)
            self.spin_strength.blockSignals(True)
            self.spin_strength.setValue(route.amount)
            self.spin_strength.blockSignals(False)
            self._set_mod_active(True)
        else:
            self.combo_source.blockSignals(True)
            self.combo_source.setCurrentIndex(0)
            self.combo_source.blockSignals(False)
            self._set_mod_active(False)
        self._updating_mod = False

    def _set_mod_active(self, active: bool):
        self.spin_strength.setEnabled(active)
        self.lbl_strength.setEnabled(active)
        self.btn_detail.setEnabled(active)
        if hasattr(self, "combo_scope"):
            self.combo_scope.setEnabled(active)
        if active:
            self.lbl_mod.setStyleSheet("color: #f0a040; font-size: 11px; font-weight: bold;")
        else:
            self.lbl_mod.setStyleSheet("color: #778899; font-size: 11px; font-weight: bold;")

    def _on_source_changed(self, idx: int):
        if getattr(self, "_updating_mod", False) or not self.project:
            return
        data = self.combo_source.itemData(idx)
        route, _ = self.find_existing_route()
        target = self.get_target_key()
        if not data:
            # User selected None -> remove all routes for this target
            tgt_exact = f"spawner.{self.spawner_uid}.{self.p.key}" if self.target_type == "spawner" else target
            tgt_all = f"spawner.*.{self.p.key}" if self.target_type == "spawner" else target
            self.project.routes = [r for r in self.project.routes if r.target != target and r.target != tgt_exact and r.target != tgt_all]
            self._set_mod_active(False)
            self.routeChanged.emit()
        else:
            src_id, f_key = str(data).split("::", 1)
            if route is not None:
                route.source = src_id
                route.feature = f_key
                route.target = target
                route.amount = self.spin_strength.value()
                route.enabled = True
            else:
                new_r = Route(
                    source=src_id,
                    feature=f_key,
                    target=target,
                    amount=self.spin_strength.value(),
                    attack=0.01,
                    release=0.15,
                    curve=1.0,
                    enabled=True
                )
                self.project.routes.append(new_r)
            self._set_mod_active(True)
            self.routeChanged.emit()

    def _on_strength_changed(self, val: float):
        if getattr(self, "_updating_mod", False) or not self.project:
            return
        route, _ = self.find_existing_route()
        if route is not None:
            route.amount = val
            self.routeChanged.emit()

    def _on_scope_changed(self, text: str):
        if getattr(self, "_updating_mod", False) or not self.project:
            return
        route, _ = self.find_existing_route()
        if route is not None:
            new_target = f"spawner.*.{self.p.key}" if text == "All Spawners" else f"spawner.{self.spawner_uid}.{self.p.key}"
            route.target = new_target
            self.routeChanged.emit()

    def _on_detail_clicked(self):
        if not self.project:
            return
        route, _ = self.find_existing_route()
        if route is not None:
            from .matrix import RouteDetailDialog
            src_lbl = self.combo_source.currentText()
            tgt_lbl = f"{self.p.label} ({route.target})"
            dlg = RouteDetailDialog(route, src_lbl, tgt_lbl, self)
            if dlg.exec():
                dlg.apply_changes()
                self._updating_mod = True
                self.spin_strength.setValue(route.amount)
                self._updating_mod = False
                self.routeChanged.emit()

    def _to_pos(self, v):
        return int(round((v - self.p.min) / self.p.span * SLIDER_STEPS)) if self.p.span else 0

    def _from_pos(self, pos):
        v = self.p.min + pos / SLIDER_STEPS * self.p.span
        if self.p.step:
            v = self.p.min + round((v - self.p.min) / self.p.step) * self.p.step
        return self.p.clamp(v)

    def set_value(self, v):
        v = self.p.clamp(float(v))
        self.slider.blockSignals(True)
        self.spin.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.spin.setValue(v)
        self.slider.blockSignals(False)
        self.spin.blockSignals(False)

    def _set_from_user(self, v):
        self.set_value(v)
        self.valueChanged.emit(self.p.key, float(v))

    def _slider_moved(self, pos):
        v = self._from_pos(pos)
        self.spin.blockSignals(True)
        self.spin.setValue(v)
        self.spin.blockSignals(False)
        self.valueChanged.emit(self.p.key, float(v))

    def _spin_changed(self, v):
        self.slider.blockSignals(True)
        self.slider.setValue(self._to_pos(v))
        self.slider.blockSignals(False)
        self.valueChanged.emit(self.p.key, float(v))

    def set_live(self, v):
        if v is None or abs(v - self.spin.value()) < self.p.span * 0.002:
            self.live.setText("")
        else:
            self.live.setText(f"→{v:.3g}")


class ParamPanel(QWidget):
    """Builds grouped sliders from a list of Params."""
    changed = Signal(str, float)
    routeChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lay = QVBoxLayout(self)
        self.lay.setContentsMargins(2, 2, 2, 2)
        self.sliders: dict[str, ParamSlider] = {}
        self.project: Project | None = None
        self.target_type: str = "sim"
        self.spawner_uid: int | None = None

    def build(self, params: list[Param], values: dict, project: Project | None = None,
              target_type: str = "sim", spawner_uid: int | None = None):
        self.project = project
        self.target_type = target_type
        self.spawner_uid = spawner_uid
        while self.lay.count():
            it = self.lay.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        self.sliders.clear()
        groups: dict[str, QVBoxLayout] = {}
        for p in params:
            if p.hidden:
                continue
            if p.group not in groups:
                box = QGroupBox(p.group)
                gl = QVBoxLayout(box)
                gl.setSpacing(2)
                gl.setContentsMargins(6, 4, 6, 4)
                groups[p.group] = gl
                self.lay.addWidget(box)
            s = ParamSlider(p, project=project, target_type=target_type, spawner_uid=spawner_uid)
            s.set_value(values.get(p.key, p.default))
            s.valueChanged.connect(self.changed)
            s.routeChanged.connect(self.routeChanged)
            groups[p.group].addWidget(s)
            self.sliders[p.key] = s
        self.lay.addStretch(1)

    def refresh_modulations(self):
        for s in self.sliders.values():
            s.refresh_modulation()

    def set_values(self, values: dict):
        for k, s in self.sliders.items():
            if k in values:
                s.set_value(values[k])

    def set_live(self, values: dict | None):
        for k, s in self.sliders.items():
            s.set_live(None if values is None else values.get(k))


# ---------------------------------------------------------------------------
def hsv_to_rgb_np(h, s, v):
    h = (h % 1.0) * 6.0
    i = np.floor(h).astype(int) % 6
    f = h - np.floor(h)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    return r, g, b


class ColorWheel(QWidget):
    """Hue = angle, saturation = radius; brightness via the slider. Emits (h_deg, s, v)."""
    colorChanged = Signal(float, float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.h, self.s, self.v = 0.0, 1.0, 1.0
        self.wheel = _Wheel(self)
        self.vslider = QSlider(Qt.Vertical)
        self.vslider.setRange(0, 1000)
        self.vslider.setToolTip("Brightness")
        self.hex = QLineEdit()
        self.hex.setFixedWidth(80)
        self.hex.setToolTip("Hex colour")
        self.swatch = QLabel()
        self.swatch.setFixedSize(80, 36)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.wheel)
        lay.addWidget(self.vslider)
        side = QVBoxLayout()
        side.addWidget(QLabel("Colour"))
        side.addWidget(self.swatch)
        side.addWidget(self.hex)
        side.addStretch(1)
        lay.addLayout(side)
        self.vslider.valueChanged.connect(self._v_changed)
        self.hex.editingFinished.connect(self._hex_changed)
        self._sync()

    def set_hsv(self, h, s, v):
        self.h, self.s, self.v = h % 360, s, v
        self._sync()

    def _sync(self):
        self.vslider.blockSignals(True)
        self.vslider.setValue(int(self.v * 1000))
        self.vslider.blockSignals(False)
        c = QColor.fromHsvF((self.h % 360) / 360.0, self.s, self.v)
        self.hex.setText(c.name())
        self.swatch.setStyleSheet(f"background:{c.name()}; border:1px solid #888; border-radius:4px;")
        self.wheel.update()

    def _emit(self):
        self._sync()
        self.colorChanged.emit(self.h, self.s, self.v)

    def _v_changed(self, pos):
        self.v = pos / 1000.0
        self._emit()

    def _hex_changed(self):
        c = QColor(self.hex.text())
        if c.isValid():
            h, s, v, _ = c.getHsvF()
            self.h, self.s, self.v = (h if h >= 0 else 0) * 360, s, v
            self._emit()


class _Wheel(QWidget):
    def __init__(self, owner: ColorWheel):
        super().__init__(owner)
        self.o = owner
        self.setFixedSize(150, 150)
        self._img = None
        self._img_v = None

    def _image(self):
        n = self.width()
        if self._img is not None and self._img_v == round(self.o.v, 3):
            return self._img
        y, x = np.mgrid[0:n, 0:n].astype(np.float32)
        c = (n - 1) / 2
        dx, dy = (x - c) / c, (c - y) / c
        r = np.sqrt(dx * dx + dy * dy)
        hue = (np.arctan2(dy, dx) / (2 * np.pi)) % 1.0
        v = max(self.o.v, 0.15)
        R, G, B = hsv_to_rgb_np(hue, np.clip(r, 0, 1), np.full_like(r, v))
        a = np.clip((1.0 - r) * c + 0.5, 0, 1)
        rgba = np.dstack([B, G, R, a]) * 255  # QImage ARGB32 is BGRA in memory
        rgba[..., :3] *= a[..., None]
        buf = np.ascontiguousarray(rgba.astype(np.uint8))
        self._img = QImage(buf.data, n, n, 4 * n, QImage.Format_ARGB32_Premultiplied).copy()
        self._img_v = round(self.o.v, 3)
        return self._img

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.drawImage(0, 0, self._image())
        c = (self.width() - 1) / 2
        a = math.radians(self.o.h)
        pt = QPointF(c + math.cos(a) * self.o.s * c, c - math.sin(a) * self.o.s * c)
        p.setPen(QPen(Qt.black, 3))
        p.drawEllipse(pt, 6, 6)
        p.setPen(QPen(Qt.white, 1.5))
        p.drawEllipse(pt, 6, 6)

    def _pick(self, pos):
        c = (self.width() - 1) / 2
        dx, dy = (pos.x() - c) / c, (c - pos.y()) / c
        self.o.h = math.degrees(math.atan2(dy, dx)) % 360
        self.o.s = min(1.0, math.hypot(dx, dy))
        self.o._emit()

    def mousePressEvent(self, e):
        self._pick(e.position())

    def mouseMoveEvent(self, e):
        if e.buttons() & Qt.LeftButton:
            self._pick(e.position())

