"""
Interactive Modulation Matrix Widget.

Rows: Audio & Global Modulation Sources (e.g. Stem 1 RMS, Beat Pulse, Pitch F0, etc.)
Columns: Target Parameters (Simulation params, Post FX, Spawner params)
Cells: Interactive sliders/spinners controlling routing amount, attack, release, and curves.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
    QHeaderView, QDoubleSpinBox, QCheckBox, QPushButton, QLabel, QComboBox,
    QDialog, QFormLayout, QGroupBox, QTabWidget
)

from ..project import Project, Route, SPAWNER_PARAMS
from ..audio_analysis import FEATURES
from ..modulation import GLOBAL_FEATURES
from ..params import Param
from ..postfx import POST_PARAMS


class RouteDetailDialog(QDialog):
    """Detailed popup to tweak envelope attack, release, response curve, and threshold."""
    def __init__(self, route: Route, source_label: str, target_label: str, parent=None):
        super().__init__(parent)
        self.route = route
        self.setWindowTitle(f"Route: {source_label} ➔ {target_label}")
        self.resize(360, 280)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_amount = QDoubleSpinBox()
        self.spin_amount.setRange(-1.0, 1.0)
        self.spin_amount.setSingleStep(0.05)
        self.spin_amount.setValue(route.amount)

        self.spin_attack = QDoubleSpinBox()
        self.spin_attack.setRange(0.001, 2.0)
        self.spin_attack.setSingleStep(0.01)
        self.spin_attack.setSuffix(" s")
        self.spin_attack.setValue(route.attack)

        self.spin_release = QDoubleSpinBox()
        self.spin_release.setRange(0.001, 5.0)
        self.spin_release.setSingleStep(0.05)
        self.spin_release.setSuffix(" s")
        self.spin_release.setValue(route.release)

        self.spin_curve = QDoubleSpinBox()
        self.spin_curve.setRange(0.1, 4.0)
        self.spin_curve.setSingleStep(0.1)
        self.spin_curve.setValue(route.curve)

        self.chk_invert = QCheckBox("Invert Input Curve")
        self.chk_invert.setChecked(route.invert)

        self.chk_enabled = QCheckBox("Route Enabled")
        self.chk_enabled.setChecked(route.enabled)

        form.addRow("Modulation Amount:", self.spin_amount)
        form.addRow("Envelope Attack:", self.spin_attack)
        form.addRow("Envelope Release:", self.spin_release)
        form.addRow("Response Curve Exponent:", self.spin_curve)
        form.addRow("", self.chk_invert)
        form.addRow("", self.chk_enabled)

        layout.addLayout(form)

        btn_box = QHBoxLayout()
        btn_ok = QPushButton("OK")
        btn_ok.clicked.connect(self.accept)
        btn_cancel = QPushButton("Cancel")
        btn_cancel.clicked.connect(self.reject)
        btn_box.addStretch(1)
        btn_box.addWidget(btn_cancel)
        btn_box.addWidget(btn_ok)
        layout.addLayout(btn_box)

    def apply_changes(self):
        self.route.amount = self.spin_amount.value()
        self.route.attack = self.spin_attack.value()
        self.route.release = self.spin_release.value()
        self.route.curve = self.spin_curve.value()
        self.route.invert = self.chk_invert.isChecked()
        self.route.enabled = self.chk_enabled.isChecked()


class ModMatrixTable(QTableWidget):
    routeChanged = Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.sources: list[tuple[str, str, str]] = []  # (source_id, feature_key, display_label)
        self.targets: list[tuple[str, str, Param]] = []  # (target_id, display_label, param_def)
        
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setDefaultSectionSize(110)
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.verticalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)

    def build_matrix(self, sim_params: list[Param]):
        self.blockSignals(True)
        self.clear()

        # 1. Collect Sources
        self.sources.clear()
        
        # Global sources
        for f_key, f_label in GLOBAL_FEATURES:
            self.sources.append(("global", f_key, f"[Global] {f_label}"))

        # Own Stem (dynamic for each spawner)
        for f_key, f_label in FEATURES:
            self.sources.append(("own", f_key, f"[Spawner Stem] {f_label}"))

        # Mix source
        for f_key, f_label in FEATURES:
            self.sources.append(("mix", f_key, f"[Mix Audio] {f_label}"))

        # Individual Stems
        for idx, stem in enumerate(self.project.stems):
            s_name = stem.label()
            for f_key, f_label in FEATURES:
                self.sources.append((f"stem:{idx}", f_key, f"[{s_name}] {f_label}"))

        # 2. Collect Targets
        self.targets.clear()

        # Sim Targets
        for p in sim_params:
            if p.modulatable:
                self.targets.append((f"sim.{p.key}", f"Sim: {p.label}", p))

        # Post FX Targets
        for p in POST_PARAMS:
            if p.modulatable:
                self.targets.append((f"post.{p.key}", f"FX: {p.label}", p))

        # Spawner Global (All Spawners)
        for p in SPAWNER_PARAMS:
            if p.modulatable and not p.hidden:
                self.targets.append((f"spawner.*.{p.key}", f"All Spawners: {p.label}", p))

        # Selected / Individual Spawners
        for sp in self.project.spawners:
            for p in SPAWNER_PARAMS:
                if p.modulatable and not p.hidden:
                    self.targets.append((f"spawner.{sp.uid}.{p.key}", f"Sp#{sp.uid}: {p.label}", p))

        self.setRowCount(len(self.sources))
        self.setColumnCount(len(self.targets))

        # Set Headers
        for col_idx, (_, col_label, _) in enumerate(self.targets):
            item = QTableWidgetItem(col_label)
            item.setToolTip(col_label)
            self.setHorizontalHeaderItem(col_idx, item)

        for row_idx, (_, _, row_label) in enumerate(self.sources):
            item = QTableWidgetItem(row_label)
            item.setToolTip(row_label)
            self.setVerticalHeaderItem(row_idx, item)

        # Populate Cells
        for r_idx, (src_id, f_key, src_lbl) in enumerate(self.sources):
            for c_idx, (tgt_id, tgt_lbl, p_def) in enumerate(self.targets):
                route = self.project.find_route(src_id, f_key, tgt_id)
                cell_widget = self._create_cell_widget(route, src_id, f_key, tgt_id, src_lbl, tgt_lbl)
                self.setCellWidget(r_idx, c_idx, cell_widget)

        self.blockSignals(False)

    def _create_cell_widget(self, route: Route | None, src_id: str, f_key: str, tgt_id: str, src_lbl: str, tgt_lbl: str) -> QWidget:
        widget = QWidget()
        lay = QHBoxLayout(widget)
        lay.setContentsMargins(2, 2, 2, 2)
        lay.setSpacing(2)

        spin = QDoubleSpinBox()
        spin.setRange(-1.0, 1.0)
        spin.setSingleStep(0.05)
        spin.setDecimals(2)
        spin.setFixedWidth(54)

        btn_more = QPushButton("⚙")
        btn_more.setFixedWidth(24)
        btn_more.setToolTip("Route details (Attack/Release, Curve, Invert)")

        if route and route.enabled:
            spin.setValue(route.amount)
            if abs(route.amount) > 0.001:
                spin.setStyleSheet("background-color: #2b4436; color: #a6f3a6; font-weight: bold;")
        else:
            spin.setValue(0.0)
            spin.setStyleSheet("color: #777777;")

        def on_val_changed(val: float):
            nonlocal route
            if abs(val) > 0.001:
                if not route:
                    route = Route(source=src_id, feature=f_key, target=tgt_id, amount=val)
                    self.project.routes.append(route)
                else:
                    route.amount = val
                    route.enabled = True
                spin.setStyleSheet("background-color: #2b4436; color: #a6f3a6; font-weight: bold;")
            else:
                if route:
                    route.amount = 0.0
                    route.enabled = False
                spin.setStyleSheet("color: #777777;")
            self.routeChanged.emit()

        def on_open_detail():
            nonlocal route
            if not route:
                route = Route(source=src_id, feature=f_key, target=tgt_id, amount=spin.value() or 0.5)
                self.project.routes.append(route)
            dlg = RouteDetailDialog(route, src_lbl, tgt_lbl, self)
            if dlg.exec():
                dlg.apply_changes()
                spin.blockSignals(True)
                spin.setValue(route.amount if route.enabled else 0.0)
                spin.blockSignals(False)
                on_val_changed(spin.value())

        spin.valueChanged.connect(on_val_changed)
        btn_more.clicked.connect(on_open_detail)

        lay.addWidget(spin)
        lay.addWidget(btn_more)
        return widget


class ModMatrixPanel(QWidget):
    routesModified = Signal()

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # Header bar
        top_bar = QHBoxLayout()
        info_lbl = QLabel("🎛️ <b>Audio & Dynamic Modulation Matrix</b> — Map volume, timbre, pitch, & beats to any parameter.")
        btn_reset_routes = QPushButton("Clear Inactive Routes")
        btn_reset_routes.clicked.connect(self._clean_routes)
        btn_default_routes = QPushButton("Reset to Audio Defaults")
        btn_default_routes.clicked.connect(self._reset_defaults)

        top_bar.addWidget(info_lbl)
        top_bar.addStretch(1)
        top_bar.addWidget(btn_default_routes)
        top_bar.addWidget(btn_reset_routes)
        layout.addLayout(top_bar)

        self.table = ModMatrixTable(project, self)
        self.table.routeChanged.connect(self.routesModified)
        layout.addWidget(self.table)

    def _clean_routes(self):
        self.project.routes = [r for r in self.project.routes if r.enabled and abs(r.amount) > 0.001]
        self.routesModified.emit()

    def _reset_defaults(self):
        self.project.routes.clear()
        self.project.add_default_routes()
        self.routesModified.emit()

    def refresh(self, sim_params: list[Param]):
        self.table.build_matrix(sim_params)

