"""
AudioVisual Studio Main Window.
Coordinates Viewport, Draggable Canvas, Simulation Plugins, Sliders,
Color Wheel, Audio stems playback, and Modulation Matrix.
"""
from __future__ import annotations

import os
import shutil
import sys
import time
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QIcon, QKeySequence
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QSplitter,
    QTabWidget, QScrollArea, QFileDialog, QMessageBox, QPushButton,
    QLabel, QSlider, QComboBox, QToolBar, QStatusBar, QListWidget,
    QListWidgetItem, QGroupBox, QSpinBox, QCheckBox
)

from ..project import Project, Spawner, StemSpec, SPAWNER_PARAMS
from ..audio_player import AudioPlayer
from ..audio_analysis import (
    load_audio, cached_analyze, file_signature, FeatureBank,
    bandpass, band_edges
)
from ..sims import REGISTRY
from ..params import by_key
from ..postfx import POST_PARAMS

from .viewport import ViewportWidget
from .canvas import SpawnerCanvas
from .widgets import ParamPanel, ColorWheel
from .matrix import ModMatrixPanel
from .render_dialog import RenderDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AudioVisual Studio — Audio-Reactive GPU Simulation Engine")
        self.resize(1600, 950)

        self.project = Project()
        self.player = AudioPlayer()
        self.bank = FeatureBank(sr=44100)
        self.audio_stems_raw: list = []  # numpy stereo arrays for sounddevice

        self.selected_spawner_uid: int | None = None
        self.last_clock_time = 0.0

        self._init_ui()
        self._init_timer()
        self._load_default_spawners()

    def _init_ui(self):
        # Menu Bar
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        act_new = file_menu.addAction("&New Project")
        act_new.setShortcut(QKeySequence.New)
        act_new.triggered.connect(self._on_new_project)

        act_open = file_menu.addAction("&Open Project...")
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._on_open_project)

        act_save = file_menu.addAction("&Save Project")
        act_save.setShortcut(QKeySequence.Save)
        act_save.triggered.connect(self._on_save_project)

        act_save_as = file_menu.addAction("Save Project &As...")
        act_save_as.triggered.connect(self._on_save_project_as)

        file_menu.addSeparator()
        act_render = file_menu.addAction("🎬 &Render Video Export...")
        act_render.setShortcut("Ctrl+R")
        act_render.triggered.connect(self._on_open_render)

        file_menu.addSeparator()
        act_exit = file_menu.addAction("E&xit")
        act_exit.triggered.connect(self.close)

        sim_menu = menubar.addMenu("&Simulation")
        act_reset_sim = sim_menu.addAction("Reset Simulation State")
        act_reset_sim.setShortcut("Space")
        act_reset_sim.triggered.connect(self._on_reset_sim)

        # Main Central Area Splitter
        main_splitter = QSplitter(Qt.Horizontal)
        self.setCentralWidget(main_splitter)

        # LEFT COLUMN: Visualizer / Canvas / Matrix Tabs
        left_box = QWidget()
        left_layout = QVBoxLayout(left_box)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Mode Tabs: Live Viewport vs Spawner Canvas vs Mod Matrix
        self.center_tabs = QTabWidget()
        
        # 1. Live Viewport
        self.viewport = ViewportWidget(self.project)
        self.viewport.frameRendered.connect(self._on_frame_rendered)
        self.viewport.spawnerSelected.connect(self._on_spawner_selected)
        self.viewport.spawnerMoved.connect(self._on_spawner_moved)
        self.viewport.spawnerAngleChanged.connect(self._on_spawner_angle_changed)
        self.viewport.canvasDoubleClicked.connect(self._on_canvas_double_clicked)
        self.center_tabs.addTab(self.viewport, "🖥️ Live Simulation Viewport")

        # 2. Spawner Canvas
        self.canvas = SpawnerCanvas(self.project)
        self.canvas.spawnerSelected.connect(self._on_spawner_selected)
        self.canvas.spawnerMoved.connect(self._on_spawner_moved)
        self.canvas.spawnerAngleChanged.connect(self._on_spawner_angle_changed)
        self.canvas.canvasDoubleClicked.connect(self._on_canvas_double_clicked)
        self.center_tabs.addTab(self.canvas, "📍 Spawner Placement & Grid")

        # 3. Modulation Matrix
        self.matrix_panel = ModMatrixPanel(self.project)
        self.matrix_panel.routesModified.connect(self._on_routes_modified)
        self.center_tabs.addTab(self.matrix_panel, "🎛️ Modulation Routing Matrix")

        left_layout.addWidget(self.center_tabs, 1)

        # Transport Controls Bar
        transport_box = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setStyleSheet("font-weight: bold; padding: 6px 16px; background-color: #2e623c;")
        self.btn_play.clicked.connect(self._toggle_playback)

        self.btn_stop = QPushButton("⏹ Stop")
        self.btn_stop.clicked.connect(self._stop_playback)

        self.lbl_time = QLabel("00:00.00 / 00:00.00")
        self.lbl_time.setStyleSheet("font-family: monospace; font-size: 13px;")

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.sliderMoved.connect(self._on_seek_moved)

        self.chk_overlay = QCheckBox("👁️ Overlay Spawners")
        self.chk_overlay.setChecked(True)
        self.chk_overlay.setToolTip("Superimpose spawner nodes and handles directly on the simulation")
        self.chk_overlay.setStyleSheet("font-weight: bold; margin-left: 4px; margin-right: 4px;")
        self.chk_overlay.toggled.connect(self.viewport.set_show_overlay)

        self.btn_popout = QPushButton("🗗 Pop-out Preview")
        self.btn_popout.setStyleSheet("font-weight: bold; background-color: #29384d;")
        self.btn_popout.clicked.connect(self._toggle_popout_preview)

        self.btn_render_quick = QPushButton("🎬 Export Video")
        self.btn_render_quick.setStyleSheet("font-weight: bold; background-color: #3b5072;")
        self.btn_render_quick.clicked.connect(self._on_open_render)

        transport_box.addWidget(self.btn_play)
        transport_box.addWidget(self.btn_stop)
        transport_box.addWidget(self.lbl_time)
        transport_box.addWidget(self.seek_slider, 1)
        transport_box.addWidget(self.chk_overlay)
        transport_box.addWidget(self.btn_popout)
        transport_box.addWidget(self.btn_render_quick)
        left_layout.addLayout(transport_box)

        main_splitter.addWidget(left_box)

        # RIGHT COLUMN: Inspector Sidebar (Tabs for Sim Params, Spawners, Stems, Post FX)
        self.right_tabs = QTabWidget()
        self.right_tabs.setMinimumWidth(380)

        # TAB 1: Simulation Setup & Sliders
        tab_sim = QWidget()
        sim_layout = QVBoxLayout(tab_sim)
        
        sim_select_box = QHBoxLayout()
        sim_select_box.addWidget(QLabel("Algorithm:"))
        self.combo_sim = QComboBox()
        self.combo_sim.addItems(list(REGISTRY.keys()))
        self.combo_sim.currentTextChanged.connect(self._on_sim_changed)
        sim_select_box.addWidget(self.combo_sim, 1)
        sim_layout.addLayout(sim_select_box)

        self.lbl_sim_desc = QLabel()
        self.lbl_sim_desc.setWordWrap(True)
        self.lbl_sim_desc.setStyleSheet("color: #8899aa; font-style: italic; margin-bottom: 6px;")
        sim_layout.addWidget(self.lbl_sim_desc)

        scroll_sim = QScrollArea()
        scroll_sim.setWidgetResizable(True)
        self.panel_sim = ParamPanel()
        self.panel_sim.changed.connect(self._on_sim_param_changed)
        self.panel_sim.routeChanged.connect(self._on_routes_modified)
        scroll_sim.setWidget(self.panel_sim)
        sim_layout.addWidget(scroll_sim, 1)
        self.right_tabs.addTab(tab_sim, "Simulation")

        # TAB 2: Spawner Inspector & Color Wheel
        tab_spawners = QWidget()
        sp_layout = QVBoxLayout(tab_spawners)

        sp_actions = QHBoxLayout()
        btn_add_sp = QPushButton("➕ Add Spawner")
        btn_add_sp.clicked.connect(self._add_spawner)
        btn_del_sp = QPushButton("❌ Remove")
        btn_del_sp.clicked.connect(self._remove_spawner)
        btn_dup_sp = QPushButton("📋 Duplicate")
        btn_dup_sp.clicked.connect(self._duplicate_spawner)
        sp_actions.addWidget(btn_add_sp)
        sp_actions.addWidget(btn_del_sp)
        sp_actions.addWidget(btn_dup_sp)
        sp_layout.addLayout(sp_actions)

        # Grid Snapping Controls
        grid_box = QHBoxLayout()
        grid_box.addWidget(QLabel("Grid:"))
        self.spin_grid_cols = QSpinBox()
        self.spin_grid_cols.setRange(2, 64)
        self.spin_grid_cols.setValue(self.project.grid.get("cols", 16))
        self.spin_grid_cols.valueChanged.connect(self._on_grid_changed)
        self.spin_grid_rows = QSpinBox()
        self.spin_grid_rows.setRange(2, 64)
        self.spin_grid_rows.setValue(self.project.grid.get("rows", 9))
        self.spin_grid_rows.valueChanged.connect(self._on_grid_changed)
        grid_box.addWidget(self.spin_grid_cols)
        grid_box.addWidget(QLabel("x"))
        grid_box.addWidget(self.spin_grid_rows)
        sp_layout.addLayout(grid_box)

        # Spawner selector dropdown
        self.combo_spawner = QComboBox()
        self.combo_spawner.currentIndexChanged.connect(self._on_combo_spawner_changed)
        sp_layout.addWidget(self.combo_spawner)

        # Assigned Stem Selector
        stem_row = QHBoxLayout()
        stem_row.addWidget(QLabel("Assigned Audio Stem:"))
        self.combo_spawner_stem = QComboBox()
        self.combo_spawner_stem.currentIndexChanged.connect(self._on_spawner_stem_changed)
        stem_row.addWidget(self.combo_spawner_stem, 1)
        sp_layout.addLayout(stem_row)

        # HSV Wheel
        self.color_wheel = ColorWheel()
        self.color_wheel.colorChanged.connect(self._on_color_changed)
        sp_layout.addWidget(self.color_wheel)

        # Spawner Sliders Scroll
        scroll_sp = QScrollArea()
        scroll_sp.setWidgetResizable(True)
        self.panel_spawner = ParamPanel()
        self.panel_spawner.changed.connect(self._on_spawner_param_changed)
        self.panel_spawner.routeChanged.connect(self._on_routes_modified)
        scroll_sp.setWidget(self.panel_spawner)
        sp_layout.addWidget(scroll_sp, 1)

        self.right_tabs.addTab(tab_spawners, "Spawners")

        # TAB 3: Audio Tracks & Stems
        tab_audio = QWidget()
        audio_layout = QVBoxLayout(tab_audio)
        
        btn_load_audio = QPushButton("📁 Load Audio File(s)... (WAV/MP3/FLAC)")
        btn_load_audio.setStyleSheet("padding: 8px; font-weight: bold;")
        btn_load_audio.clicked.connect(self._on_load_audio_dialog)
        audio_layout.addWidget(btn_load_audio)

        self.list_stems = QListWidget()
        audio_layout.addWidget(self.list_stems, 1)

        btn_split_bands = QPushButton("Split Loaded Track Into 6 Frequency Bands")
        btn_split_bands.clicked.connect(self._on_split_bands)
        audio_layout.addWidget(btn_split_bands)

        self.right_tabs.addTab(tab_audio, "Audio Stems")

        # TAB 4: Post FX / Shading / Bloom / Feedback
        tab_fx = QWidget()
        fx_layout = QVBoxLayout(tab_fx)
        scroll_fx = QScrollArea()
        scroll_fx.setWidgetResizable(True)
        self.panel_fx = ParamPanel()
        self.panel_fx.changed.connect(self._on_fx_param_changed)
        self.panel_fx.routeChanged.connect(self._on_routes_modified)
        scroll_fx.setWidget(self.panel_fx)
        fx_layout.addWidget(scroll_fx)
        self.right_tabs.addTab(tab_fx, "Post FX")

        main_splitter.addWidget(self.right_tabs)
        main_splitter.setStretchFactor(0, 3)
        main_splitter.setStretchFactor(1, 1)

        self._refresh_sim_ui()
        self._refresh_fx_ui()

    def _init_timer(self):
        self.render_timer = QTimer(self)
        self.render_timer.setInterval(16)  # ~60 FPS
        self.render_timer.timeout.connect(self._on_tick)
        self.render_timer.start()

    def _load_default_spawners(self):
        # Default 8 circle spawners if none present
        if not self.project.spawners:
            for i in range(8):
                self.project.new_spawner(x=(i + 1) / 9.0, y=0.5, stem=i)
        self._refresh_spawners_list()
        if self.project.spawners:
            self.selected_spawner_uid = self.project.spawners[0].uid
            self._refresh_spawner_inspector()

    # -------------------------------------------------------------
    # TICK / PLAYBACK LOOP
    # -------------------------------------------------------------
    def _on_tick(self):
        cur_t = self.player.time()
        dt = cur_t - self.last_clock_time if self.last_clock_time > 0 else 1.0 / 60.0
        if dt < 0 or dt > 0.2:
            dt = 1.0 / 60.0
        self.last_clock_time = cur_t

        # Step viewport simulation
        self.viewport.render_frame(cur_t, dt)

        # Update Time UI
        dur = self.player.duration
        cur_m, cur_s = divmod(cur_t, 60)
        dur_m, dur_s = divmod(dur, 60)
        self.lbl_time.setText(f"{int(cur_m):02d}:{cur_s:05.2f} / {int(dur_m):02d}:{dur_s:05.2f}")

        if dur > 0:
            self.seek_slider.blockSignals(True)
            self.seek_slider.setValue(int((cur_t / dur) * 10000))
            self.seek_slider.blockSignals(False)

        if self.player.finished:
            self.player.pause()
            self.btn_play.setText("▶ Play")

    def _on_frame_rendered(self, state: dict):
        # Update canvas pulse rings when visible
        if self.center_tabs.currentIndex() == 1:
            self.canvas.set_live_state(state)

        # Throttle slider live indicators to ~12 FPS instead of 60 FPS to prevent Qt layout/paint freezing
        self._frame_count = getattr(self, "_frame_count", 0) + 1
        if self._frame_count % 5 == 0:
            cur_tab = self.right_tabs.currentIndex()
            if cur_tab == 1 and self.selected_spawner_uid and "spawners" in state:
                sp_live = state["spawners"].get(self.selected_spawner_uid)
                self.panel_spawner.set_live(sp_live)
            elif cur_tab == 0 and "sim" in state:
                self.panel_sim.set_live(state["sim"])
            elif cur_tab == 3 and "post" in state:
                self.panel_fx.set_live(state["post"])

    def _toggle_popout_preview(self):
        """Pop out the live preview into a separate floating window or dock it back."""
        if not hasattr(self, "popout_dialog") or self.popout_dialog is None:
            from PySide6.QtWidgets import QDialog
            self.popout_dialog = QDialog(self)
            self.popout_dialog.setWindowTitle("AudioVisual Studio — Floating Live Preview")
            self.popout_dialog.resize(960, 540)
            pop_layout = QVBoxLayout(self.popout_dialog)
            pop_layout.setContentsMargins(0, 0, 0, 0)
            
            # Detach viewport from center_tabs and put into popout dialog
            self.center_tabs.removeTab(self.center_tabs.indexOf(self.viewport))
            pop_layout.addWidget(self.viewport)

            def on_popout_closed(e):
                # Put viewport back into center_tabs as first tab
                self.viewport.setParent(self)
                self.center_tabs.insertTab(0, self.viewport, "🖥️ Live Simulation Viewport")
                self.center_tabs.setCurrentIndex(0)
                self.btn_popout.setText("🗗 Pop-out Preview")
                self.popout_dialog = None
                e.accept()

            self.popout_dialog.closeEvent = on_popout_closed
            self.popout_dialog.show()
            self.btn_popout.setText("🗗 Dock Preview")
        else:
            self.popout_dialog.close()

    def _toggle_playback(self):
        if self.player.playing:
            self.player.pause()
            self.btn_play.setText("▶ Play")
        else:
            self.player.play()
            self.btn_play.setText("⏸ Pause")

    def _stop_playback(self):
        self.player.stop()
        self.last_clock_time = 0.0
        self.btn_play.setText("▶ Play")
        self.viewport.reset_simulation()

    def _on_seek_moved(self, val: int):
        dur = self.player.duration
        target_t = (val / 10000.0) * dur
        self.player.seek(target_t)
        self.last_clock_time = target_t

    def _on_reset_sim(self):
        self.viewport.reset_simulation()

    # -------------------------------------------------------------
    # SIMULATION PLUGIN HANDLING
    # -------------------------------------------------------------
    def _on_sim_changed(self, name: str):
        if name in REGISTRY:
            self.project.sim_name = name
            self._refresh_sim_ui()
            self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])
            self.viewport.reset_simulation()

    def _refresh_sim_ui(self):
        name = self.project.sim_name
        cls = REGISTRY.get(name)
        if not cls:
            return
        self.combo_sim.blockSignals(True)
        self.combo_sim.setCurrentText(name)
        self.combo_sim.blockSignals(False)
        self.lbl_sim_desc.setText(cls.description)
        vals = self.project.sim_values.setdefault(name, {})
        self.panel_sim.build(cls.params, vals, project=self.project, target_type="sim")

    def _on_sim_param_changed(self, key: str, val: float):
        vals = self.project.sim_values.setdefault(self.project.sim_name, {})
        vals[key] = val

    def _refresh_fx_ui(self):
        self.panel_fx.build(POST_PARAMS, self.project.post_values, project=self.project, target_type="post")

    def _on_fx_param_changed(self, key: str, val: float):
        self.project.post_values[key] = val

    # -------------------------------------------------------------
    # SPAWNER HANDLING
    # -------------------------------------------------------------
    def _refresh_spawners_list(self):
        self.combo_spawner.blockSignals(True)
        self.combo_spawner.clear()
        for sp in self.project.spawners:
            self.combo_spawner.addItem(sp.label(), sp.uid)
        self.combo_spawner.blockSignals(False)
        self.canvas.update()

    def _on_spawner_selected(self, uid: int):
        if uid > 0:
            self.selected_spawner_uid = uid
            idx = self.combo_spawner.findData(uid)
            if idx >= 0:
                self.combo_spawner.blockSignals(True)
                self.combo_spawner.setCurrentIndex(idx)
                self.combo_spawner.blockSignals(False)
            self.canvas.set_selected_uid(uid)
            self.viewport.set_selected_uid(uid)
            self._refresh_spawner_inspector()
            self.right_tabs.setCurrentIndex(1)  # Focus Spawners tab
        else:
            self.selected_spawner_uid = None
            self.canvas.set_selected_uid(None)
            self.viewport.set_selected_uid(None)

    def _on_combo_spawner_changed(self, idx: int):
        uid = self.combo_spawner.currentData()
        if uid:
            self.selected_spawner_uid = uid
            self.canvas.set_selected_uid(uid)
            self.viewport.set_selected_uid(uid)
            self._refresh_spawner_inspector()

    def _refresh_spawner_inspector(self):
        if not self.selected_spawner_uid:
            return
        sp = self.project.spawner_by_uid(self.selected_spawner_uid)
        if not sp:
            return

        # Update stem dropdown
        self.combo_spawner_stem.blockSignals(True)
        self.combo_spawner_stem.clear()
        if self.project.stems:
            for s_idx, stem in enumerate(self.project.stems):
                self.combo_spawner_stem.addItem(f"Stem {s_idx + 1}: {stem.label()}", s_idx)
            sel_stem_idx = max(0, min(sp.stem, len(self.project.stems) - 1))
            self.combo_spawner_stem.setCurrentIndex(sel_stem_idx)
        else:
            self.combo_spawner_stem.addItem("No Audio Loaded", 0)
        self.combo_spawner_stem.blockSignals(False)

        h = sp.params.get("hue", 180.0)
        s = sp.params.get("sat", 1.0)
        v = sp.params.get("val", 1.0)
        self.color_wheel.set_hsv(h, s, v)
        self.panel_spawner.build(SPAWNER_PARAMS, sp.params, project=self.project,
                                  target_type="spawner", spawner_uid=sp.uid)

    def _on_spawner_stem_changed(self, idx: int):
        if not self.selected_spawner_uid:
            return
        sp = self.project.spawner_by_uid(self.selected_spawner_uid)
        if sp and self.project.stems:
            sp.stem = max(0, min(idx, len(self.project.stems) - 1))

    def _on_color_changed(self, h: float, s: float, v: float):
        if not self.selected_spawner_uid:
            return
        sp = self.project.spawner_by_uid(self.selected_spawner_uid)
        if sp:
            sp.params["hue"] = h
            sp.params["sat"] = s
            sp.params["val"] = v
            self.canvas.update()
            self.viewport.update()

    def _on_spawner_param_changed(self, key: str, val: float):
        if not self.selected_spawner_uid:
            return
        sp = self.project.spawner_by_uid(self.selected_spawner_uid)
        if sp:
            sp.params[key] = val
            self.canvas.update()
            self.viewport.update()

    def _on_spawner_moved(self, uid: int, x: float, y: float):
        sp = self.project.spawner_by_uid(uid)
        if sp:
            sp.x, sp.y = x, y
            self.canvas.update()
            self.viewport.update()

    def _on_spawner_angle_changed(self, uid: int, ang: float):
        sp = self.project.spawner_by_uid(uid)
        if sp:
            sp.params["angle"] = ang
            if self.selected_spawner_uid == uid:
                self.panel_spawner.set_values({"angle": ang})
            self.canvas.update()
            self.viewport.update()

    def _on_canvas_double_clicked(self, x: float, y: float):
        sp = self.project.new_spawner(x, y)
        self._refresh_spawners_list()
        self._on_spawner_selected(sp.uid)
        self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])

    def _add_spawner(self):
        sp = self.project.new_spawner(0.5, 0.5)
        self._refresh_spawners_list()
        self._on_spawner_selected(sp.uid)
        self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])

    def _remove_spawner(self):
        if self.selected_spawner_uid:
            self.project.remove_spawner(self.selected_spawner_uid)
            self.selected_spawner_uid = self.project.spawners[0].uid if self.project.spawners else None
            self._refresh_spawners_list()
            self._refresh_spawner_inspector()
            self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])

    def _duplicate_spawner(self):
        if self.selected_spawner_uid:
            sp = self.project.duplicate_spawner(self.selected_spawner_uid)
            if sp:
                self._refresh_spawners_list()
                self._on_spawner_selected(sp.uid)
                self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])

    def _on_grid_changed(self):
        self.project.grid["cols"] = self.spin_grid_cols.value()
        self.project.grid["rows"] = self.spin_grid_rows.value()
        self.canvas.update()

    # -------------------------------------------------------------
    # AUDIO STEMS LOADING & ANALYSIS
    # -------------------------------------------------------------
    def _on_load_audio_dialog(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio File(s)", "",
            "Audio Files (*.wav *.mp3 *.flac *.ogg *.aif);;All Files (*.*)"
        )
        if not files:
            return

        self.statusBar().showMessage(f"Loading and analysing {len(files)} tracks...")
        self.player.pause()

        stems_data = []
        feature_matrices = []
        sr_ref = None

        # Reset stems list for fresh track batch
        self.project.stems.clear()

        for path in files:
            sig = file_signature(path)
            stereo, sr = load_audio(path)
            if sr_ref is None:
                sr_ref = sr
            stems_data.append(stereo)
            # Fast cached analysis
            feats = cached_analyze((sig, sr), lambda: (stereo, sr))
            feature_matrices.append(feats)
            self.project.stems.append(StemSpec(path=path))

        self.audio_stems_raw = stems_data
        self.bank = FeatureBank(sr=sr_ref, stems=feature_matrices)
        
        # Calculate mix feature matrix
        if feature_matrices:
            max_frames = max(f.shape[1] for f in feature_matrices)
            mix_feats = sum(
                (f if f.shape[1] == max_frames else np.pad(f, ((0, 0), (0, max_frames - f.shape[1]))))
                for f in feature_matrices
            ) / len(feature_matrices)
            self.bank.mix = mix_feats

        self.player.set_audio(stems_data, sr_ref)
        self.viewport.set_audio_bank(self.bank, self.player.duration)

        # Distribute loaded stems across spawners if needed
        num_stems = len(self.project.stems)
        for i, sp in enumerate(self.project.spawners):
            sp.stem = i % num_stems

        self._refresh_stems_list()
        self._refresh_spawner_inspector()
        self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])
        self.statusBar().showMessage(f"Audio ready: {len(files)} stems loaded.", 5000)

    def _on_split_bands(self):
        if not self.audio_stems_raw:
            QMessageBox.warning(self, "No Audio", "Please load an audio file first!")
            return

        sr = self.player.sr
        base_stereo = self.audio_stems_raw[0]
        edges = band_edges(6)
        band_names = ["Sub", "Low", "Low-Mid", "Mid", "High", "Air"]

        new_stems_feats = []
        for i, (lo, hi) in enumerate(edges):
            b_audio = bandpass(base_stereo, sr, lo, hi)
            feats = cached_analyze((f"band_{i}_{lo}_{hi}", file_signature(self.project.stems[0].path)),
                                   lambda: (b_audio, sr))
            new_stems_feats.append(feats)
            self.project.stems.append(StemSpec(path=self.project.stems[0].path, kind="band", lo=lo, hi=hi, name=band_names[i]))

        self.bank.stems.extend(new_stems_feats)
        self.viewport.set_audio_bank(self.bank, self.player.duration)
        self._refresh_stems_list()
        self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])
        QMessageBox.information(self, "Bands Split", "Created 6 frequency band modulation sources!")

    def _refresh_stems_list(self):
        self.list_stems.clear()
        for idx, stem in enumerate(self.project.stems):
            self.list_stems.addItem(f"{idx + 1}. {stem.label()}")
        self._refresh_sim_ui()
        self._refresh_spawner_inspector()
        self._refresh_fx_ui()

    def _on_routes_modified(self):
        sim_params = self.viewport.engine.sim_params() if (self.viewport and self.viewport.engine) else []
        self.matrix_panel.refresh(sim_params)
        self.panel_sim.refresh_modulations()
        self.panel_spawner.refresh_modulations()
        self.panel_fx.refresh_modulations()

    # -------------------------------------------------------------
    # RENDER & PROJECT SERIALIZATION
    # -------------------------------------------------------------
    def _on_open_render(self):
        dlg = RenderDialog(
            project=self.project,
            bank=self.bank,
            duration=self.player.duration,
            audio_stems=self.audio_stems_raw,
            gains=self.player.gains,
            sr=self.player.sr,
            parent=self
        )
        dlg.exec()

    def _reload_audio_from_project(self, project_dir: str):
        if not self.project.stems:
            self.audio_stems_raw = []
            self.bank = None
            self.player.close()
            self._refresh_stems_list()
            return

        file_stems = [s for s in self.project.stems if s.kind == "file"]
        band_stems = [s for s in self.project.stems if s.kind == "band"]

        stems_data = []
        feature_matrices = []
        sr_ref = None

        for stem in file_stems:
            path = stem.path
            if not os.path.isabs(path) or not os.path.exists(path):
                cand = os.path.join(project_dir, path)
                if os.path.exists(cand):
                    path = cand
                else:
                    cand = os.path.join(project_dir, os.path.basename(path))
                    if os.path.exists(cand):
                        path = cand
            if not os.path.exists(path):
                print(f"Warning: Audio file {stem.path} not found.")
                continue

            stem.path = path
            sig = file_signature(path)
            stereo, sr = load_audio(path)
            if sr_ref is None:
                sr_ref = sr
            stems_data.append(stereo)
            feats = cached_analyze((sig, sr), lambda: (stereo, sr))
            feature_matrices.append(feats)

        if not stems_data:
            self.statusBar().showMessage("Notice: Audio file specified in project not found.", 5000)
            self._refresh_stems_list()
            return

        self.audio_stems_raw = stems_data
        self.bank = FeatureBank(sr=sr_ref, stems=feature_matrices)

        if feature_matrices:
            max_frames = max(f.shape[1] for f in feature_matrices)
            mix_feats = sum(
                (f if f.shape[1] == max_frames else np.pad(f, ((0, 0), (0, max_frames - f.shape[1]))))
                for f in feature_matrices
            ) / len(feature_matrices)
            self.bank.mix = mix_feats

        # If there were frequency band stems, recreate them
        if band_stems and self.audio_stems_raw:
            base_stereo = self.audio_stems_raw[0]
            for b in band_stems:
                b_audio = bandpass(base_stereo, sr_ref, b.lo, b.hi)
                feats = cached_analyze((f"band_{b.name}_{b.lo}_{b.hi}", file_signature(file_stems[0].path)),
                                       lambda: (b_audio, sr_ref))
                self.bank.stems.append(feats)

        self.player.set_audio(stems_data, sr_ref)
        self.viewport.set_audio_bank(self.bank, self.player.duration)
        self._refresh_stems_list()
        self.statusBar().showMessage(f"Audio ready: {len(file_stems)} tracks loaded.", 4000)

    def _on_new_project(self):
        if QMessageBox.question(self, "New Project", "Create a new project?") == QMessageBox.Yes:
            self.project = Project()
            self.viewport.project = self.project
            self.canvas.project = self.project
            self.matrix_panel.project = self.project
            if self.viewport.engine:
                self.viewport.engine.project = self.project
                self.viewport.engine.mods.project = self.project
            self.audio_stems_raw = []
            self.bank = None
            self.player.close()
            self._load_default_spawners()
            self._refresh_sim_ui()
            self._refresh_fx_ui()
            self._refresh_spawners_list()
            self._refresh_spawner_inspector()
            self._refresh_stems_list()
            self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])
            self.viewport.reset_simulation()
            self.canvas.update()
            self.viewport.update()

    def _on_save_project(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Project", "MyLiquidProject.json", "JSON (*.json)")
        if not path:
            return

        project_dir = os.path.dirname(os.path.abspath(path))

        # Copy audio file(s) into project directory if not already there,
        # so the project is completely self-contained with its WAV and all settings!
        for stem in self.project.stems:
            if stem.kind == "file" and stem.path:
                orig_path = os.path.abspath(stem.path) if not os.path.isabs(stem.path) else stem.path
                if not os.path.exists(orig_path):
                    cand = os.path.join(project_dir, stem.path)
                    if os.path.exists(cand):
                        orig_path = cand
                if os.path.exists(orig_path):
                    dest_path = os.path.join(project_dir, os.path.basename(orig_path))
                    if os.path.abspath(orig_path) != os.path.abspath(dest_path):
                        try:
                            shutil.copy2(orig_path, dest_path)
                        except Exception as e:
                            print(f"Notice: Could not copy audio to project directory: {e}")
                    stem.path = os.path.basename(orig_path)

        self.project.grid["cols"] = self.spin_grid_cols.value()
        self.project.grid["rows"] = self.spin_grid_rows.value()
        self.project.save(path)
        self.statusBar().showMessage(f"Project and WAV saved: {path}", 5000)

    def _on_save_project_as(self):
        self._on_save_project()

    def open_project_path(self, path: str):
        if not path or not os.path.exists(path):
            return

        project_dir = os.path.dirname(os.path.abspath(path))
        self.project = Project.load(path)

        # Re-link project reference to all sub-components
        self.viewport.project = self.project
        self.canvas.project = self.project
        self.matrix_panel.project = self.project
        if self.viewport.engine:
            self.viewport.engine.project = self.project
            self.viewport.engine.mods.project = self.project

        # Reload the audio and feature matrices
        self._reload_audio_from_project(project_dir)

        # Restore UI
        self.combo_sim.blockSignals(True)
        self.combo_sim.setCurrentText(self.project.sim_name)
        self.combo_sim.blockSignals(False)

        self.selected_spawner_uid = self.project.spawners[0].uid if self.project.spawners else None

        self._refresh_sim_ui()
        self._refresh_fx_ui()
        self._refresh_spawners_list()
        self._refresh_spawner_inspector()
        self.spin_grid_cols.setValue(self.project.grid.get("cols", 16))
        self.spin_grid_rows.setValue(self.project.grid.get("rows", 9))
        self.matrix_panel.refresh(self.viewport.engine.sim_params() if self.viewport.engine else [])
        self.viewport.reset_simulation()
        self.canvas.update()
        self.viewport.update()
        self.statusBar().showMessage(f"Project loaded with audio: {path}", 5000)

    def _on_open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Project", "", "JSON (*.json)")
        if path:
            self.open_project_path(path)

    def closeEvent(self, event):
        self.render_timer.stop()
        self.player.close()
        self.viewport.cleanup()
        event.accept()

