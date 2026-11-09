"""
Render Export Dialog.
Supports selecting resolutions, codecs (H.264, NVENC, ProRes, PNG sequence),
pre-roll, start/end trimming, and runs the export asynchronously with progress tracking.
"""
from __future__ import annotations

import os
import threading
from PySide6.QtCore import Qt, Signal, QObject
from PySide6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QSpinBox,
    QDoubleSpinBox, QLineEdit, QPushButton, QProgressBar, QLabel, QFileDialog,
    QMessageBox
)

from ..project import Project
from ..render import CODECS, render
from ..audio_analysis import FeatureBank


class RenderSignals(QObject):
    progress = Signal(int, int, float)  # frame, total, fps
    finished = Signal(str)             # output path
    error = Signal(str)


class RenderDialog(QDialog):
    def __init__(self, project: Project, bank: FeatureBank | None, duration: float,
                 audio_stems: list, gains: list, sr: int, parent=None):
        super().__init__(parent)
        self.project = project
        self.bank = bank
        self.duration = duration
        self.audio_stems = audio_stems
        self.gains = gains
        self.sr = sr

        self.signals = RenderSignals()
        self.signals.progress.connect(self._on_progress)
        self.signals.finished.connect(self._on_finished)
        self.signals.error.connect(self._on_error)

        self.cur_time = parent.player.time() if parent and hasattr(parent, "player") else 0.0
        self.is_rendering = False
        self.cancel_requested = False

        self.setWindowTitle("Render & Export Video")
        self.resize(520, 420)
        self._init_ui()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        form = QFormLayout()

        # Preset resolution
        self.combo_res = QComboBox()
        self.combo_res.addItems([
            "1280x720 (720p HD)",
            "1920x1080 (1080p Full HD)",
            "2560x1440 (1440p 2K QHD)",
            "3840x2160 (4K UHD)",
            "Custom"
        ])
        self.combo_res.setCurrentIndex(1)
        self.combo_res.currentIndexChanged.connect(self._on_res_preset)

        res_box = QHBoxLayout()
        self.spin_w = QSpinBox()
        self.spin_w.setRange(256, 7680)
        self.spin_w.setValue(self.project.render.get("width", 1920))
        self.spin_h = QSpinBox()
        self.spin_h.setRange(144, 4320)
        self.spin_h.setValue(self.project.render.get("height", 1080))
        res_box.addWidget(self.spin_w)
        res_box.addWidget(QLabel("x"))
        res_box.addWidget(self.spin_h)

        # FPS
        self.spin_fps = QSpinBox()
        self.spin_fps.setRange(15, 120)
        self.spin_fps.setValue(self.project.render.get("fps", 60))

        # Codec
        self.combo_codec = QComboBox()
        self.combo_codec.addItems(list(CODECS.keys()))
        codec_name = self.project.render.get("codec", "H.264 (x264)")
        if codec_name in CODECS:
            self.combo_codec.setCurrentText(codec_name)

        # Quality (CRF/CQ)
        self.spin_quality = QSpinBox()
        self.spin_quality.setRange(0, 51)
        self.spin_quality.setValue(self.project.render.get("quality", 18))
        self.spin_quality.setToolTip("Lower is higher quality (18 is visually lossless)")

        # Simulation Grid Quality
        self.combo_sim_quality = QComboBox()
        self.combo_sim_quality.addItems([
            "Auto (Match Video Height for Max Quality)",
            "Ultra (1080p Grid)",
            "High (768p Grid)",
            "Medium (512p Grid)",
            "Project Default (Use Viewport Setting)"
        ])
        self.combo_sim_quality.setCurrentIndex(0)
        self.combo_sim_quality.setToolTip("Velocity simulation grid resolution. 'Auto' computes fluid dynamics at full video resolution for razor-sharp swirls and vortices.")

        # Timeline Trimming
        time_box = QHBoxLayout()
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setRange(0.0, 99999.0)
        self.spin_start.setSuffix(" s")
        self.spin_start.setValue(self.project.render.get("start", 0.0))

        self.spin_end = QDoubleSpinBox()
        self.spin_end.setRange(0.0, 99999.0)
        self.spin_end.setSuffix(" s")
        end_val = self.project.render.get("end", 0.0) or self.duration or 30.0
        self.spin_end.setValue(end_val)

        time_box.addWidget(QLabel("Start:"))
        time_box.addWidget(self.spin_start)
        time_box.addWidget(QLabel("End:"))
        time_box.addWidget(self.spin_end)

        # Quick Range presets
        range_btn_box = QHBoxLayout()
        btn_full = QPushButton("⏮ Full Track")
        btn_full.setToolTip("Export the full audio track from 0:00 to end")
        btn_full.clicked.connect(self._on_full_track)

        btn_cur_time = QPushButton(f"📍 From Current ({self.cur_time:.1f}s)")
        btn_cur_time.setToolTip(f"Set start time to current player position ({self.cur_time:.2f}s)")
        btn_cur_time.clicked.connect(self._on_use_cur_time)

        btn_clip_10 = QPushButton("⚡ 10s Clip")
        btn_clip_10.setToolTip("Export a 10-second preview clip starting from current position")
        btn_clip_10.clicked.connect(lambda: self._on_quick_clip(10.0))

        btn_clip_30 = QPushButton("⚡ 30s Clip")
        btn_clip_30.setToolTip("Export a 30-second preview clip starting from current position")
        btn_clip_30.clicked.connect(lambda: self._on_quick_clip(30.0))

        range_btn_box.addWidget(btn_full)
        range_btn_box.addWidget(btn_cur_time)
        range_btn_box.addWidget(btn_clip_10)
        range_btn_box.addWidget(btn_clip_30)

        # Pre-roll
        self.spin_preroll = QDoubleSpinBox()
        self.spin_preroll.setRange(0.0, 10.0)
        self.spin_preroll.setSingleStep(0.5)
        self.spin_preroll.setSuffix(" s")
        init_preroll = self.project.render.get("preroll", 0.0 if self.spin_start.value() == 0.0 else 2.5)
        self.spin_preroll.setValue(init_preroll)
        self.spin_preroll.setToolTip("Warm up fluid simulation before start time so motion and dye match the preview (used when start > 0)")

        # Output path
        path_box = QHBoxLayout()
        self.txt_path = QLineEdit()
        default_out = self.project.render.get("path", "")
        if not default_out:
            default_out = os.path.abspath("Fluid_Export.mp4")
        self.txt_path.setText(default_out)
        btn_browse = QPushButton("Browse...")
        btn_browse.clicked.connect(self._on_browse)
        path_box.addWidget(self.txt_path)
        path_box.addWidget(btn_browse)

        form.addRow("Resolution Preset:", self.combo_res)
        form.addRow("Resolution (WxH):", res_box)
        form.addRow("Framerate (FPS):", self.spin_fps)
        form.addRow("Encoding Format:", self.combo_codec)
        form.addRow("Quality (CRF/CQ):", self.spin_quality)
        form.addRow("Simulation Grid:", self.combo_sim_quality)
        form.addRow("Render Time Range:", time_box)
        form.addRow("Quick Presets:", range_btn_box)
        form.addRow("Simulation Pre-roll:", self.spin_preroll)
        form.addRow("Output File:", path_box)

        layout.addLayout(form)

        # Progress elements
        self.lbl_status = QLabel("Ready to export.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        layout.addWidget(self.lbl_status)
        layout.addWidget(self.progress_bar)

        # Action Buttons
        btn_box = QHBoxLayout()
        self.btn_cancel = QPushButton("Close")
        self.btn_cancel.clicked.connect(self._on_close_clicked)
        self.btn_render = QPushButton("🚀 Start Render")
        self.btn_render.setStyleSheet("background-color: #2b5588; font-weight: bold; padding: 6px 14px;")
        self.btn_render.clicked.connect(self._start_render)

        btn_box.addStretch(1)
        btn_box.addWidget(self.btn_cancel)
        btn_box.addWidget(self.btn_render)
        layout.addLayout(btn_box)

    def _on_full_track(self):
        self.spin_start.setValue(0.0)
        end_val = self.duration or 30.0
        self.spin_end.setValue(round(end_val, 2))
        self.spin_preroll.setValue(0.0)

    def _on_use_cur_time(self):
        t = max(0.0, getattr(self, "cur_time", 0.0))
        self.spin_start.setValue(round(t, 2))
        if t > 0:
            self.spin_preroll.setValue(2.5)

    def _on_quick_clip(self, length: float):
        t = max(0.0, getattr(self, "cur_time", 0.0))
        self.spin_start.setValue(round(t, 2))
        max_dur = self.duration or (t + length)
        self.spin_end.setValue(round(min(max_dur, t + length), 2))
        if t > 0:
            self.spin_preroll.setValue(2.5)

    def _on_res_preset(self, idx: int):
        presets = [(1280, 720), (1920, 1080), (2560, 1440), (3840, 2160)]
        if idx < len(presets):
            w, h = presets[idx]
            self.spin_w.setValue(w)
            self.spin_h.setValue(h)

    def _on_browse(self):
        codec = self.combo_codec.currentText()
        ext = CODECS[codec]["ext"]
        if ext == "":  # PNG Sequence
            path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        else:
            filter_str = f"Video Files (*{ext});;All Files (*.*)"
            path, _ = QFileDialog.getSaveFileName(self, "Save Video As", self.txt_path.text(), filter_str)
        if path:
            self.txt_path.setText(path)

    def _start_render(self):
        if self.is_rendering:
            return

        settings = {
            "width": self.spin_w.value(),
            "height": self.spin_h.value(),
            "fps": self.spin_fps.value(),
            "codec": self.combo_codec.currentText(),
            "quality": self.spin_quality.value(),
            "sim_quality": self.combo_sim_quality.currentText(),
            "start": self.spin_start.value(),
            "end": self.spin_end.value(),
            "preroll": self.spin_preroll.value(),
            "path": self.txt_path.text()
        }
        self.project.render.update(settings)

        self.is_rendering = True
        self.cancel_requested = False
        self.btn_render.setEnabled(False)
        self.btn_cancel.setText("Cancel Render")
        self.lbl_status.setText("Rendering in background...")
        self.progress_bar.setValue(0)

        # Launch rendering in background worker thread
        p_dict = self.project.to_dict()

        def worker():
            try:
                def prog(done, total, fps):
                    self.signals.progress.emit(done, total, fps)

                out = render(
                    project_dict=p_dict,
                    bank=self.bank,
                    duration=self.duration,
                    audio_stems=self.audio_stems,
                    gains=self.gains,
                    sr=self.sr,
                    settings=settings,
                    progress=prog,
                    cancelled=lambda: self.cancel_requested,
                    log=print
                )
                self.signals.finished.emit(out)
            except Exception as e:
                self.signals.error.emit(str(e))

        t = threading.Thread(target=worker, daemon=True)
        t.start()

    def _on_progress(self, done: int, total: int, fps: float):
        pct = int((done / total) * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.lbl_status.setText(f"Encoding Frame {done}/{total} ({pct}%) — {fps:.1f} FPS")

    def _on_finished(self, out_path: str):
        self.is_rendering = False
        self.btn_render.setEnabled(True)
        self.btn_cancel.setText("Close")
        self.lbl_status.setText(f"✅ Render complete: {out_path}")
        self.progress_bar.setValue(100)
        QMessageBox.information(self, "Render Finished", f"Video successfully rendered to:\n{out_path}")

    def _on_error(self, err_msg: str):
        self.is_rendering = False
        self.btn_render.setEnabled(True)
        self.btn_cancel.setText("Close")
        self.lbl_status.setText("❌ Render failed.")
        QMessageBox.critical(self, "Render Error", f"Render failed with error:\n{err_msg}")

    def _on_close_clicked(self):
        if self.is_rendering:
            if QMessageBox.question(self, "Cancel Render", "Cancel ongoing render job?") == QMessageBox.Yes:
                self.cancel_requested = True
        else:
            self.accept()

