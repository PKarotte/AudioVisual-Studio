"""
OpenGL Live Viewport with ModernGL integration and aspect-ratio preservation.
Renders real-time simulation output, blits to letterboxed frame, and supports
interactive spawner overlay so user placement matches simulation 1:1.
"""
from __future__ import annotations

import math
import numpy as np
import moderngl
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtCore import Qt, Signal, QRectF, QPointF
from PySide6.QtGui import QSurfaceFormat, QPainter, QColor, QPen, QBrush, QFont

from ..engine import Engine
from ..project import Project
from ..audio_analysis import FeatureBank


class ViewportWidget(QOpenGLWidget):
    frameRendered = Signal(dict)              # passes engine.last_state
    spawnerSelected = Signal(int)             # spawner uid
    spawnerMoved = Signal(int, float, float)  # uid, x, y
    spawnerAngleChanged = Signal(int, float)  # uid, angle_deg
    canvasDoubleClicked = Signal(float, float)# norm_x, norm_y

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.ctx: moderngl.Context | None = None
        self.engine: Engine | None = None
        self.bank: FeatureBank | None = None
        self.duration: float = 0.0

        # Spawner overlay & interaction state
        self.show_overlay: bool = True
        self.selected_uid: int | None = None
        self.hover_uid: int | None = None
        self.dragging_uid: int | None = None
        self.drag_mode: str | None = None  # "pos" or "angle"

        # Full-screen blit pass
        self.blit_prog = None
        self.blit_vao = None
        self.quad_buf = None

        # Setup high performance format
        self._cur_t = 0.0
        self._cur_dt = 1.0 / 60.0
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.CoreProfile)
        fmt.setSwapInterval(1)  # V-Sync
        self.setFormat(fmt)
        self.setMouseTracking(True)

    def set_show_overlay(self, show: bool):
        self.show_overlay = bool(show)
        self.update()

    def set_selected_uid(self, uid: int | None):
        if self.selected_uid != uid:
            self.selected_uid = uid
            self.update()

    def set_audio_bank(self, bank: FeatureBank | None, duration: float):
        self.bank = bank
        self.duration = duration
        if self.engine:
            self.engine.set_bank(bank, duration)

    def _calc_viewport_rect(self) -> tuple[int, int, int, int]:
        """Compute letterbox / pillarbox rectangle matching project aspect ratio."""
        w, h = self.width(), self.height()
        target_aspect = self.project.aspect[0] / self.project.aspect[1]
        cur_aspect = w / h if h > 0 else 1.0

        if cur_aspect > target_aspect:
            vh = h
            vw = int(round(vh * target_aspect))
        else:
            vw = w
            vh = int(round(vw / target_aspect))
        
        vw = max(16, (vw // 2) * 2)
        vh = max(16, (vh // 2) * 2)
        vx = (w - vw) // 2
        vy = (h - vh) // 2
        return vx, vy, vw, vh

    def _norm_to_screen(self, nx: float, ny: float, vx: int, vy: int, vw: int, vh: int) -> QPointF:
        return QPointF(vx + nx * vw, vy + ny * vh)

    def _screen_to_norm(self, pt: QPointF, vx: int, vy: int, vw: int, vh: int) -> tuple[float, float]:
        nx = (pt.x() - vx) / vw
        ny = (pt.y() - vy) / vh
        return nx, ny

    def _snap(self, nx: float, ny: float) -> tuple[float, float]:
        if not self.project.grid.get("snap", True):
            return nx, ny
        cols = self.project.grid.get("cols", 16)
        rows = self.project.grid.get("rows", 9)
        snx = round(nx * cols) / cols
        sny = round(ny * rows) / rows
        return max(0.01, min(0.99, snx)), max(0.01, min(0.99, sny))

    def initializeGL(self):
        # Create moderngl context from Qt active context
        self.ctx = moderngl.create_context()
        self.engine = Engine(self.ctx, self.project, self.bank, self.duration, render_mode=False)
        _, _, vw, vh = self._calc_viewport_rect()
        self.engine.set_output_size(vw, vh)

        # Simple blit shader to draw the engine's final texture to viewport
        vs = """
        #version 330
        in vec2 in_pos;
        out vec2 uv;
        void main() {
            uv = in_pos * 0.5 + 0.5;
            gl_Position = vec4(in_pos, 0.0, 1.0);
        }
        """
        fs = """
        #version 330
        in vec2 uv;
        out vec4 fragColor;
        uniform sampler2D uTex;
        void main() {
            fragColor = texture(uTex, uv);
        }
        """
        self.blit_prog = self.ctx.program(vertex_shader=vs, fragment_shader=fs)
        self.quad_buf = self.ctx.buffer(np.array([-1, -1, 1, -1, -1, 1, 1, 1], dtype="f4").tobytes())
        self.blit_vao = self.ctx.vertex_array(self.blit_prog, [(self.quad_buf, "2f", "in_pos")])

    def resizeGL(self, w: int, h: int):
        if self.engine:
            _, _, vw, vh = self._calc_viewport_rect()
            self.engine.set_output_size(vw, vh)

    def paintGL(self):
        if not self.engine or not self.blit_prog:
            return

        vx, vy, vw, vh = self._calc_viewport_rect()
        if self.engine.out_size != (vw, vh):
            self.engine.set_output_size(vw, vh)

        out_target = self.engine.frame(self._cur_t, self._cur_dt)

        # Qt binds the widget's render target FBO before calling paintGL.
        fbo = self.ctx.detect_framebuffer(self.defaultFramebufferObject())
        fbo.use()
        
        # Clear background margins (#111317)
        self.ctx.clear(0.067, 0.075, 0.090, 1.0)

        # Blit inside aspect-preserved letterbox rect
        fbo.viewport = (vx, vy, vw, vh)

        out_target.tex.use(location=0)
        self.blit_prog["uTex"].value = 0
        self.blit_vao.render(moderngl.TRIANGLE_STRIP)

        if self.engine.last_state:
            self.frameRendered.emit(self.engine.last_state)

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.show_overlay:
            return

        vx, vy, vw, vh = self._calc_viewport_rect()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        font = QFont("sans-serif", 9, QFont.Bold)
        painter.setFont(font)

        # Draw aspect border around active simulation rectangle
        painter.setPen(QPen(QColor("#2d3340"), 1.0))
        painter.drawRect(QRectF(vx, vy, vw, vh))

        for sp in self.project.spawners:
            if not sp.enabled:
                continue

            pos = self._norm_to_screen(sp.x, sp.y, vx, vy, vw, vh)
            h = sp.params.get("hue", 180.0)
            s = sp.params.get("sat", 1.0)
            v = sp.params.get("val", 1.0)
            base_col = QColor.fromHsvF((h % 360) / 360.0, s, v)

            is_sel = (sp.uid == self.selected_uid)
            is_hov = (sp.uid == self.hover_uid)

            # Node circle
            r = 16.0
            painter.setBrush(QBrush(base_col))
            pen_col = QColor("#ffffff") if is_sel else (QColor("#aaaaaa") if is_hov else QColor("#444444"))
            pen_w = 3.0 if is_sel else 1.5
            painter.setPen(QPen(pen_col, pen_w))
            painter.drawEllipse(pos, r, r)

            # Spawner Number
            painter.setPen(QPen(QColor("#000000") if v > 0.4 else QColor("#ffffff")))
            painter.drawText(QRectF(pos.x() - r, pos.y() - r, r*2, r*2), Qt.AlignCenter, str(sp.uid))

            # Direction Handle
            ang = sp.params.get("angle", 0.0)
            rad = math.radians(ang)
            dir_len = 38.0
            tip = QPointF(pos.x() + math.sin(rad) * dir_len,
                          pos.y() - math.cos(rad) * dir_len)
            painter.setPen(QPen(QColor("#ffffff") if is_sel else QColor("#778899"), 2.0))
            painter.drawLine(pos, tip)
            painter.setBrush(QBrush(QColor("#ffffff") if is_sel else QColor("#778899")))
            painter.drawEllipse(tip, 4.0, 4.0)

        painter.end()

    def _hit_test(self, pos: QPointF, vx: int, vy: int, vw: int, vh: int) -> tuple[int | None, str | None]:
        for sp in reversed(self.project.spawners):
            if not sp.enabled:
                continue
            s_pos = self._norm_to_screen(sp.x, sp.y, vx, vy, vw, vh)
            ang = sp.params.get("angle", 0.0)
            rad = math.radians(ang)
            dir_len = 38.0
            tip = QPointF(s_pos.x() + math.sin(rad) * dir_len,
                          s_pos.y() - math.cos(rad) * dir_len)
            if math.hypot(pos.x() - tip.x(), pos.y() - tip.y()) <= 8.0:
                return sp.uid, "angle"
            if math.hypot(pos.x() - s_pos.x(), pos.y() - s_pos.y()) <= 18.0:
                return sp.uid, "pos"
        return None, None

    def mousePressEvent(self, event):
        if not self.show_overlay:
            return super().mousePressEvent(event)
        if event.button() == Qt.LeftButton:
            vx, vy, vw, vh = self._calc_viewport_rect()
            uid, mode = self._hit_test(event.position(), vx, vy, vw, vh)
            if uid is not None:
                self.selected_uid = uid
                self.dragging_uid = uid
                self.drag_mode = mode
                self.spawnerSelected.emit(uid)
                self.update()
            else:
                self.selected_uid = None
                self.spawnerSelected.emit(-1)
                self.update()

    def mouseMoveEvent(self, event):
        if not self.show_overlay:
            return super().mouseMoveEvent(event)
        vx, vy, vw, vh = self._calc_viewport_rect()
        pos = event.position()

        if self.dragging_uid is not None:
            sp = self.project.spawner_by_uid(self.dragging_uid)
            if sp:
                if self.drag_mode == "pos":
                    nx, ny = self._screen_to_norm(pos, vx, vy, vw, vh)
                    if not (event.modifiers() & Qt.ShiftModifier):
                        nx, ny = self._snap(nx, ny)
                    nx = max(0.01, min(0.99, nx))
                    ny = max(0.01, min(0.99, ny))
                    sp.x, sp.y = nx, ny
                    self.spawnerMoved.emit(sp.uid, nx, ny)
                elif self.drag_mode == "angle":
                    s_pos = self._norm_to_screen(sp.x, sp.y, vx, vy, vw, vh)
                    dx = pos.x() - s_pos.x()
                    dy = -(pos.y() - s_pos.y())
                    ang = math.degrees(math.atan2(dx, dy))
                    sp.params["angle"] = ang
                    self.spawnerAngleChanged.emit(sp.uid, ang)
                self.update()
        else:
            uid, _ = self._hit_test(pos, vx, vy, vw, vh)
            if uid != self.hover_uid:
                self.hover_uid = uid
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_uid = None
            self.drag_mode = None
            self.update()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if not self.show_overlay:
            return super().mouseDoubleClickEvent(event)
        if event.button() == Qt.LeftButton:
            vx, vy, vw, vh = self._calc_viewport_rect()
            nx, ny = self._screen_to_norm(event.position(), vx, vy, vw, vh)
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                if not (event.modifiers() & Qt.ShiftModifier):
                    nx, ny = self._snap(nx, ny)
                self.canvasDoubleClicked.emit(nx, ny)

    def render_frame(self, t: float, dt: float):
        self._cur_t = t
        self._cur_dt = dt
        self.update()

    def reset_simulation(self):
        if self.engine:
            self.engine.request_reset()

    def cleanup(self):
        self.makeCurrent()
        if self.engine:
            self.engine.release()
            self.engine = None
        if self.blit_vao:
            self.blit_vao.release()
        if self.blit_prog:
            self.blit_prog.release()
        if self.quad_buf:
            self.quad_buf.release()
        self.doneCurrent()
