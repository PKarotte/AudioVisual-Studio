"""
Interactive canvas with customizable grid, snapping, and draggable spawner nodes.
Users can drag spawner positions, rotate directions, and view real-time state.
"""
from __future__ import annotations

import math
from PySide6.QtCore import Qt, QRectF, QPointF, Signal
from PySide6.QtGui import QPainter, QColor, QPen, QBrush, QFont
from PySide6.QtWidgets import QWidget

from ..project import Project, Spawner


class SpawnerCanvas(QWidget):
    spawnerSelected = Signal(int)       # spawner uid
    spawnerMoved = Signal(int, float, float)  # uid, x, y
    spawnerAngleChanged = Signal(int, float)  # uid, angle_deg
    canvasDoubleClicked = Signal(float, float) # norm_x, norm_y

    def __init__(self, project: Project, parent=None):
        super().__init__(parent)
        self.project = project
        self.selected_uid: int | None = None
        self.hover_uid: int | None = None
        self.dragging_uid: int | None = None
        self.drag_mode: str | None = None  # "pos" or "angle"
        self.live_state: dict | None = None

        self.setMouseTracking(True)
        self.setMinimumSize(400, 225)  # 16:9 base

    def set_live_state(self, state: dict | None):
        self.live_state = state
        self.update()

    def set_selected_uid(self, uid: int | None):
        if self.selected_uid != uid:
            self.selected_uid = uid
            self.update()

    def _get_canvas_rect(self) -> QRectF:
        """Calculate aspect-ratio preserved canvas rect."""
        w, h = self.width(), self.height()
        target_aspect = self.project.aspect[0] / self.project.aspect[1]
        cur_aspect = w / h if h > 0 else 1.0

        if cur_aspect > target_aspect:
            ch = h - 20
            cw = ch * target_aspect
        else:
            cw = w - 20
            ch = cw / target_aspect
        
        cx = (w - cw) / 2
        cy = (h - ch) / 2
        return QRectF(cx, cy, cw, ch)

    def _norm_to_screen(self, nx: float, ny: float, crect: QRectF) -> QPointF:
        return QPointF(crect.left() + nx * crect.width(), crect.top() + ny * crect.height())

    def _screen_to_norm(self, pt: QPointF, crect: QRectF) -> tuple[float, float]:
        nx = (pt.x() - crect.left()) / crect.width()
        ny = (pt.y() - crect.top()) / crect.height()
        return nx, ny

    def _snap(self, nx: float, ny: float) -> tuple[float, float]:
        if not self.project.grid.get("snap", True):
            return nx, ny
        cols = self.project.grid.get("cols", 16)
        rows = self.project.grid.get("rows", 9)
        snx = round(nx * cols) / cols
        sny = round(ny * rows) / rows
        return max(0.01, min(0.99, snx)), max(0.01, min(0.99, sny))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor("#111317"))

        crect = self._get_canvas_rect()
        painter.fillRect(crect, QColor("#181b22"))
        painter.setPen(QPen(QColor("#2d3340"), 1.5))
        painter.drawRect(crect)

        # Grid lines
        cols = self.project.grid.get("cols", 16)
        rows = self.project.grid.get("rows", 9)
        painter.setPen(QPen(QColor("#222834"), 1, Qt.DotLine))
        for c in range(1, cols):
            gx = crect.left() + (c / cols) * crect.width()
            painter.drawLine(QPointF(gx, crect.top()), QPointF(gx, crect.bottom()))
        for r in range(1, rows):
            gy = crect.top() + (r / rows) * crect.height()
            painter.drawLine(QPointF(crect.left(), gy), QPointF(crect.right(), gy))

        # Center crosshair
        painter.setPen(QPen(QColor("#2a3548"), 1, Qt.DashLine))
        painter.drawLine(QPointF(crect.left() + crect.width()*0.5, crect.top()),
                         QPointF(crect.left() + crect.width()*0.5, crect.bottom()))
        painter.drawLine(QPointF(crect.left(), crect.top() + crect.height()*0.5),
                         QPointF(crect.right(), crect.top() + crect.height()*0.5))

        # Draw Spawners
        font = QFont("sans-serif", 9, QFont.Bold)
        painter.setFont(font)

        for sp in self.project.spawners:
            if not sp.enabled:
                continue

            pos = self._norm_to_screen(sp.x, sp.y, crect)
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

            # Live Force / Audio Pulse indicator (rings)
            if self.live_state and "frames" in self.live_state:
                for f in self.live_state["frames"]:
                    if f.uid == sp.uid and f.force > 0.5:
                        pulse_r = r + min(25.0, f.force * 0.3)
                        painter.setBrush(Qt.NoBrush)
                        painter.setPen(QPen(QColor(base_col.red(), base_col.green(), base_col.blue(), 100), 1.5))
                        painter.drawEllipse(pos, pulse_r, pulse_r)

    def _hit_test(self, pos: QPointF, crect: QRectF) -> tuple[int | None, str | None]:
        for sp in reversed(self.project.spawners):
            if not sp.enabled:
                continue
            s_pos = self._norm_to_screen(sp.x, sp.y, crect)
            
            # Test direction handle tip
            ang = sp.params.get("angle", 0.0)
            rad = math.radians(ang)
            dir_len = 38.0
            tip = QPointF(s_pos.x() + math.sin(rad) * dir_len,
                          s_pos.y() - math.cos(rad) * dir_len)
            if math.hypot(pos.x() - tip.x(), pos.y() - tip.y()) <= 8.0:
                return sp.uid, "angle"

            # Test body circle
            if math.hypot(pos.x() - s_pos.x(), pos.y() - s_pos.y()) <= 18.0:
                return sp.uid, "pos"

        return None, None

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            crect = self._get_canvas_rect()
            uid, mode = self._hit_test(event.position(), crect)
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
        crect = self._get_canvas_rect()
        pos = event.position()

        if self.dragging_uid is not None:
            sp = self.project.spawner_by_uid(self.dragging_uid)
            if sp:
                if self.drag_mode == "pos":
                    nx, ny = self._screen_to_norm(pos, crect)
                    if not (event.modifiers() & Qt.ShiftModifier):
                        nx, ny = self._snap(nx, ny)
                    nx = max(0.01, min(0.99, nx))
                    ny = max(0.01, min(0.99, ny))
                    sp.x, sp.y = nx, ny
                    self.spawnerMoved.emit(sp.uid, nx, ny)
                elif self.drag_mode == "angle":
                    s_pos = self._norm_to_screen(sp.x, sp.y, crect)
                    dx = pos.x() - s_pos.x()
                    dy = -(pos.y() - s_pos.y())
                    ang = math.degrees(math.atan2(dx, dy))
                    sp.params["angle"] = ang
                    self.spawnerAngleChanged.emit(sp.uid, ang)
                self.update()
        else:
            uid, _ = self._hit_test(pos, crect)
            if uid != self.hover_uid:
                self.hover_uid = uid
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.dragging_uid = None
            self.drag_mode = None
            self.update()

    def mouseDoubleClickEvent(self, event):
        if event.button() == Qt.LeftButton:
            crect = self._get_canvas_rect()
            nx, ny = self._screen_to_norm(event.position(), crect)
            if 0.0 <= nx <= 1.0 and 0.0 <= ny <= 1.0:
                if not (event.modifiers() & Qt.ShiftModifier):
                    nx, ny = self._snap(nx, ny)
                self.canvasDoubleClicked.emit(nx, ny)

