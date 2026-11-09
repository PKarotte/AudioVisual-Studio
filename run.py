"""Direct launcher for AudioVisual Studio."""
import os
import sys
import importlib

this_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(this_dir)
for p in (this_dir, parent_dir):
    if p not in sys.path:
        sys.path.insert(0, p)

pkg_name = os.path.basename(this_dir)

try:
    mod = importlib.import_module(f"{pkg_name}.gui.main_window")
    MainWindow = mod.MainWindow
except Exception:
    from gui.main_window import MainWindow

from PySide6.QtWidgets import QApplication
from PySide6.QtCore import Qt
from PySide6.QtGui import QPalette, QColor


def setup_dark_theme(app: QApplication):
    app.setStyle("Fusion")
    dark_palette = QPalette()
    dark_palette.setColor(QPalette.Window, QColor("#14171d"))
    dark_palette.setColor(QPalette.WindowText, QColor("#d0d4dc"))
    dark_palette.setColor(QPalette.Base, QColor("#1c2027"))
    dark_palette.setColor(QPalette.AlternateBase, QColor("#222730"))
    dark_palette.setColor(QPalette.ToolTipBase, QColor("#ffffff"))
    dark_palette.setColor(QPalette.ToolTipText, QColor("#ffffff"))
    dark_palette.setColor(QPalette.Text, QColor("#e0e4ec"))
    dark_palette.setColor(QPalette.Button, QColor("#232832"))
    dark_palette.setColor(QPalette.ButtonText, QColor("#d0d4dc"))
    dark_palette.setColor(QPalette.BrightText, Qt.red)
    dark_palette.setColor(QPalette.Link, QColor("#4488ff"))
    dark_palette.setColor(QPalette.Highlight, QColor("#3066a3"))
    dark_palette.setColor(QPalette.HighlightedText, Qt.white)
    app.setPalette(dark_palette)


def main():
    QApplication.setHighDpiScaleFactorRoundingPolicy(Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    app = QApplication(sys.argv)
    setup_dark_theme(app)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
