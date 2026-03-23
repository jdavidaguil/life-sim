"""Entry point for the Life-Sim Research Workbench desktop application."""

import os
os.environ["QT_API"] = "pyside6"  # tell matplotlib which Qt binding to use
import matplotlib
matplotlib.use("QtAgg")            # must happen before any pyplot / reporter import

from PySide6.QtWidgets import QApplication
from app.main_window import MainWindow, DARK_STYLE
import sys


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_STYLE)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
