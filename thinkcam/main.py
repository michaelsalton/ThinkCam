import sys

from PySide6.QtWidgets import QApplication

from thinkcam.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("ThinkCam")
    app.setStyle("Fusion")

    window = MainWindow()
    window.resize(1400, 850)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
