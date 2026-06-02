from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget


HISTORY_LEN = 600


class DerivativePlotWindow(QWidget):
    """Side-by-side rolling plots of EVS event-stream derivative signals.

    Left: total activity (|positive| + |negative|) — how much ∂I/∂t energy.
    Right: net polarity (positive − negative) — balance of brightness change.
    A stable scene reads as a flat line near zero on both.
    """

    def __init__(self, history_len: int = HISTORY_LEN, parent=None):
        super().__init__(parent)
        self.setWindowTitle("EVS Derivative Signals")
        self.resize(1000, 360)

        self._history_len = history_len
        self._pos = deque(maxlen=history_len)
        self._neg = deque(maxlen=history_len)
        self._frames = deque(maxlen=history_len)

        pg.setConfigOptions(antialias=True, background="#1a1a1a", foreground="#cccccc")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        glw = pg.GraphicsLayoutWidget()
        layout.addWidget(glw)

        self._total_plot = glw.addPlot(title="Total activity  |∂I/∂t|  (pos + neg)")
        self._total_plot.setLabel("left", "changed pixels")
        self._total_plot.setLabel("bottom", "frame")
        self._total_plot.showGrid(x=True, y=True, alpha=0.2)
        self._total_curve = self._total_plot.plot(
            pen=pg.mkPen("#33aaff", width=2)
        )

        self._net_plot = glw.addPlot(title="Net polarity  (pos − neg)")
        self._net_plot.setLabel("left", "Δ pixels")
        self._net_plot.setLabel("bottom", "frame")
        self._net_plot.showGrid(x=True, y=True, alpha=0.2)
        self._net_plot.addLine(
            y=0, pen=pg.mkPen("#888888", width=1, style=Qt.PenStyle.DashLine)
        )
        self._net_curve = self._net_plot.plot(pen=pg.mkPen("#ffaa33", width=2))

    def push(self, frame_id: int, pos_count: int, neg_count: int):
        self._frames.append(frame_id)
        self._pos.append(pos_count)
        self._neg.append(neg_count)

        frames = np.fromiter(self._frames, dtype=np.int64)
        pos = np.fromiter(self._pos, dtype=np.int64)
        neg = np.fromiter(self._neg, dtype=np.int64)

        self._total_curve.setData(frames, pos + neg)
        self._net_curve.setData(frames, pos - neg)

    def clear(self):
        self._frames.clear()
        self._pos.clear()
        self._neg.clear()
        self._total_curve.clear()
        self._net_curve.clear()
