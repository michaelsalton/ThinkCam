from PySide6.QtWidgets import QLabel, QStatusBar


def _fmt_rate(r: float) -> str:
    if r < 1e3:
        return f"{r:.0f} ev/s"
    if r < 1e6:
        return f"{r / 1e3:.1f} Kev/s"
    if r < 1e9:
        return f"{r / 1e6:.1f} Mev/s"
    return f"{r / 1e9:.1f} Gev/s"


def _fmt_bw(r: float) -> str:
    if r < 1e3:
        return f"{r:.0f} Bps"
    if r < 1e6:
        return f"{r / 1e3:.1f} KBps"
    if r < 1e9:
        return f"{r / 1e6:.1f} MBps"
    return f"{r / 1e9:.1f} GBps"


def _fmt_bytes(n: float) -> str:
    if n < 1e3:
        return f"{n:.0f} B"
    if n < 1e6:
        return f"{n / 1e3:.1f} KB"
    if n < 1e9:
        return f"{n / 1e6:.1f} MB"
    return f"{n / 1e9:.1f} GB"


def _fmt_dur(s: float) -> str:
    m, sec = divmod(int(s), 60)
    return f"{m:d}:{sec:02d}"


class StatsStatusBar(QStatusBar):
    def __init__(self, parent=None):
        super().__init__(parent)

        self._flash_label = QLabel("")
        self._raw_label = QLabel("")
        self._events_label = QLabel("Events: --")
        self._fps_label = QLabel("FPS: --")
        self._bw_label = QLabel("BW: --")
        self._render_label = QLabel("Render: --")
        self._frame_label = QLabel("Frame: --")

        for label in (
            self._flash_label,
            self._raw_label,
            self._events_label,
            self._fps_label,
            self._bw_label,
            self._render_label,
            self._frame_label,
        ):
            label.setStyleSheet("padding: 0 8px;")
            self.addPermanentWidget(label)

    def update_stats(self, stats: dict):
        if stats.get("flash_filtered"):
            self._flash_label.setText("FLASH SUPPRESSED")
            self._flash_label.setStyleSheet(
                "padding: 0 8px; color: #ffcc33; font-weight: bold;"
            )
        else:
            self._flash_label.setText("")

        self._events_label.setText(f"Events: {_fmt_rate(stats.get('event_rate', 0))}")
        self._fps_label.setText(f"FPS: {stats.get('gvsp_fps', 0):.1f}")
        self._bw_label.setText(f"BW: {_fmt_bw(stats.get('throughput', 0))}")
        self._render_label.setText(f"Render: {stats.get('render_ms', 0):.1f} ms")
        self._frame_label.setText(f"Frame: {stats.get('frame_id', 0)}")

    def update_raw_stats(self, stats: dict):
        dropped = stats.get("raw_dropped_events", 0)
        text = (
            f"● REC {_fmt_dur(stats.get('raw_elapsed_s', 0))}  "
            f"{stats.get('raw_events_written', 0):,} ev  "
            f"{_fmt_rate(stats.get('raw_event_rate', 0))}  "
            f"{_fmt_bytes(stats.get('raw_file_size', 0))}  "
            f"drop {dropped}"
        )
        self._raw_label.setText(text)
        # Red while clean, brighter alarm color if any event was ever dropped.
        color = "#ff5555" if dropped else "#ff3333"
        self._raw_label.setStyleSheet(
            f"padding: 0 8px; color: {color}; font-weight: bold;"
        )

    def clear_raw_stats(self):
        self._raw_label.setText("")
