"""Qt replay viewer. Widgets stay on the UI thread; a worker handles all input."""
import queue
import sys
import time

from PySide6.QtCore import Qt, QTimer, QPointF
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (QApplication, QComboBox, QGridLayout, QHBoxLayout,
                              QLabel, QMainWindow, QPushButton, QSpinBox,
                              QVBoxLayout, QWidget)

from .data import FeatureConfig, MODES, load_catalog
from .replay import ReplayWorker


class SignalPlot(QWidget):
    def __init__(self, title, field, parent=None):
        super().__init__(parent)
        self.title, self.field = title, field
        self.points = ()
        self.setMinimumHeight(180)

    def set_points(self, points):
        self.points = points
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor('#0a1727'))
        painter.setPen(QColor('#d9e7f5'))
        painter.drawText(16, 25, self.title)
        left, right, top, bottom = 76, self.width() - 20, 48, self.height() - 34
        painter.setPen(QPen(QColor('#29425b'), 1))
        painter.drawRect(left, top, right - left, bottom - top)
        if not self.points:
            painter.setPen(QColor('#98adc2'))
            painter.drawText(left + 15, top + 35, 'Choose a recording and start replay')
            return
        values = [getattr(p, self.field) for p in self.points]
        low, high = min(values), max(values)
        margin = max((high - low) * 0.1, max(abs(low), abs(high), 1.0) * 0.001)
        low, high = low - margin, high + margin
        end = self.points[-1].time
        start = max(0.0, end - 60)
        span = max(end - start, 1.0)
        painter.setPen(QColor('#98adc2'))
        painter.drawText(4, top + 6, f'{high:.3g}')
        painter.drawText(4, bottom, f'{low:.3g}')
        painter.drawText(left, bottom + 23, f'{start:.1f} s')
        painter.drawText(right - 70, bottom + 23, f'{end:.1f} s')
        path = QPainterPath()
        previous_segment = None
        for point in self.points:
            x = left + (point.time - start) / span * (right - left)
            y = bottom - (getattr(point, self.field) - low) / (high - low) * (bottom - top)
            if point.segment != previous_segment:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))
            previous_segment = point.segment
        painter.setPen(QPen(QColor('#3dd5b7'), 1.5))
        painter.drawPath(path)


class MainWindow(QMainWindow):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.worker = None
        self.paused = False
        self.stopping = False
        self.last_update = None
        self.last_estimate = None
        self.setWindowTitle('Wi-Fi CSI Breathing — offline research preview')
        self.resize(1120, 850)
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(14)
        heading = QLabel('Wi-Fi CSI Breathing  ·  Replay lab')
        heading.setStyleSheet('font-size: 25px; font-weight: bold;')
        layout.addWidget(heading)
        note = QLabel('Recorded data only. Periodic candidates are unvalidated — not a person-presence or medical measurement.')
        note.setWordWrap(True)
        note.setStyleSheet('color: #f2c078;')
        layout.addWidget(note)
        self.dataset_box = QComboBox()
        for dataset in datasets:
            self.dataset_box.addItem(dataset.label)
        layout.addWidget(self.dataset_box)
        options = QGridLayout()
        self.mode_box = QComboBox()
        self.mode_box.addItems(MODES)
        self.tx, self.rx, self.ref, self.group = (QSpinBox() for _ in range(4))
        for spin in (self.tx, self.rx, self.ref):
            spin.setRange(1, 3)
        self.ref.setValue(2)
        self.group.setRange(1, 30)
        self.group.setValue(15)
        self.speed_box = QComboBox()
        self.speed_box.addItems(('1x', '2x', '4x', '8x'))
        for column, (label, control) in enumerate((('Feature', self.mode_box), ('TX', self.tx), ('RX', self.rx),
                                                  ('Reference RX', self.ref), ('Subcarrier group', self.group),
                                                  ('Replay speed', self.speed_box))):
            options.addWidget(QLabel(label), 0, column)
            options.addWidget(control, 1, column)
        layout.addLayout(options)
        buttons = QHBoxLayout()
        self.start_button = QPushButton('Start replay')
        self.pause_button = QPushButton('Pause')
        self.stop_button = QPushButton('Stop')
        for button in (self.start_button, self.pause_button, self.stop_button):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self.start_button.clicked.connect(self.start_replay)
        self.pause_button.clicked.connect(self.toggle_pause)
        self.stop_button.clicked.connect(self.stop_replay)
        self.status_label = QLabel('Ready — source files are read-only')
        self.status_label.setWordWrap(True)
        self.rate_label = QLabel('Periodic candidate: —')
        self.rate_label.setStyleSheet('font-size: 21px; font-weight: bold;')
        self.metrics_label = QLabel('About 40 continuous seconds are needed before the first trailing-window estimate.')
        self.metrics_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        layout.addWidget(self.rate_label)
        layout.addWidget(self.metrics_label)
        self.raw_plot = SignalPlot('Feature after past-only median / smoothing (relative units)', 'raw')
        self.filtered_plot = SignalPlot('Causal 0.1–0.5 Hz component (relative units; not chest displacement)', 'filtered')
        layout.addWidget(self.raw_plot, 1)
        layout.addWidget(self.filtered_plot, 1)
        self.detail_label = QLabel('Gaps and antenna-dimension changes reset the window. No live hardware is connected.')
        self.detail_label.setWordWrap(True)
        layout.addWidget(self.detail_label)
        self.controls = [self.dataset_box, self.mode_box, self.tx, self.rx, self.ref, self.group, self.speed_box]
        self._set_running(False)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.setStyleSheet('''
            QMainWindow, QWidget { background: #102136; color: #d9e7f5; font-size: 14px; }
            QComboBox, QSpinBox { background: #0a1727; border: 1px solid #35516d; padding: 7px; }
            QPushButton { background: #184d51; border: 1px solid #3dd5b7; padding: 10px; border-radius: 7px; }
            QPushButton:disabled { background: #273747; border-color: #35485a; color: #8091a2; }
        ''')

    def _set_running(self, running):
        self.start_button.setEnabled(not running)
        self.pause_button.setEnabled(running)
        self.stop_button.setEnabled(running)
        for control in self.controls:
            control.setEnabled(not running)

    def _clear_rate(self):
        self.last_estimate = None
        self.rate_label.setText('Periodic candidate: —')

    def start_replay(self):
        if self.worker is not None and self.worker.is_alive():
            return
        try:
            config = FeatureConfig(self.mode_box.currentText(), self.tx.value(), self.rx.value(),
                                   self.ref.value(), self.group.value())
            dataset = self.datasets[self.dataset_box.currentIndex()]
            self.worker = ReplayWorker(dataset, config, float(self.speed_box.currentText()[:-1]))
        except Exception as error:
            self.status_label.setText(f'Cannot start: {error}')
            return
        self.paused = False
        self.stopping = False
        self.pause_button.setText('Pause')
        self._clear_rate()
        self.last_update = None
        self.raw_plot.set_points(())
        self.filtered_plot.set_points(())
        self.metrics_label.setText('Waiting for a continuous window: 10 s settling + 30 s analysis.')
        self.detail_label.setText('Verifying the catalog hashes; labels are for display only.')
        self._set_running(True)
        self.worker.start()

    def toggle_pause(self):
        if self.worker is None or not self.worker.is_alive():
            return
        self.paused = not self.paused
        if self.paused:
            self.worker.pause()
            self.pause_button.setText('Resume')
            self.status_label.setText('Paused — historical waveform is frozen')
            self._clear_rate()
        else:
            self.worker.resume()
            self.pause_button.setText('Pause')
            self.status_label.setText('Resuming replay')
            self.last_update = time.monotonic()

    def stop_replay(self):
        if self.worker is not None:
            self.stopping = True
            self.worker.stop()
            self.stop_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.status_label.setText('Stopping replay…')
            self._clear_rate()

    def apply_message(self, message):
        if message.kind == 'frame' and (self.paused or self.stopping):
            return
        self.last_update = time.monotonic()
        self.status_label.setText(f'{message.status}  ·  recorded time {message.time:.1f} s')
        if message.points:
            self.raw_plot.set_points(message.points)
            self.filtered_plot.set_points(message.points)
        if message.kind == 'frame':
            estimate = message.estimate
            self.last_estimate = estimate
            if estimate is not None and estimate.candidate_bpm is not None:
                self.rate_label.setText(f'Periodic candidate: {estimate.candidate_bpm:.1f} cycles/min  ·  unvalidated')
            else:
                self.rate_label.setText('Periodic candidate: —')
            if estimate is None:
                self.metrics_label.setText('Waiting for a continuous window. Missing data or dimension changes restart warm-up.')
            else:
                self.metrics_label.setText(f'Trailing window at {estimate.time:.1f} s  ·  '
                                          f'ACF {estimate.autocorrelation:.2f}  ·  '
                                          f'spectral concentration {estimate.concentration:.2f}  ·  '
                                          f'band fraction {estimate.band_fraction:.2f}  (not confidence scores)')
            self.detail_label.setText(f'{message.processed} packets processed  ·  '
                                      f'{message.display_drops} superseded display updates  ·  '
                                      'No raw packets are dropped by the processor.')
        elif message.kind in ('eof', 'error', 'stopped'):
            self._clear_rate()
            self.metrics_label.setText('No current estimate. Any remaining curve is historical replay data.')
            self.detail_label.setText(message.detail or 'Replay stopped; source files were not edited.')
            self._set_running(False)
            self.paused = False
            self.stopping = False
            self.pause_button.setText('Pause')

    def poll(self):
        if self.worker is None:
            return
        while True:
            try:
                self.apply_message(self.worker.messages.get_nowait())
            except queue.Empty:
                break
        if (self.worker.is_alive() and not self.paused and self.last_update is not None
                and time.monotonic() - self.last_update > 2):
            self._clear_rate()
            self.status_label.setText('Waiting for replay data — no current estimate')

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.stop()
            self.worker.join(timeout=2)
        event.accept()


def launch(catalog):
    datasets = load_catalog(catalog)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    window = MainWindow(datasets)
    window.show()
    return app.exec()
