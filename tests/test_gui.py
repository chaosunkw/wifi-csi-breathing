import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import unittest
import time
import queue
from unittest.mock import Mock

from PySide6.QtWidgets import QApplication

from wifi_csi_breathing.data import Dataset
from wifi_csi_breathing.gui import MainWindow
from wifi_csi_breathing.processing import Estimate, Point
from wifi_csi_breathing.replay import ReplayMessage


class GuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.window = MainWindow([Dataset('fixture', Path('/tmp/unused-fixture'), (), 'Synthetic fixture only')])

    def tearDown(self):
        self.window.close()

    def frame(self):
        return ReplayMessage('frame', 60, 'periodic_candidate', (Point(60, 60, 1, 0, 0),),
                             Estimate(60, 60, 0, 'periodic_candidate', 14, 14, 14, 0.8, 0.9, 0.9))

    def test_terminal_states_clear_current_rate(self):
        for kind in ('eof', 'stopped', 'error'):
            self.window.apply_message(self.frame())
            self.assertIn('14.0', self.window.rate_label.text())
            self.window.apply_message(ReplayMessage(kind))
            self.assertNotIn('14.0', self.window.rate_label.text())
            self.assertIsNone(self.window.last_estimate)
            self.assertTrue(self.window.start_button.isEnabled())

    def test_paused_or_stopping_frames_do_not_restore_current_rate(self):
        self.window.paused = True
        self.window.apply_message(self.frame())
        self.assertNotIn('14.0', self.window.rate_label.text())
        self.window.paused = False
        self.window.stopping = True
        self.window.apply_message(self.frame())
        self.assertNotIn('14.0', self.window.rate_label.text())

    def test_offscreen_window_renders(self):
        self.window.show()
        self.app.processEvents()
        self.assertFalse(self.window.grab().isNull())
        self.assertFalse(self.window.pause_button.isEnabled())
        self.assertFalse(self.window.stop_button.isEnabled())

    def test_stale_replay_clears_current_rate(self):
        self.window.worker = Mock()
        self.window.worker.is_alive.return_value = True
        self.window.worker.messages = queue.Queue()
        self.window.apply_message(self.frame())
        self.window.last_update = time.monotonic() - 3
        self.window.poll()
        self.assertNotIn('14.0', self.window.rate_label.text())
        self.assertIn('no current estimate', self.window.status_label.text())
