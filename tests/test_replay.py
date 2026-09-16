import queue
import threading
import time
import unittest
from unittest.mock import Mock, patch

from wifi_csi_breathing.data import FeatureConfig, Observation
from wifi_csi_breathing.replay import ReplayWorker
from wifi_csi_breathing.processing import Processor
from wifi_csi_breathing.analysis import process_observations
from test_processing import sine


class ReplayTests(unittest.TestCase):
    def test_replay_speed_does_not_change_offline_values(self):
        observations = sine(45)
        _, expected_points, expected_estimates = process_observations(observations, 'amplitude')
        for speed in (500, 2000):
            collected_points, collected_estimates = [], []
            class RecordingProcessor(Processor):
                def feed(self, observation):
                    points, estimates = super().feed(observation)
                    collected_points.extend(points)
                    collected_estimates.extend(estimates)
                    return points, estimates
            with patch('wifi_csi_breathing.replay.read_observations', return_value=(observations, {'packets': len(observations)})), \
                    patch('wifi_csi_breathing.replay.Processor', RecordingProcessor):
                worker = ReplayWorker(Mock(), FeatureConfig(), speed=speed)
                worker.start()
                worker.join(3)
                self.assertFalse(worker.is_alive())
            self.assertEqual(collected_points, expected_points)
            self.assertEqual(collected_estimates, expected_estimates)

    def test_paced_replay_eof_and_no_processing_drops(self):
        observations = [Observation(i / 1000, float(i)) for i in range(600)]
        dataset = Mock()
        with patch('wifi_csi_breathing.replay.read_observations', return_value=(observations, {'packets': len(observations)})):
            worker = ReplayWorker(dataset, FeatureConfig(), speed=2)
            started = time.monotonic()
            worker.start()
            worker.join(3)
            elapsed = time.monotonic() - started
        self.assertFalse(worker.is_alive())
        self.assertEqual(worker.processed, 600)
        self.assertGreaterEqual(elapsed, 0.28)
        self.assertLess(elapsed, 2)
        messages = []
        while not worker.messages.empty():
            messages.append(worker.messages.get_nowait())
        self.assertEqual(messages[-1].kind, 'eof')
        self.assertIsNone(messages[-1].estimate)
        self.assertGreater(worker.display_drops, 0)
        dataset.verify.assert_called_once()

    def test_pause_resume_preserves_position_and_stop_is_prompt(self):
        observations = [Observation(i / 100, float(i)) for i in range(500)]
        with patch('wifi_csi_breathing.replay.read_observations', return_value=(observations, {'packets': len(observations)})):
            worker = ReplayWorker(Mock(), FeatureConfig())
            worker.start()
            time.sleep(0.07)
            worker.pause()
            time.sleep(0.05)
            paused_count = worker.processed
            time.sleep(0.12)
            self.assertEqual(worker.processed, paused_count)
            worker.resume()
            time.sleep(0.07)
            self.assertGreater(worker.processed, paused_count)
            worker.pause()
            worker.stop()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            messages = list(worker.messages.queue)
            self.assertEqual(messages[-1].kind, 'stopped')

    def test_source_error_becomes_terminal_message(self):
        with patch('wifi_csi_breathing.replay.read_observations', side_effect=ValueError('SHA-256 mismatch')):
            worker = ReplayWorker(Mock(), FeatureConfig())
            worker.start()
            worker.join(1)
            messages = list(worker.messages.queue)
            self.assertEqual(messages[-1].kind, 'error')
            self.assertIn('SHA-256', messages[-1].detail)

    def test_stop_during_loading(self):
        def loading(dataset, config, event):
            from wifi_csi_breathing.data import Cancelled
            event.wait(1)
            raise Cancelled()
        with patch('wifi_csi_breathing.replay.read_observations', side_effect=loading):
            worker = ReplayWorker(Mock(), FeatureConfig())
            worker.start()
            worker.stop()
            worker.join(1)
            self.assertFalse(worker.is_alive())
            self.assertEqual(list(worker.messages.queue)[-1].kind, 'stopped')
