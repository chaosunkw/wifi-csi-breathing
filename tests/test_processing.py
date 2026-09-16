import math
import unittest

import numpy as np

from wifi_csi_breathing.analysis import process_observations
from wifi_csi_breathing.data import Observation
from wifi_csi_breathing.processing import Processor, ProcessingConfig


def sine(seconds=90, hz=0.25, seed=9):
    rng = np.random.default_rng(seed)
    times = [0.0]
    while times[-1] < seconds:
        times.append(times[-1] + float(rng.uniform(0.025, 0.045)))
    return [Observation(t, 5 + math.sin(2 * math.pi * hz * t), index=i + 1)
            for i, t in enumerate(times)]


class ProcessingTests(unittest.TestCase):
    def test_jittered_sine_periodic_candidate(self):
        processor, points, estimates = process_observations(sine(), 'amplitude')
        self.assertGreater(len(estimates), 20)
        good = [e for e in estimates if e.candidate_bpm is not None]
        self.assertGreater(len(good), 10)
        self.assertAlmostEqual(good[-1].candidate_bpm, 15, delta=1.1)
        self.assertEqual(processor.resets, 0)
        self.assertGreaterEqual(estimates[0].time, 39.9)
        self.assertTrue(all(p.time <= p.available_at + 1e-9 for p in points))
        self.assertTrue(all(p.available_at - p.time < 0.046 for p in points))

    def test_prefix_is_invariant_to_future_data(self):
        observations = sine()
        _, full_points, full_estimates = process_observations(observations, 'amplitude')
        prefix = observations[:1800]
        _, points, estimates = process_observations(prefix, 'amplitude')
        cutoff = prefix[-1].time
        self.assertEqual(points, [p for p in full_points if p.available_at <= cutoff])
        self.assertEqual(estimates, [e for e in full_estimates if e.available_at <= cutoff])

    def test_flat_trace_never_claims_rate(self):
        values = [Observation(i / 30, 4.0) for i in range(2400)]
        _, _, estimates = process_observations(values, 'amplitude')
        self.assertTrue(estimates)
        self.assertTrue(all(e.candidate_bpm is None for e in estimates))
        self.assertEqual(estimates[-1].status, 'insufficient_variation')

    def test_gap_never_interpolates_or_carries_rate_across_boundary(self):
        processor = Processor()
        for o in sine(50):
            processor.feed(o)
        last = processor.last_input_time
        points, estimates = processor.feed(Observation(last + 2, 4))
        self.assertEqual(len(points), 1)
        self.assertAlmostEqual(points[0].time, last + 2)
        self.assertFalse(estimates)
        self.assertIsNone(processor.latest)
        self.assertEqual(len(processor.history), 1)
        self.assertEqual(processor.resets, 1)

    def test_missing_dimension_and_duplicate_reset(self):
        p = Processor()
        p.feed(Observation(0, 3))
        p.feed(Observation(0.03, None, True, 'dimension_change'))
        self.assertEqual(len(p.history), 0)
        points, _ = p.feed(Observation(0.06, 5, True, 'dimension_change'))
        self.assertEqual(points[0].segment, 2)
        points, _ = p.feed(Observation(0.06, 5))
        self.assertEqual(points, [])
        self.assertEqual(p.status, 'duplicate_timestamp')

    def test_phase_wrapping_is_causal_and_segment_local(self):
        times = np.arange(0, 50, 1 / 30)
        wrapped = [Observation(float(t), float(np.angle(np.exp(1j * (3 + 0.6 * math.sin(t * 1.5)))))) for t in times]
        _, points, _ = process_observations(wrapped, 'phase_difference')
        self.assertLess(max(abs(b.raw - a.raw) for a, b in zip(points, points[1:])), 0.1)
        self.assertGreater(min(p.raw for p in points), 2)

    def test_invalid_clock_or_settings_rejected(self):
        p = Processor()
        p.feed(Observation(1, 0))
        with self.assertRaises(ValueError):
            p.feed(Observation(0, 0))
        with self.assertRaises(ValueError):
            ProcessingConfig(sample_hz=0.5)
        with self.assertRaises(ValueError):
            ProcessingConfig(window_seconds=10)


if __name__ == '__main__':
    unittest.main()
