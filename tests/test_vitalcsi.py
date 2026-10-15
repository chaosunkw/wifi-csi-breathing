"""Numerical checks for the independently implemented VitalCSI adaptation."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from wifi_csi_breathing.baselines.vitalcsi import (
    KalmanFusion, VitalCSIConfig, amplitudes, analyze_frame, analyze_interval,
    breath_onsets, power_spectrum, resample_window, select_components, spectral_purity)


def synthetic(seconds=90, bpm=15.):
    times = np.arange(round(seconds*20))/20
    values = 20+np.sin(2*np.pi*bpm/60*times[:, None])*np.linspace(.8, 1.2, 30)
    return times, values


class VitalCSITests(unittest.TestCase):
    def test_raw_magnitude_uses_first_stream_and_fixed_rx_without_normalizing(self):
        packets = [{'ntx': i+1, 'nrx': 3, 'rate': 273+i,
                    'csi': [np.full((3, 30), 3+4j)]+[np.full((3, 30), 999+0j)]*i} for i in range(3)]
        capture = SimpleNamespace(t0=0, packets=packets, inside=lambda ns: True,
            index=[{'host_monotonic_ns': i*50_000_000, 'record_index': i+1} for i in range(3)])
        _, x, audit = amplitudes(capture)
        np.testing.assert_array_equal(x, np.full((3, 30), 5.))
        self.assertEqual([a['ntx'] for a in audit], [1, 2, 3])
        packets[0]['csi'][0][0, 2] = 0
        packets[1]['csi'][0][0] *= 2
        _, x, _ = amplitudes(capture)
        self.assertTrue(np.isnan(x[0, 2]))
        np.testing.assert_array_equal(x[1], np.full(30, 10.))

    def test_pca_reconstructs_centered_input_and_variance(self):
        rng = np.random.default_rng(19)
        x = rng.normal(size=(600, 30))*np.linspace(1, 5, 30)+30
        a = select_components(x, VitalCSIConfig())
        np.testing.assert_allclose(a['pcs'] @ a['loadings'], x-x.mean(axis=0), atol=1e-12)
        np.testing.assert_allclose(a['variance'].sum(), np.var(x, axis=0, ddof=1).sum())
        np.testing.assert_allclose(a['pcs'].T @ a['pcs']/599, np.diag(a['variance']), atol=1e-12)
        self.assertLessEqual(len(a['selected_indices']), 5)
        self.assertTrue(np.all(a['cqi'][a['selected_indices']] > 0))

    def test_spectral_energy_and_moments_match_two_sided_definition(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(600, 2))
        hz, p = power_spectrum(x, 20)
        full = abs(np.fft.fft((x-x.mean(axis=0))*np.hanning(600)[:, None], axis=0))**2
        f = np.fft.fftfreq(600, 1/20)
        np.testing.assert_allclose(p.sum(axis=0), full.sum(axis=0))
        expected = (full*f[:, None]**2).sum(axis=0)**2/(full.sum(axis=0)*(full*f[:, None]**4).sum(axis=0))
        np.testing.assert_allclose(spectral_purity(hz, p), expected)
        _, a = synthetic(30)
        result = select_components(a, VitalCSIConfig())
        self.assertEqual(len(result['selected_indices']), 1)
        self.assertGreater(result['snr'][0], .99)

    def test_bssf_counts_known_cycles_and_ignores_flat_signal(self):
        cfg = VitalCSIConfig()
        for bpm in (6, 12, 15, 24):
            t, _ = synthetic(30)
            rate, onsets, _, boxes, _, decisions = breath_onsets(np.sin(2*np.pi*bpm/60*t), cfg)
            self.assertAlmostEqual(rate, bpm, places=8)
            self.assertTrue(np.all(np.diff(onsets)/cfg.sample_hz >= 60/cfg.high_bpm))
            self.assertEqual(len(onsets), sum(d['accepted'] for d in decisions))
            self.assertTrue(np.all(boxes >= 0))
        self.assertIsNone(breath_onsets(np.ones(600), cfg)[0])

    def test_constant_signal_does_not_manufacture_components(self):
        t, a = synthetic()
        frame, arrays = analyze_frame(t, np.ones_like(a)*20, 10)
        self.assertEqual(frame['reason'], 'no_positive_cqi_component')
        self.assertEqual(frame['selected_components'], 0)
        self.assertEqual(KalmanFusion().update(frame, arrays), (None, []))

    def test_real_time_gaps_and_invalid_groups_are_audited(self):
        cfg = VitalCSIConfig()
        t, x = synthetic(40)
        keep = ~((t > 7) & (t < 7.3))
        meta, grid, _ = resample_window(t[keep], x[keep], 0, cfg)
        self.assertTrue(meta['ready'])
        self.assertGreater(meta['bridged_grid_fraction'], 0)
        self.assertEqual(grid[-1], 29.95)
        self.assertEqual(meta['available_at_s'], 30)
        keep = ~((t > 7) & (t < 8))
        meta, _, y = resample_window(t[keep], x[keep], 0, cfg)
        self.assertEqual(meta['reason'], 'long_gap')
        self.assertTrue(np.isnan(y).all())
        x[70, 1] = np.nan
        meta, _, y = resample_window(t, x, 0, cfg)
        self.assertEqual(meta['valid_groups'], 29)
        self.assertTrue(np.isnan(y[:, 1]).all())
        duplicate = np.insert(t, 101, t[100])
        meta, _, _ = resample_window(duplicate, np.insert(x, 101, x[100], axis=0), 0, cfg)
        self.assertEqual(meta['reason'], 'duplicate_timestamp')
        meta, _, _ = resample_window(t[:300], x[:300], 0, cfg)
        self.assertEqual(meta['reason'], 'missing_right_endpoint')

    def test_kalman_fusion_matches_scalar_update_and_inverse_variance(self):
        cfg = VitalCSIConfig()
        fusion = KalmanFusion(cfg)
        arrays = {'loadings': np.eye(30)[:2], 'group_indices': np.arange(30)}
        frame = {'end_s': 30., 'components': [
            {'pc_index': 0, 'combined_bpm': 12., 'sqi': 1.},
            {'pc_index': 1, 'combined_bpm': 16., 'sqi': .5}]}
        first, details = fusion.update(frame, arrays)
        self.assertAlmostEqual(first, (4*12+16)/5)
        self.assertTrue(all(d['reinitialized'] for d in details))
        frame['end_s'] = 31.
        frame['components'][0]['combined_bpm'] = 14.
        frame['components'][1]['combined_bpm'] = 15.
        value, details = fusion.update(frame, arrays)
        gain = 5/9
        estimates = np.array([12+gain*2, 16-gain])
        # (2/1)^2 = (1/.5)^2, so the two fusion weights are equal.
        self.assertAlmostEqual(value, estimates.mean())
        self.assertTrue(all(abs(d['gain']-gain) < 1e-12 for d in details))
        frame['end_s'] = 32.
        arrays['loadings'] = np.eye(30)[2:4]
        _, details = fusion.update(frame, arrays)
        self.assertTrue(all(d['reinitialized'] for d in details))
        frame['components'] = []
        self.assertEqual(fusion.update(frame, None), (None, []))
        self.assertEqual(fusion.states, {})

    def test_known_rate_and_completed_window_prefix_invariance(self):
        t, x = synthetic()
        summary, frames, data = analyze_interval(t, x, 10, 70)
        self.assertEqual(summary['total_frames'], 31)
        self.assertEqual(summary['available_frames'], 31)
        self.assertLess(abs(summary['fused_mean_bpm']-15), 1)
        for cutoff in (45, 55):
            keep = t <= cutoff
            _, prefix, arrays = analyze_interval(t[keep], x[keep], 10, cutoff)
            self.assertEqual(prefix, frames[:len(prefix)])
            for a, b in zip(arrays, data):
                np.testing.assert_array_equal(a['filtered'], b['filtered'])
            self.assertTrue(all(f['available_at_s'] <= cutoff for f in prefix))

    def test_missing_frames_cannot_be_filled_by_kalman_prediction(self):
        t, x = synthetic()
        keep = ~((t > 37) & (t < 38))
        summary, frames, _ = analyze_interval(t[keep], x[keep], 10, 70)
        self.assertIsNone(summary['fused_mean_bpm'])
        self.assertIsNotNone(summary['available_only_mean_bpm'])
        missing = [f for f in frames if f['reason'] == 'long_gap']
        self.assertTrue(missing)
        self.assertTrue(all(f['fused_bpm'] is None and f['fusion'] == [] for f in missing))

    def test_invalid_settings(self):
        for options in ({'rx': 0}, {'top_k': 31}, {'window_seconds': 1}, {'sample_hz': 0},
                        {'bssf_box_seconds': 2}, {'residual_floor_bpm': 0}, {'loading_similarity': 2},
                        {'process_variance_per_second': -1}, {'measurement_variance': float('nan')}):
            with self.assertRaises(ValueError):
                replace(VitalCSIConfig(), **options)


if __name__ == '__main__':
    unittest.main()
