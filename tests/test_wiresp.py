"""Numerical contracts for the adapted WiResP pipeline, without real labels."""
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from wifi_csi_breathing.baselines.wiresp import (
    WiResPConfig, analyze_frame, analyze_interval, breathing_noise_ratio, covariance_acf,
    enhance_history, equalize_ecdf, first_peak, loess_matrix, normalized_amplitudes,
    otsu_threshold, pulse_check, resample_window)


def synthetic(seconds=90, bpm=15.):
    times = np.arange(round(seconds*20))/20
    sensitivity = np.tile(np.linspace(-1, 1, 30), 3)
    values = 1+.12*np.sin(2*np.pi*bpm/60*times[:, None])*sensitivity
    return times, values


class WiResPTests(unittest.TestCase):
    def test_normalization_and_first_stream_are_gain_invariant(self):
        _, values = synthetic(seconds=.2)
        packets = []
        for i in range(3):
            first = values[i].reshape(3, 30)*np.array([1, 3, 9])[:, None]
            packets.append({'ntx': i+1, 'nrx': 3, 'rate': 275,
                            'csi': [first.astype(complex)]+[np.ones((3, 30))*999 for _ in range(i)]})
        c = SimpleNamespace(index=[{'host_monotonic_ns': i*50_000_000, 'record_index': i+1} for i in range(3)],
                            packets=packets, t0=0, inside=lambda ns: True)
        _, amplitudes, audit = normalized_amplitudes(c)
        expected = values[:3].reshape(3, 3, 30)
        expected /= np.sqrt(np.mean(expected**2, axis=2))[:, :, None]
        np.testing.assert_allclose(amplitudes, expected.reshape(3, 90))
        self.assertEqual([r['ntx'] for r in audit], [1, 2, 3])
        packets[0]['csi'][0][1, 5] = 0
        _, amplitudes, audit = normalized_amplitudes(c)
        self.assertTrue(np.isnan(amplitudes[0, 30:60]).all())
        self.assertEqual(audit[0]['valid_channels'], 60)

    def test_equation5_matches_direct_overlap_covariance(self):
        x = np.random.default_rng(7).normal(size=(80, 3))
        centered = x-x.mean(axis=0)
        expected = np.array([(centered[:80-k]*centered[k:]).mean(axis=0)/np.mean(centered**2, axis=0) for k in range(80)])
        np.testing.assert_allclose(covariance_acf(x), expected, atol=1e-12)

    def test_bnr_matches_full_two_sided_fft_energy(self):
        cfg = WiResPConfig()
        _, x = synthetic(15)
        bnr, _, _ = breathing_noise_ratio(x, cfg)
        centered = (x-x.mean(axis=0))*np.hanning(len(x))[:, None]
        p = np.abs(np.fft.fft(centered, cfg.nfft, axis=0))**2
        hz = abs(np.fft.fftfreq(cfg.nfft, 1/cfg.sample_hz))
        band = (hz*60 >= cfg.low_bpm) & (hz*60 <= cfg.high_bpm)
        expected = p[band].sum(axis=0)/p[(hz > 0) & ~band].sum(axis=0)
        np.testing.assert_allclose(bnr, expected, rtol=1e-11)

    def test_weighted_acf_and_loess_preserve_polynomial(self):
        cfg = WiResPConfig()
        _, x = synthetic(15)
        result, arrays = analyze_frame(x, cfg)
        sel = arrays['selected_indices']
        np.testing.assert_allclose(arrays['combined_acf'], sum(arrays['acf'][:, i]*arrays['bnr'][i] for i in sel))
        self.assertEqual(len(result['selected_channels']), 10)
        t = np.arange(40, dtype=float)
        np.testing.assert_allclose(loess_matrix(40, 11) @ (2+3*t+.5*t*t), 2+3*t+.5*t*t, atol=1e-10)

    def test_ecdf_otsu_and_majority_follow_equations(self):
        x = np.array([[1., 1., 5.], [2., 5., 9.]])
        y = equalize_ecdf(x)
        np.testing.assert_allclose(y, [[2/6, 2/6, 5/6], [3/6, 5/6, 1.]])
        unique = np.unique(y)
        candidates = (unique[:-1]+unique[1:])/2
        scores = []
        for t in candidates:
            low, high = y[y < t], y[y >= t]
            scores.append(low.size*high.size/y.size**2*(low.mean()-high.mean())**2)
        self.assertEqual(otsu_threshold(y), candidates[np.argmax(scores)])
        self.assertIsNone(otsu_threshold(np.ones((10, 4))))
        cfg = WiResPConfig()
        lag = np.arange(300)/20
        image = np.cos(2*np.pi*lag/4)[:, None]*np.linspace(.4, 1, 30)
        check, arrays = enhance_history(image, cfg)
        np.testing.assert_array_equal(arrays['binary_trend'], arrays['binary'].mean(axis=1) >= .5)
        self.assertTrue(check['passed'])
        self.assertAlmostEqual(check['candidate_bpm'], 15)
        self.assertEqual(first_peak(image[:, -1], cfg), first_peak(arrays['enhanced'][:, -1], cfg))

    def test_constant_and_nonperiodic_trends_are_not_accepted(self):
        cfg = WiResPConfig()
        result, arrays = analyze_frame(np.ones((300, 90)), cfg)
        self.assertIsNone(arrays)
        self.assertEqual(result['reason'], 'insufficient_variation')
        for trend in (np.zeros(300, bool), np.ones(300, bool), np.arange(300) > 100):
            result, _ = pulse_check(trend, cfg)
            self.assertFalse(result['passed'])

    def test_short_gaps_use_real_time_and_long_gaps_stay_missing(self):
        cfg = WiResPConfig()
        times, x = synthetic(20)
        keep = ~((times > 7) & (times < 7.3))
        meta, grid, values = resample_window(times[keep], x[keep], 0, cfg)
        self.assertTrue(meta['ready'])
        self.assertGreater(meta['bridged_grid_fraction'], 0)
        self.assertEqual(meta['available_at_s'], 15)
        self.assertEqual(len(grid), 300)
        keep = ~((times > 7) & (times < 8))
        meta, _, values = resample_window(times[keep], x[keep], 0, cfg)
        self.assertFalse(meta['ready'])
        self.assertEqual(meta['reason'], 'long_gap')
        self.assertTrue(np.isnan(values).all())
        meta, _, _ = resample_window(times[:200], x[:200], 0, cfg)
        self.assertEqual(meta['reason'], 'missing_right_endpoint')

    def test_duplicate_and_invalid_channels_are_not_silently_filled(self):
        cfg = WiResPConfig()
        times, x = synthetic(20)
        times = np.insert(times, 101, times[100])
        x = np.insert(x, 101, x[100], axis=0)
        meta, _, _ = resample_window(times, x, 0, cfg)
        self.assertEqual(meta['reason'], 'duplicate_timestamp')
        times, x = synthetic(20)
        x[100, :30] = np.nan
        meta, _, values = resample_window(times, x, 0, cfg)
        self.assertEqual(meta['valid_channels'], 60)
        self.assertTrue(np.isnan(values[:, :30]).all())

    def test_rate_and_history_timing_prefix_invariance(self):
        cfg = WiResPConfig()
        times, x = synthetic()
        summary, frames, arrays = analyze_interval(times, x, 10, 70, cfg)
        self.assertEqual(summary['total_frames'], 46)
        self.assertEqual(summary['candidate_frames'], 46)
        self.assertLess(abs(summary['candidate_mean_bpm']-15), .6)
        first_check = next(f for f in frames if f['continuity_pass'] is not None)
        self.assertEqual(first_check['end_s'], 54)
        self.assertIsNone(frames[0]['continuity_pass'])
        self.assertIsNotNone(frames[0]['candidate_bpm'])
        self.assertGreater(summary['accepted_frames'], 0)
        self.assertFalse(summary['motion_gate_enabled'])
        for cutoff in (35, 55):
            prefix = times <= cutoff
            _, partial, partial_arrays = analyze_interval(times[prefix], x[prefix], 10, cutoff, cfg)
            self.assertEqual(partial, frames[:len(partial)])
            for a, b in zip(partial_arrays, arrays):
                np.testing.assert_array_equal(a['smoothed_acf'], b['smoothed_acf'])
        _, gated, _ = analyze_interval(times, x, 10, 70, replace(cfg, motion_threshold=-1.))
        self.assertTrue(all(f['accepted_bpm'] is None for f in gated))
        self.assertEqual([f['candidate_bpm'] for f in frames], [f['candidate_bpm'] for f in gated])

    def test_missing_frames_are_not_a_complete_minute(self):
        times, values = synthetic()
        keep = ~((times > 37) & (times < 38))
        summary, frames, _ = analyze_interval(times[keep], values[keep], 10, 70)
        self.assertIsNone(summary['candidate_mean_bpm'])
        self.assertIsNotNone(summary['available_candidates_mean_bpm'])
        self.assertTrue(any(f['reason'] == 'long_gap' for f in frames))
        self.assertTrue(any(f['continuity_reason'] == 'missing_acf_history' for f in frames))

    def test_configuration_validation(self):
        for settings in ({'top_k': 0}, {'top_k': 91}, {'median_samples': 4}, {'sample_hz': 0},
                         {'window_seconds': 2}, {'nfft': 32}, {'max_period_cv': -1},
                         {'hop_seconds': 1.3}, {'motion_threshold': float('nan')}):
            with self.assertRaises(ValueError):
                WiResPConfig(**settings)


if __name__ == '__main__':
    unittest.main()
