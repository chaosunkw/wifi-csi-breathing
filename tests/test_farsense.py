"""Equation checks, synthetic identifiability, missing data and window causality."""
from dataclasses import replace
import math
import unittest

import numpy as np
from scipy import signal

from test_rate_comparison import mixed_capture
from wifi_csi_breathing.baselines.farsense import (FarSenseConfig, analyze_window,
    complex_ratios, estimate_core, normalized_acf, projection_search, uniform_window)


def synthetic_ratios(hz=.25, seconds=90, fs=30):
    times = np.arange(round(seconds*fs))/fs
    # Thirty differently oriented channel-ratio trajectories, same target period.
    direction = np.exp(1j*np.linspace(0, 2*np.pi, 30, endpoint=False))
    ratios = (1.4+.6j) + .15*np.sin(2*np.pi*hz*times[:, None])*direction
    return times, ratios


class FarSenseTests(unittest.TestCase):
    def test_projection_and_bnr_match_direct_full_fft_for_every_angle(self):
        cfg = FarSenseConfig()
        rng = np.random.default_rng(7)
        t = np.arange(240)/cfg.sample_hz
        z = 3+4j + np.sin(2*np.pi*.25*t)*(1+2j) + .1*(rng.normal(size=240)+1j*rng.normal(size=240))
        y, best, scores, spectrum, frequencies = projection_search(z, cfg)
        z = z-z.mean()
        for index in range(100):
            theta = index*np.pi/50
            projected = z.real*np.cos(theta)+z.imag*np.sin(theta)
            full = np.abs(np.fft.fft(projected, cfg.nfft))**2
            freq = np.fft.fftfreq(cfg.nfft, 1/cfg.sample_hz)
            mask = (freq*60 >= 10) & (freq*60 <= 37)
            self.assertAlmostEqual(scores[index], full[mask].max()/full.sum(), places=13)
        np.testing.assert_allclose(y, z.real*np.cos(best*np.pi/50)+z.imag*np.sin(best*np.pi/50), atol=1e-12)
        np.testing.assert_allclose(spectrum, np.abs(np.fft.rfft(y, cfg.nfft))**2, atol=1e-9)
        self.assertAlmostEqual(scores[best], scores.max())

    def test_acf_equation_10_and_weighted_fusion_equation_11(self):
        y = np.array([.2, 1.3, -.4, 2.2, 3.1, .7])
        c = y-y.mean()
        expected = np.array([np.dot(c[k:], c[:len(c)-k])/np.dot(c, c) for k in range(len(c))])
        np.testing.assert_allclose(normalized_acf(y), expected, atol=1e-14)
        np.testing.assert_allclose(normalized_acf(-y), expected, atol=1e-14)
        t, z = synthetic_ratios(fs=20, seconds=12)
        result, a = estimate_core(z)
        np.testing.assert_allclose(a['fused_acf'], a['acfs'][:, a['included']] @ a['bnr'][a['included']])
        np.testing.assert_array_equal(a['included'], a['bnr'] > .7*a['bnr'].max())
        self.assertLess(abs(result['candidate_bpm']-15), .5)

    def test_complex_projection_detects_quadrature_motion_invisible_to_log_magnitude(self):
        t = np.arange(240)/20
        z = np.tile(np.exp(1j*.12*np.sin(2*np.pi*.4*t))[:, None], (1, 30))
        self.assertLess(np.std(np.log(np.abs(z))), 1e-14)
        result, arrays = estimate_core(z)
        self.assertLess(abs(result['candidate_bpm']-24), .6)
        self.assertGreater(np.std(arrays['patterns'][:, 0]), .01)
        self.assertEqual(len(result['selected_groups']), 30)

    def test_projection_bnr_can_select_second_harmonic_without_a_correction(self):
        # At 18/min the real projection's 36/min harmonic lies inside the
        # paper's search band and can outrank the fundamental's periodicity.
        # Retain this limitation instead of tuning the method to the reference.
        t = np.arange(240)/20
        z = np.tile(np.exp(1j*.12*np.sin(2*np.pi*.3*t))[:, None], (1, 30))
        result, _ = estimate_core(z)
        self.assertLess(abs(result['candidate_bpm']-36), 1)

    def test_ratio_cancels_common_complex_gain_without_averaging_streams(self):
        capture = mixed_capture(2)
        original = complex_ratios(capture)[1]
        for packet in capture.packets:
            for rx in (0, 1):
                packet['csi'][0][rx] = [v*(2+3j) for v in packet['csi'][0][rx]]
            for stream in packet['csi'][1:]:
                stream[0] = [100000j]*30
        _, ratios, audit = complex_ratios(capture)
        np.testing.assert_allclose(original, ratios)
        self.assertEqual({row['ntx'] for row in audit}, {1, 2, 3})
        capture.packets[10]['csi'][0][1][2] = 0j
        _, ratios, _ = complex_ratios(capture)
        self.assertTrue(np.isnan(ratios[10, 2]))
        self.assertTrue(np.isfinite(ratios[10, 3]))

    def test_fixed_window_uses_no_future_block_and_reports_interpolation_latency(self):
        times, z = synthetic_ratios()
        cfg = FarSenseConfig()
        full, arrays = analyze_window(times, z, 10, cfg)
        last = np.searchsorted(times, 10+12-1/cfg.sample_hz-1e-9)
        prefix, shorter = analyze_window(times[:last+1], z[:last+1], 10, cfg)
        self.assertEqual(full, prefix)
        for key in arrays:
            np.testing.assert_array_equal(arrays[key], shorter[key])
        changed = z.copy()
        changed[last+1:] = 1e9+1e9j
        again, _ = analyze_window(times, changed, 10, cfg)
        self.assertEqual(full, again)
        missing, _ = analyze_window(times[:last], z[:last], 10, cfg)
        self.assertEqual(missing['reason'], 'missing_right_endpoint')
        keep = ~((times > 21.8) & (times < 22.1))
        delayed, _ = analyze_window(times[keep], z[keep], 10, cfg)
        self.assertGreater(delayed['available_at_s'], 22)
        self.assertLessEqual(delayed['available_at_s'], 22.5)

    def test_gaps_and_invalid_groups_are_not_concatenated_or_invented(self):
        times, z = synthetic_ratios()
        short = ~((times > 15) & (times < 15.3))
        result, a = analyze_window(times[short], z[short], 10)
        self.assertIsNotNone(result['candidate_bpm'])
        self.assertGreater(result['bridged_grid_samples'], 0)
        np.testing.assert_allclose(np.diff(a['times']), .05, atol=1e-12)
        long = ~((times > 15) & (times < 16))
        result, a = analyze_window(times[long], z[long], 10)
        self.assertEqual(result['reason'], 'long_gap')
        self.assertIsNone(a)
        corrupt = z.copy()
        corrupt[450, 2] = np.nan
        result, a = analyze_window(times, corrupt, 10)
        self.assertEqual(result['valid_groups'], 29)
        self.assertNotIn(3, result['selected_groups'])
        corrupt[450, :] = np.nan
        result, a = analyze_window(times, corrupt, 10)
        self.assertEqual(result['reason'], 'no_complete_subcarrier')
        duplicate_t = np.insert(times, 450, times[450])
        result, a = analyze_window(duplicate_t, np.insert(z, 450, z[450], axis=0), 10)
        self.assertEqual(result['reason'], 'duplicate_timestamp')

    def test_first_peak_rule_is_not_global_maximum_and_flat_is_unavailable(self):
        result, arrays = estimate_core(np.ones((240, 30), complex))
        self.assertIsNone(result['candidate_bpm'])
        self.assertEqual(result['selected_groups'], [])
        t, z = synthetic_ratios(seconds=12, fs=20)
        result, a = estimate_core(z)
        peaks, _ = signal.find_peaks(a['fused_acf'])
        valid = [k for k in peaks if math.ceil(1200/37) <= k <= 120 and a['fused_acf'][k] > 0]
        self.assertEqual(result['lag_samples'], valid[0])

    def test_paper_sampling_configuration_and_parameter_validation(self):
        cfg = FarSenseConfig(sample_hz=100, nfft=8192, sg_samples=51)
        t, z = synthetic_ratios(seconds=12, fs=100)
        result, _ = estimate_core(z, cfg)
        self.assertLess(abs(result['candidate_bpm']-15), .5)
        for kwargs in ({'nfft': 100}, {'sample_hz': 0}, {'sg_samples': 10}, {'relative_bnr': 1}, {'angles': 1}):
            with self.assertRaises(ValueError):
                replace(FarSenseConfig(), **kwargs)


if __name__ == '__main__':
    unittest.main()
