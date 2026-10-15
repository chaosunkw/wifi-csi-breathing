"""Numerical diagnostic exports must explain the same estimate, including failures."""
from dataclasses import asdict
from pathlib import Path
import csv
import json
import math
import tempfile
import unittest

from test_rate_comparison import signal_observations
from wifi_csi_breathing.processing import Point, inspect_window, estimate_window
from wifi_csi_breathing.rate_comparison import PROCESSING, estimate_minute, gap_processing
from wifi_csi_breathing.rate_diagnostics import export_diagnostics, prepare_diagnostics


class RateDiagnosticTests(unittest.TestCase):
    def test_known_rate_exports_full_arrays_that_explain_quality_values(self):
        result, points = estimate_minute(signal_observations())
        summary, selected, arrays = prepare_diagnostics(points, result, PROCESSING)
        self.assertEqual(summary['fft_peak_bpm'], 14)
        self.assertEqual(summary['accepted_candidate_bpm'], 14)
        self.assertEqual(summary['sample_count'], 1200)
        self.assertEqual(summary['fft_bin_spacing_per_min'], 1)
        self.assertEqual(len(arrays['power']), 601)
        self.assertEqual(len(arrays['ac']), 1200)
        self.assertEqual(arrays['ac'][0], 1)
        self.assertEqual(arrays['frequencies'][-1], 10)
        self.assertAlmostEqual(60/summary['acf_peak_lag_s'], summary['acf_peak_bpm'])
        self.assertTrue(all(c['passed'] for c in summary['checks']))
        self.assertEqual(estimate_window(selected, PROCESSING), inspect_window(selected, PROCESSING)[0])
        with tempfile.TemporaryDirectory() as root:
            exported, page = export_diagnostics(root, points, result, PROCESSING)
            self.assertEqual(exported, summary)
            self.assertIn('All four heuristic checks passed', page)
            def rows(name):
                with (Path(root)/name).open() as f:
                    return list(csv.DictReader(f))
            spectrum = rows('spectrum.csv')
            self.assertEqual(len(spectrum), 601)
            self.assertEqual(sum(r['selected_fft_peak'] == 'True' for r in spectrum), 1)
            peak = next(r for r in spectrum if r['selected_fft_peak'] == 'True')
            self.assertEqual(float(peak['rate_per_min']), 14)
            near = sum(float(r['filtered_power']) for r in spectrum if r['in_concentration_bins'] == 'True')
            band = sum(float(r['filtered_power']) for r in spectrum if r['in_search_band'] == 'True')
            self.assertAlmostEqual(near/band, result['concentration'])
            band_raw = sum(float(r['pre_bandpass_power']) for r in spectrum if r['in_search_band'] == 'True')
            total_raw = sum(float(r['pre_bandpass_power']) for r in spectrum if float(r['frequency_hz']) > 0)
            self.assertAlmostEqual(band_raw/total_raw, result['band_fraction'])
            self.assertAlmostEqual(sum(float(r['normalized_filtered_power']) for r in spectrum[1:]), 1)
            acf = rows('autocorrelation.csv')
            peak = next(r for r in acf if r['selected_acf_peak'] == 'True')
            self.assertNotEqual(int(peak['lag_samples']), 0)
            self.assertEqual(float(peak['normalized_acf']), result['autocorrelation'])
            inputs = rows('diagnostic-input.csv')
            self.assertEqual(len(inputs), 1200)
            self.assertEqual(float(inputs[0]['hann_weight']), 0)
            self.assertEqual(float(inputs[-1]['hann_weight']), 0)
            self.assertTrue((Path(root)/'estimator-diagnostics.png').is_file())

    def test_ambiguous_two_frequency_signal_keeps_peaks_but_rejects_rate(self):
        points = []
        for i in range(1200):
            t = 10+i/20
            value = math.sin(2*math.pi*.2*t) + math.sin(2*math.pi*.4*t)
            points.append(Point(t, t, value, value, 0))
        estimate = estimate_window(points, PROCESSING)
        result = {**asdict(estimate), 'start_s': 10, 'end_s': 70, 'reason': estimate.status}
        self.assertEqual(estimate.status, 'inconclusive')
        self.assertIsNone(estimate.candidate_bpm)
        summary, _, arrays = prepare_diagnostics(points, result, PROCESSING)
        self.assertTrue(summary['computed'])
        self.assertIsNotNone(summary['fft_peak_bpm'])
        self.assertIn('Spectral concentration', summary['failed_checks'])
        self.assertLess(estimate.concentration, .6)
        self.assertTrue(any(arrays['power'] > 0))

    def test_ineligible_minute_is_not_silently_concatenated_or_zero_filled(self):
        for observations in (
            [o for o in signal_observations() if not 30 < o.time < 32],
            [type(o)(o.time, -.4) for o in signal_observations()],
        ):
            result, points = estimate_minute(observations)
            summary, selected, arrays = prepare_diagnostics(points, result, PROCESSING)
            self.assertFalse(summary['computed'])
            self.assertEqual(selected, [])
            self.assertIsNone(arrays)
            with tempfile.TemporaryDirectory() as root:
                summary, page = export_diagnostics(root, points, result, PROCESSING)
                self.assertIn('not computed', page)
                with (Path(root)/'spectrum.csv').open() as f:
                    self.assertEqual(list(csv.DictReader(f)), [])
                self.assertEqual(json.loads((Path(root)/'estimator-diagnostics.json').read_text())['reason'], result['reason'])

    def test_short_gap_diagnostics_keep_internal_samples_and_report_interpolation(self):
        observations = [o for o in signal_observations() if not 37 < o.time < 37.3]
        result, points = estimate_minute(observations, gap_policy='preserve-short')
        summary, selected, arrays = prepare_diagnostics(points, result, gap_processing('preserve-short'))
        self.assertTrue(summary['internal_gap_interpolation_included'])
        self.assertEqual(summary['bridged_grid_samples'], result['bridged_grid_samples'])
        self.assertEqual(len(selected), 1200)
        self.assertTrue(any(37.01 < p.time < 37.29 for p in selected))
        self.assertEqual(len(arrays['filtered_detrended']), 1200)
        # Do not silently display diagnostics for a different window or result.
        with self.assertRaises(ValueError):
            prepare_diagnostics(points, {**result, 'spectral_peak_bpm': 99}, PROCESSING)
        with self.assertRaises(ValueError):
            prepare_diagnostics([p for p in points if not 37.01 < p.time < 37.29], result, PROCESSING)


if __name__ == '__main__':
    unittest.main()
