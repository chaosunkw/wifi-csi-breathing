"""Synthetic-only all-group diagnostics; never depend on participant recordings."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch
import io
import math
import tempfile
import unittest

import numpy as np

from test_batch_diagnostics import six_sources
from test_decoder import record
from test_joint import fixture
from wifi_csi_breathing.batch_diagnostics import archive_batch, diagnostic_feature, load_archive
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import FeatureConfig, sha256
from wifi_csi_breathing.joint import JointCapture
from wifi_csi_breathing.subcarriers import (
    all_features, analyze_subcarriers, centered_features, extract, heat_values,
    heatmap_figure, limits_for, line_arrays, overview_figure, packet_cells, trace_figure,
    waterfall_figure,
)
from wifi_csi_breathing.vendor.intel5300 import decode_bfee


def small_data(values=None):
    raw = np.tile(np.arange(30, dtype=float), (5, 1)) if values is None else values
    return dict(times=np.asarray([0., .03, .06, .09, .8]),
                shapes=np.asarray([[3, 3], [3, 3], [1, 3], [3, 3], [3, 3]]),
                segments=np.asarray([1, 1, 2, 3, 4]), raw=raw, mode='amplitude',
                markers=np.asarray([0., .25]))


class SubcarrierTests(unittest.TestCase):
    def test_all_thirty_groups_match_existing_features_for_each_shape(self):
        for ntx in (1, 2, 3):
            packet = decode_bfee(record(ntx, 3)[3:])
            for mode in ('amplitude', 'log_ratio', 'phase_difference'):
                cfg = FeatureConfig(mode)
                result = all_features(packet, cfg)
                self.assertEqual(result.shape, (30,))
                for k in range(1, 31):
                    self.assertAlmostEqual(result[k - 1], diagnostic_feature(packet, FeatureConfig(mode, subcarrier=k)))

    def test_invalid_inputs_produce_nan_and_common_phase_cancels(self):
        packet = decode_bfee(record(1, 3)[3:])
        self.assertTrue(np.isnan(all_features(packet, FeatureConfig(tx=2))).all())
        phase = all_features(packet, FeatureConfig('phase_difference'))
        for rx in packet['csi'][0]:
            for k in range(30):
                rx[k] *= np.exp(1j * .8)
        np.testing.assert_allclose(all_features(packet, FeatureConfig('phase_difference')), phase, atol=1e-12)
        packet['csi'][0][1][0] = 0j
        self.assertTrue(np.isnan(all_features(packet, FeatureConfig('phase_difference'))[0]))
        self.assertTrue(np.isnan(all_features(packet, FeatureConfig('log_ratio'))[0]))
        packet['rssi'] = [0, 0, 0]
        self.assertTrue(np.isnan(all_features(packet, FeatureConfig())).all())

    def test_median_centering_is_group_and_shape_local_not_standardized(self):
        d = small_data()
        d['raw'][0] += 1
        d['raw'][1] += 3
        d['raw'][2] += 100
        d['raw'][3] += 5
        d['raw'][4] += 7
        centered, centers, _ = centered_features(d, 'amplitude')
        np.testing.assert_allclose(centered[:, 0], [-3, -1, 0, 1, 3])
        np.testing.assert_allclose(centers[2], np.arange(30) + 4)
        np.testing.assert_allclose(centers[0], np.arange(30) + 100)
        self.assertTrue(np.isnan(centers[1]).all())
        d['raw'][:] = 7
        result, _, _ = centered_features(d, 'amplitude')
        np.testing.assert_allclose(result, 0)

    def test_circular_center_handles_branch_cut_and_undefined_center(self):
        d = small_data(np.tile(np.asarray([math.pi - .1, -math.pi + .1, .2, math.pi - .1, -math.pi + .1])[:, None], (1, 30)))
        values, centers, resultant = centered_features(d, 'phase_difference')
        np.testing.assert_allclose(values[[0, 1, 3, 4], 0], [-.1, .1, -.1, .1], atol=1e-12)
        self.assertAlmostEqual(abs(centers[2, 0]), math.pi)
        self.assertGreater(resultant[2, 0], .99)
        d['raw'][[0, 1, 3, 4]] = np.asarray([0, math.pi, 0, math.pi])[:, None]
        values, centers, _ = centered_features(d, 'phase_difference')
        self.assertTrue(np.isnan(values[[0, 1, 3, 4]]).all())
        self.assertTrue(np.isnan(centers[2]).all())

    def test_extract_keeps_half_open_integer_host_time_and_boundaries(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = JointCapture(fixture(Path(temp) / 'one'))
            capture.packets[2]['rate'] += 1
            d = extract(capture, FeatureConfig(), max_gap=.08)
            np.testing.assert_allclose(d['times'], [0, .100000001, .15])
            np.testing.assert_equal(d['segments'], [1, 2, 3])
            np.testing.assert_allclose(d['markers'], [0, .250000001])
            self.assertEqual(d['raw'].shape, (3, 30))
            capture.packets[2]['rate'] = capture.packets[1]['rate']
            d = extract(capture, FeatureConfig(), max_gap=.2)
            np.testing.assert_equal(d['segments'], [1, 1, 1])
            capture.packets[2] = decode_bfee(record(1, 3)[3:])
            d = extract(capture, FeatureConfig())
            np.testing.assert_equal(d['segments'], [1, 2, 3])

    def test_cells_preserve_gaps_neighbors_shapes_and_duplicate_time(self):
        d = small_data()
        edges, owner = packet_cells(d['times'], 1, .05)
        mids = (edges[:-1] + edges[1:]) / 2
        self.assertTrue(np.all(owner[(mids > .2) & (mids < .7)] == -1))
        self.assertTrue(np.all(np.diff(edges)[owner >= 0] <= .05 + 1e-12))
        matrix = heat_values(d, owner, 1, 'raw')
        self.assertTrue(np.ma.getmaskarray(matrix)[:, owner != 2].all())
        self.assertFalse(np.ma.getmaskarray(matrix)[:, owner == 2].any())
        edges, owner = packet_cells(np.array([.1, .1, .2]), 1)
        self.assertFalse(np.any(np.isin(owner, [0, 1])))

    def test_line_breaks_keep_first_points_and_wraps_without_interpolation(self):
        d = small_data()
        d['centered'] = d['raw'].copy()
        times, vals = line_arrays(d, 3, 'centered')
        np.testing.assert_allclose(times[np.isfinite(times)], d['times'])
        finite = np.isfinite(vals[:, 0])
        np.testing.assert_allclose(times[finite], [0, .03, .09, .8])
        # No adjacent finite samples may straddle a known segment boundary.
        for k in range(len(times) - 1):
            if finite[k] and finite[k + 1]:
                self.assertLessEqual(times[k + 1] - times[k], .03 + 1e-12)
        d['mode'] = 'phase_difference'
        d['centered'][0] = 3
        d['centered'][1] = -3
        times, vals = line_arrays(d, 3, 'centered')
        self.assertTrue(np.isnan(times[1]))
        self.assertEqual(vals[0, 0], 3)
        self.assertEqual(vals[2, 0], -3)

    def test_color_limits_and_saturation_do_not_modify_data(self):
        d = small_data()
        d['centered'] = d['raw'].copy()
        d['centered'][-1, -1] = 10000
        before = d['centered'].copy()
        limits = limits_for([d], 'amplitude', 99)
        self.assertLess(limits['centered_by_shape'][2][1], 10000)
        np.testing.assert_equal(d['centered'], before)
        self.assertEqual(limits_for([d], 'phase_difference', 99)['raw'], [-math.pi, math.pi])

    def test_figures_have_thirty_groups_and_fixed_markers(self):
        d = small_data()
        d['centered'], d['centers'], d['phase_resultant'] = centered_features(d, 'amplitude')
        d['edges'], d['owner'] = packet_cells(d['times'], 1)
        d.update(config=FeatureConfig(), label='Synthetic / Trial 1', center_label='Median-centered feature')
        limits = limits_for([d], 'amplitude', 99)
        fig = heatmap_figure(d, limits, 1)
        feature_axes = [a for a in fig.axes if a.get_ylabel().endswith('Group')]
        self.assertEqual(len(feature_axes), 6)
        self.assertTrue(all(a.collections[0].get_array().shape[0] == 30 for a in feature_axes))
        for ax in feature_axes:
            mesh = ax.collections[0]
            self.assertFalse(np.allclose(mesh.cmap.get_bad(), mesh.cmap(.5)))
            np.testing.assert_allclose(ax.get_facecolor(), mesh.cmap.get_bad())
        for a in fig.axes:
            if a.get_ylabel() == 'Marker':
                np.testing.assert_allclose([s[0, 0] for s in a.collections[0].get_segments()], [0, .25])
        fig.canvas.draw()
        fig.clear()
        fig = waterfall_figure(d, 1)
        self.assertEqual(len(fig.axes), 3)
        self.assertTrue(all(len(a.lines) == 30 for a in fig.axes))
        fig.canvas.draw()
        fig.clear()
        fig = trace_figure(d, limits, 1, [1, 15, 30])
        self.assertEqual(len(fig.axes), 12)
        self.assertEqual([a.get_title() for a in fig.axes[:3]], ['Group 1', 'Group 15', 'Group 30'])
        fig.canvas.draw()
        fig.clear()

    def test_heatmaps_align_marker_time_coordinates_after_rendering(self):
        for mode in ('amplitude', 'phase_difference'):
            d = small_data()
            d['mode'] = mode
            d['centered'], d['centers'], d['phase_resultant'] = centered_features(d, mode)
            d['edges'], d['owner'] = packet_cells(d['times'], 1)
            d.update(config=FeatureConfig(mode), label='Synthetic / Trial 1',
                     center_label='Synthetic centered feature')
            limits = limits_for([d], mode, 99)
            for kind in ('session', 'overview'):
                with self.subTest(mode=mode, figure=kind):
                    fig = (heatmap_figure(d, limits, 1) if kind == 'session'
                           else overview_figure([d] * 6, limits, 1, mode))
                    try:
                        fig.canvas.draw()
                        features = [a for a in fig.axes if a.get_ylabel().endswith('Group')]
                        markers = [a for a in fig.axes if a.get_ylabel() == 'Marker']
                        panels = ([features[col::2] for col in range(2)] if kind == 'session'
                                  else [features[i:i + 3] for i in range(0, 18, 3)])
                        self.assertEqual(len(markers), len(panels))
                        coordinates = np.column_stack(([0., .25, .75, 1.], np.zeros(4)))
                        for marker, panel in zip(markers, panels):
                            self.assertEqual(len(panel), 3)
                            expected_x = marker.transData.transform(coordinates)[:, 0]
                            for ax in panel:
                                np.testing.assert_allclose(
                                    ax.transData.transform(coordinates)[:, 0], expected_x,
                                    rtol=0, atol=1e-7,
                                    err_msg='Equal times must align vertically with manual markers')
                    finally:
                        fig.clear()

    def test_end_to_end_exports_and_hashes_without_mutating_archive(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, corrections = six_sources(root)
            archive = root / 'archive'
            archive_batch(source, 'SYNTHETIC', corrections, archive)
            original_hash = sha256(archive / 'archive.json')
            output = root / 'result'
            # Real rendering exercises PNG/PDF/SVG and array exports on tiny fixtures.
            report = analyze_subcarriers(archive, output, ['phase_difference'])
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['summaries']), 6)
            self.assertEqual([s['markers'] for s in report['summaries']], [0, 0, 2, 2, 2, 2])
            for entry in report['artifacts']:
                self.assertEqual(sha256(output / entry['path']), entry['sha256'])
            for summary in report['summaries']:
                self.assertEqual(summary['feature']['subcarrier_groups'], list(range(1, 31)))
                self.assertNotIn('subcarrier', summary['feature'])
                with np.load(output / summary['samples'], allow_pickle=False) as data:
                    self.assertEqual(data['raw'].shape, (3, 30))
                    np.testing.assert_allclose(data['times'], [0, .100000001, .15])
            self.assertEqual(sha256(archive / 'archive.json'), original_hash)
            load_archive(archive)
            with self.assertRaises(FileExistsError):
                analyze_subcarriers(archive, output, ['amplitude'])
            with self.assertRaises(ValueError):
                analyze_subcarriers(archive, archive / 'bad')
            with patch('wifi_csi_breathing.subcarriers.save_figure', side_effect=OSError('Synthetic failure')):
                with self.assertRaises(OSError):
                    analyze_subcarriers(archive, root / 'partial', ['amplitude'])
            self.assertFalse((root / 'partial' / 'subcarrier-report.json').exists())
            with patch('wifi_csi_breathing.subcarriers.save_figure'), \
                    patch('wifi_csi_breathing.subcarriers.implementation_hashes', side_effect=[[], [{'sha256': 'changed'}]]):
                with self.assertRaisesRegex(ValueError, 'Implementation changed'):
                    analyze_subcarriers(archive, root / 'changed-code', ['amplitude'])
            self.assertFalse((root / 'changed-code' / 'subcarrier-report.json').exists())

    def test_validation_and_cli_fail_before_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for options in ({'max_gap': 0}, {'max_gap': float('nan')}, {'cell_seconds': .4},
                            {'color_percentile': 101}, {'groups': [0]}, {'groups': [1, 1]},
                            {'modes': []}, {'modes': ['amplitude', 'amplitude']}, {'rx': 4}):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    analyze_subcarriers(root / 'missing', root / 'out', **options)
                self.assertFalse((root / 'out').exists())
            with redirect_stderr(io.StringIO()) as errors, redirect_stdout(io.StringIO()):
                status = main(['subcarriers', '--archive', str(root / 'missing'), '--output', str(root / 'out')])
            self.assertEqual(status, 1)
            self.assertIn('Error:', errors.getvalue())


if __name__ == '__main__':
    unittest.main()
