from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from test_data import make_dataset
from test_decoder import record
from wifi_csi_breathing.data import RAW_FILES, sha256
from wifi_csi_breathing.processing import Point, ProcessingConfig
from wifi_csi_breathing.static_analysis import (
    MODES, draw_figure, normalization, pilot_layout, segment_arrays,
    shared_limits, spectrum, static_analyze,
)


def make_pilot(root):
    datasets = []
    for index, (condition, trial) in enumerate((('Empty', 1), ('Empty', 3), ('F0', 1),
                                               ('F0', 2), ('R90', 1), ('R90', 2))):
        dataset = make_dataset(root / str(index), record(3, 3) + record(3, 3, 40000))
        metadata = dataset.verify()
        metadata.update(activityLabel='empty' if condition == 'Empty' else 'normal_breathing',
                        positionId='L01_' + ('F0' if condition == 'Empty' else condition), trialNumber=trial)
        (dataset.directory / 'metadata.json').write_text(json.dumps(metadata), encoding='utf-8')
        files = tuple({'name': name, 'bytes': (dataset.directory / name).stat().st_size,
                       'sha256': sha256(dataset.directory / name)} for name in RAW_FILES)
        datasets.append(replace(dataset, id=f'synthetic-{index}', files=files))
    return datasets


def points(start, seconds, segment=0, gain=1):
    return [Point(float(t), float(t), float(8 + gain * np.sin(2 * np.pi * 0.2 * t)),
                  float(gain * np.sin(2 * np.pi * 0.2 * t)), segment)
            for t in np.arange(start, start + seconds, 0.05)]


class StaticAnalysisTests(unittest.TestCase):
    def test_layout_uses_metadata_and_keeps_empty_trial_three(self):
        with tempfile.TemporaryDirectory() as directory:
            datasets = make_pilot(Path(directory))
            layout = pilot_layout(list(reversed(datasets)))
            self.assertEqual([(c, t) for c, t, _, _ in layout],
                             [('Empty', 1), ('Empty', 3), ('F0', 1), ('F0', 2), ('R90', 1), ('R90', 2)])
            with self.assertRaises(ValueError):
                pilot_layout(datasets[:-1])
            with self.assertRaises(ValueError):
                pilot_layout(datasets[:-1] + [datasets[0]])

    def test_clip_availability_and_segment_warmup(self):
        cfg = ProcessingConfig()
        values = points(0, 50) + points(55, 70, segment=1)
        values.append(Point(120, 120.02, 1, 1, 1))
        parts = segment_arrays(values, 120, cfg)
        self.assertEqual(len(parts), 2)
        self.assertTrue(all(part['time'].max() <= 120 for part in parts))
        self.assertAlmostEqual(parts[1]['time'][parts[1]['settled']][0], 65)
        self.assertFalse(any(part['time'].max() > 120 for part in parts))
        self.assertEqual(sum(len(part['time']) for part in parts), 2301)

    def test_psd_uses_raw_and_never_joins_short_segments(self):
        cfg = ProcessingConfig()
        parts = segment_arrays(points(0, 30) + points(40, 30, 1), 120, cfg)
        _, power, count = spectrum(parts, cfg)
        self.assertIsNone(power)
        self.assertEqual(count, 0)
        parts = segment_arrays(points(0, 120), 120, cfg)
        parts[0]['filtered'][:] = 0  # Spectrum must not depend on the bandpass output.
        frequency, power, count = spectrum(parts, cfg)
        self.assertEqual(count, 6)
        self.assertAlmostEqual(frequency[np.argmax(power)], 0.2)
        self.assertAlmostEqual(float(power.sum() * (frequency[1] - frequency[0])), 0.5, delta=0.02)

    def test_normalization_is_shape_only_and_handles_flat(self):
        cfg = ProcessingConfig()
        first = segment_arrays(points(0, 120), 120, cfg)
        second = segment_arrays(points(0, 120, gain=3), 120, cfg)
        mean, std = normalization(first)
        mean2, std2 = normalization(second)
        self.assertAlmostEqual(std2 / std, 3)
        np.testing.assert_allclose((first[0]['filtered'] - mean) / std,
                                   (second[0]['filtered'] - mean2) / std2, atol=1e-12)
        first[0]['filtered'][:] = 0
        self.assertIsNone(normalization(first)[1])
        self.assertEqual(shared_limits([], True), (0, 1))

    def test_shared_axes_and_disconnected_plot_segments(self):
        cfg = ProcessingConfig()
        parts = segment_arrays(points(0, 50) + points(60, 60, 1), 120, cfg)
        items = []
        for i in range(6):
            items.append({'segments': parts, 'mean': 0, 'std': 1, 'estimates': [],
                          'metrics': {'condition': ('Empty', 'F0', 'R90')[i // 2],
                                      'trial': i % 2 + 1, 'original_status': 'completed'}})
        fig = draw_figure(items, 'amplitude', 'filtered', 120, cfg,
                          {'tx': 1, 'rx': 1, 'reference_rx': 2, 'subcarrier': 15})
        self.assertEqual(len(fig.axes), 6)
        self.assertEqual(len({ax.get_ylim() for ax in fig.axes}), 1)
        for ax in fig.axes:
            self.assertEqual(len(ax.lines), 2)
            self.assertEqual(ax.get_xlim(), (0, 120))
            self.assertLess(ax.lines[0].get_xdata()[-1], ax.lines[1].get_xdata()[0])
            self.assertGreaterEqual(ax.lines[1].get_xdata()[0], 70 - 1e-8)
        fig.clear()

    def test_end_to_end_manifest_protection_and_failure_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            datasets = make_pilot(root)
            before = [dataset.files for dataset in datasets]
            with patch('wifi_csi_breathing.static_analysis.render_all', return_value=[]):
                report = static_analyze(datasets, root / 'success')
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['metrics']), 18)
            self.assertTrue((root / 'success' / 'index.html').is_file())
            self.assertTrue(report['implementation_hashes'])
            self.assertTrue(all(m['input_band_peak_cycles_min'] is None for m in report['metrics']))
            self.assertEqual([dataset.files for dataset in datasets], before)
            for dataset in datasets:
                dataset.verify()
            with self.assertRaises(FileExistsError):
                static_analyze(datasets, root / 'success')
            with self.assertRaises(ValueError):
                static_analyze(datasets, datasets[0].directory / 'out')
            with self.assertRaises(ValueError):
                static_analyze(datasets, root / 'bad', end=float('nan'))
            with patch('wifi_csi_breathing.static_analysis.render_all', side_effect=OSError('Synthetic render failure')):
                with self.assertRaises(OSError):
                    static_analyze(datasets, root / 'partial')
            self.assertFalse((root / 'partial' / 'static-report.json').exists())
            self.assertTrue((root / 'partial' / 'numerical' / 'report.json').exists())


if __name__ == '__main__':
    unittest.main()
