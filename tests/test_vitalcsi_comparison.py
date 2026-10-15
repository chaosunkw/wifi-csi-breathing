"""Third-batch integration: previous methods, raw sources, labels and audits."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import tempfile
import unittest

import numpy as np

from test_farsense_comparison import manifest_file
from test_rate_comparison import minute_fixture
from wifi_csi_breathing.baselines.vitalcsi import amplitudes, analyze_interval
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import sha256
from wifi_csi_breathing.joint import JointCapture
from wifi_csi_breathing.vitalcsi_comparison import metrics_for


class VitalCSIComparisonTests(unittest.TestCase):
    def test_cli_preserves_sources_and_four_previous_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = minute_fixture(root/'source')
            before = {p.name: sha256(p) for p in source.iterdir()}
            manifest = manifest_file(root, source)
            output = root/'output'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['compare-vitalcsi', '--manifest', str(manifest), '--output', str(output)]), 0)
            report = json.loads((output/'report.json').read_text())
            previous = json.loads((output/'reference-methods/report.json').read_text())
            for key, value in previous['results'][0].items():
                self.assertEqual(report['results'][0][key], value)
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
            self.assertFalse(report['manual_events_used_in_estimation'])
            self.assertFalse(report['full_system_reproduction'])
            for artifact in report['artifacts']:
                self.assertEqual(sha256(output/artifact['path']), artifact['sha256'])
            arrays = np.load(output/'synthetic-10-70s/intermediates.npz', allow_pickle=False)
            self.assertEqual(arrays['filtered'].shape, (31, 600, 5))
            self.assertEqual(arrays['raw_power'].shape, (31, 301, 30))
            self.assertIn('Empty-room controls', (output/'index.html').read_text())
            capture = JointCapture(source)
            t, x, _ = amplitudes(capture)
            first, frames, _ = analyze_interval(t, x, 10, 70)
            capture.events = []
            t2, x2, _ = amplitudes(capture)
            second, frames2, _ = analyze_interval(t2, x2, 10, 70)
            self.assertEqual(first, second)
            self.assertEqual(frames, frames2)
            with self.assertRaises(FileExistsError):
                from wifi_csi_breathing.vitalcsi_comparison import compare_vitalcsi
                compare_vitalcsi(manifest, output)

    def test_metrics_keep_missing_controls_and_common_denominator(self):
        base = dict(empty_room=False, manual_reference_bpm=14., fft_bpm=14., acf_bpm=13.,
                    farsense_bpm=16., wiresp_bpm=15., vitalcsi_bpm=13.)
        rows = [base, {**base, 'vitalcsi_bpm': None}, {**base, 'empty_room': True, 'manual_reference_bpm': None}]
        metrics = metrics_for(rows)
        self.assertEqual(metrics[-1]['eligible_minutes'], 2)
        self.assertEqual(metrics[-1]['available_minutes'], 1)
        self.assertEqual(metrics[-1]['mae_available_bpm'], 1)
        self.assertTrue(all(m['common_minutes'] == 1 for m in metrics))
        self.assertTrue(all(m['empty_candidate_minutes'] == 1 for m in metrics))


if __name__ == '__main__':
    unittest.main()
