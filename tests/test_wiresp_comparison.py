"""End-to-end comparison, fixed baseline results, and reference independence."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import tempfile
import unittest

import numpy as np

from test_farsense_comparison import manifest_file
from test_rate_comparison import minute_fixture
from wifi_csi_breathing.baselines.wiresp import analyze_interval, normalized_amplitudes
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import sha256
from wifi_csi_breathing.joint import JointCapture
from wifi_csi_breathing.wiresp_comparison import metrics_for


class WiResPComparisonTests(unittest.TestCase):
    def test_cli_preserves_sources_and_three_previous_methods(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = minute_fixture(root/'source')
            before = {p.name: sha256(p) for p in source.iterdir()}
            manifest = manifest_file(root, source)
            output = root/'output'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['compare-wiresp', '--manifest', str(manifest), '--output', str(output)]), 0)
            report = json.loads((output/'report.json').read_text())
            ref = json.loads((output/'reference-methods/report.json').read_text())
            for key in ('fft_bpm', 'acf_bpm', 'farsense_bpm', 'manual_reference_bpm', 'baseline_accepted_bpm'):
                self.assertEqual(report['results'][0][key], ref['results'][0][key])
            # This fixture modulates every group identically: per-RX normalization
            # cancels it, demonstrating a limitation rather than fabricating a rate.
            self.assertIsNone(report['results'][0]['wiresp_bpm'])
            self.assertFalse(report['motion_gate_enabled'])
            self.assertFalse(report['full_system_reproduction'])
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
            for artifact in report['artifacts']:
                self.assertEqual(sha256(output/artifact['path']), artifact['sha256'])
            arrays = np.load(output/'synthetic-10-70s/intermediates.npz', allow_pickle=False)
            self.assertEqual(arrays['smoothed'].shape, (300, 46))
            self.assertIn('Empty-room controls', (output/'index.html').read_text())

    def test_manual_events_do_not_enter_wiresp(self):
        with tempfile.TemporaryDirectory() as temporary:
            capture = JointCapture(minute_fixture(Path(temporary)/'source'))
            t1, a1, _ = normalized_amplitudes(capture)
            first, _, _ = analyze_interval(t1, a1, 10, 70)
            capture.events = []
            t2, a2, _ = normalized_amplitudes(capture)
            second, _, _ = analyze_interval(t2, a2, 10, 70)
            np.testing.assert_array_equal(a1, a2)
            self.assertEqual(first, second)

    def test_missing_and_empty_results_remain_in_metrics(self):
        rows = [dict(empty_room=False, manual_reference_bpm=14., fft_bpm=14., acf_bpm=13., farsense_bpm=16., wiresp_bpm=None),
                dict(empty_room=False, manual_reference_bpm=12., fft_bpm=12., acf_bpm=12., farsense_bpm=14., wiresp_bpm=13.),
                dict(empty_room=True, manual_reference_bpm=None, fft_bpm=12., acf_bpm=10., farsense_bpm=13., wiresp_bpm=14.)]
        metrics = metrics_for(rows)
        self.assertEqual(metrics[3]['eligible_minutes'], 2)
        self.assertEqual(metrics[3]['available_minutes'], 1)
        self.assertEqual(metrics[3]['mae_available_bpm'], 1)
        self.assertTrue(all(m['common_minutes'] == 1 for m in metrics))
        self.assertTrue(all(m['empty_candidate_minutes'] == 1 for m in metrics))


if __name__ == '__main__':
    unittest.main()
