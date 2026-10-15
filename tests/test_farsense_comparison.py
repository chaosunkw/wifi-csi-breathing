"""Full comparison uses verified inputs and keeps missing/empty controls visible."""
from contextlib import redirect_stdout
from pathlib import Path
import io
import json
import tempfile
import unittest

from test_rate_comparison import minute_fixture
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import FeatureConfig, sha256
from wifi_csi_breathing.farsense_comparison import analyze_minute, summary_metrics, validate_manifest
from wifi_csi_breathing.baselines.farsense import FarSenseConfig
from wifi_csi_breathing.joint import JointCapture
from wifi_csi_breathing.rate_comparison import estimate_minute, mixed_rate_observations


def manifest_file(root, source):
    path = root/'manifest.json'
    path.write_text(json.dumps({'schema_version': 1, 'recordings': [
        {'key': 'synthetic', 'label': 'Synthetic', 'input': str(source), 'starts_s': [10], 'empty_room': False}]}))
    return path


class FarSenseComparisonTests(unittest.TestCase):
    def test_complete_cli_preserves_sources_and_existing_candidates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = minute_fixture(root/'source')
            before = {p.name: sha256(p) for p in source.iterdir()}
            manifest = manifest_file(root, source)
            output = root/'output'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['compare-farsense', '--manifest', str(manifest), '--output', str(output)]), 0)
            report = json.loads((output/'report.json').read_text())
            result = report['results'][0]
            capture = JointCapture(source)
            observations, _ = mixed_rate_observations(capture, FeatureConfig('log_ratio'), 'first-stream')
            original, _ = estimate_minute(observations, 10, 'preserve-short')
            self.assertEqual(result['fft_bpm'], original['spectral_peak_bpm'])
            self.assertEqual(result['acf_bpm'], original['autocorrelation_bpm'])
            self.assertEqual(result['baseline_accepted_bpm'], original['candidate_bpm'])
            self.assertEqual(result['farsense_valid_blocks'], 5)
            self.assertLess(abs(result['farsense_bpm']-14), 1)
            self.assertEqual(result['received_button_count'], 14)
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
            self.assertFalse(report['manual_events_used_in_estimation'])
            self.assertFalse(report['motion_gate_implemented'])
            self.assertIn('Empty-room controls', (output/'index.html').read_text())
            for artifact in report['artifacts']:
                self.assertEqual(artifact['sha256'], sha256(output/artifact['path']))
            self.assertEqual(len(list(output.glob('*/block-*/arrays.npz'))), 5)
            # Changing reference events must not change any numerical CSI result.
            first = analyze_minute(capture, 10, FarSenseConfig())[0]
            capture.events = []
            second = analyze_minute(capture, 10, FarSenseConfig())[0]
            for key in ('fft_bpm', 'acf_bpm', 'farsense_bpm', 'farsense_valid_blocks'):
                self.assertEqual(first[key], second[key])

    def test_incomplete_minute_not_reported_from_only_available_blocks(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = minute_fixture(Path(temporary)/'source', gap=(37, 38))
            c, _, _, _, blocks, _, _ = analyze_minute(JointCapture(source), 10, FarSenseConfig())
            self.assertLess(c['farsense_valid_blocks'], 5)
            self.assertIsNone(c['farsense_bpm'])
            self.assertIsNotNone(c['farsense_available_blocks_mean_bpm'])
            self.assertTrue(any(b['reason'] == 'long_gap' for b in blocks))

    def test_manifest_rejects_duplicate_and_unsafe_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = manifest_file(root, root/'source')
            original = json.loads(path.read_text())
            for key in ('../escape', 'a/b', ''):
                modified = json.loads(json.dumps(original))
                modified['recordings'][0]['key'] = key
                path.write_text(json.dumps(modified))
                with self.assertRaises(ValueError):
                    validate_manifest(path)
            original['recordings'][0]['starts_s'] = [10, 10]
            path.write_text(json.dumps(original))
            with self.assertRaises(ValueError):
                validate_manifest(path)

    def test_metrics_do_not_discard_unavailable_or_treat_empty_as_zero_truth(self):
        rows = [dict(empty_room=False, manual_reference_bpm=14., fft_bpm=14., acf_bpm=13., farsense_bpm=None),
                dict(empty_room=False, manual_reference_bpm=12., fft_bpm=12., acf_bpm=12., farsense_bpm=14.),
                dict(empty_room=True, manual_reference_bpm=None, fft_bpm=12., acf_bpm=10., farsense_bpm=13.)]
        metrics = summary_metrics(rows)
        self.assertEqual(metrics[0]['available_minutes'], 2)
        self.assertEqual(metrics[2]['available_minutes'], 1)
        self.assertEqual(metrics[2]['eligible_manual_minutes'], 2)
        self.assertEqual(metrics[2]['mae_vs_manual_available_bpm'], 2)
        self.assertTrue(all(m['common_minutes'] == 1 for m in metrics))
        self.assertTrue(all(m['empty_candidate_minutes'] == 1 for m in metrics))


if __name__ == '__main__':
    unittest.main()
