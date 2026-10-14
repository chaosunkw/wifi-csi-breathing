"""Synthetic configuration interruptions; no experimental recordings."""
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import math
import tempfile
import unittest

from test_decoder import record
from test_joint import fixture
from wifi_csi_breathing.analysis import process_observations
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import FeatureConfig, sha256
from wifi_csi_breathing.fixed_configuration import analyze_fixed, fixed_observations
from wifi_csi_breathing.joint import JointCapture
from wifi_csi_breathing.vendor.intel5300 import decode_bfee


def capture_rows(entries):
    origin = 10**17 + 123
    packets, rows = [], []
    for i, (seconds, ntx, rate) in enumerate(entries, 1):
        packet = decode_bfee(record(ntx, 3, value=lambda g,r,t: (int(30+5*math.sin(seconds)), 8))[3:]) if ntx else None
        if packet:
            packet['rate'] = rate
        packets.append(packet)
        rows.append({'record_index': i, 'host_monotonic_ns': origin + round(seconds*1e9)})
    return SimpleNamespace(t0=origin, packets=packets, index=rows, inside=lambda ns: origin <= ns < origin+10**12)


class FixedConfigurationTests(unittest.TestCase):
    def test_selects_exact_shape_and_rate_with_auditable_missing_records(self):
        c = capture_rows([(0,3,275),(.03,1,256),(.06,3,276),(.09,3,275)])
        obs, audit = fixed_observations(c, FeatureConfig(), 275)
        self.assertEqual([o.time for o in obs], [0,.09])
        self.assertEqual([o.index for o in obs], [1,4])
        self.assertEqual([a['status'] for a in audit], ['kept','other_shape','other_rate_flags','kept'])
        self.assertTrue(all(a['value'] is None for a in audit[1:3]))
        proc, points, _ = process_observations(obs, 'amplitude')
        self.assertEqual(proc.resets, 0)
        self.assertTrue(any(0 < p.time < .09 and p.available_at == .09 for p in points))

    def test_long_gap_is_not_hidden_by_many_unselected_packets(self):
        c = capture_rows([(0,3,275),(.05,3,275),(.1,1,256),(.15,3,276),(.2,1,256),(.25,3,276),(.3,3,275)])
        obs, _ = fixed_observations(c, FeatureConfig(), 275)
        proc, points, _ = process_observations(obs, 'amplitude')
        self.assertEqual(proc.resets, 1)
        self.assertFalse(any(.05+1e-9 < p.time < .3-1e-9 for p in points))
        self.assertNotEqual(points[0].segment, points[-1].segment)

    def test_duplicate_even_in_unselected_shape_and_invalid_match_are_barriers(self):
        for kind in ('duplicate','invalid','non_csi'):
            with self.subTest(kind=kind):
                c = capture_rows([(0,3,275),(.05,3,275),(.1,3,275)])
                if kind == 'duplicate':
                    c.index[1]['host_monotonic_ns'] = c.index[0]['host_monotonic_ns']
                    c.packets[1]['ntx'] = 1
                elif kind == 'invalid':
                    c.packets[1]['rssi'] = [0,0,0]
                else:
                    c.packets[1] = None
                obs, audit = fixed_observations(c, FeatureConfig(), 275)
                self.assertTrue(obs[1].boundary)
                self.assertIsNone(obs[1].value)
                self.assertNotEqual(audit[1]['status'], 'kept')
                proc, _, _ = process_observations(obs, 'amplitude')
                self.assertGreater(proc.resets, 0)

    def test_short_interruptions_allow_window_without_future_dependent_output(self):
        entries = [(i/20,1 if i%79==10 else 3,256 if i%79==10 else 276 if i%101==20 else 275) for i in range(1400)]
        c = capture_rows(entries)
        obs, _ = fixed_observations(c, FeatureConfig('log_ratio'), 275)
        _, full, estimates = process_observations(obs, 'log_ratio')
        self.assertGreater(len(estimates), 0)
        for count in (150,750,1050):
            short = capture_rows(entries[:count])
            prefix, _ = fixed_observations(short, FeatureConfig('log_ratio'), 275)
            _, points, estimates_prefix = process_observations(prefix, 'log_ratio')
            self.assertEqual([asdict(p) for p in points], [asdict(p) for p in full[:len(points)]])
            self.assertEqual([asdict(e) for e in estimates_prefix], [asdict(e) for e in estimates[:len(estimates_prefix)]])
            self.assertTrue(all(p.available_at >= p.time-1e-8 for p in points))

    def test_no_extrapolation_through_trailing_outage(self):
        c = capture_rows([(0,3,275),(.05,3,275),(.1,1,256),(.15,1,256)])
        obs, audit = fixed_observations(c, FeatureConfig(), 275)
        _, points, _ = process_observations(obs, 'amplitude')
        self.assertLessEqual(max(p.time for p in points), .05)
        self.assertEqual(len(audit), 4)

    def test_default_joint_still_resets_dimension_changes(self):
        with tempfile.TemporaryDirectory() as root:
            c = JointCapture(fixture(Path(root)/'input'))
            for packet in c.packets:
                packet['rate'] = 275
            c.packets[2]['ntx'] = 1
            strict = c.observations(FeatureConfig())
            fixed, _ = fixed_observations(c, FeatureConfig(), 275)
            self.assertIsNone(strict[1].value)
            self.assertTrue(strict[2].boundary)
            self.assertEqual(len(fixed), 2)
            self.assertFalse(any(o.boundary for o in fixed))

    def test_cli_exports_audits_estimates_and_preserves_input_hashes(self):
        with tempfile.TemporaryDirectory() as root:
            source = fixture(Path(root)/'input')
            before = {p.name:sha256(p) for p in source.iterdir()}
            output = Path(root)/'out'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['joint-fixed','--input',str(source),'--output',str(output),
                                      '--rate-flags','4660','--mode','amplitude']), 0)
            report = json.loads((output/'fixed-configuration-report.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['policy']['rate_flags'], 4660)
            self.assertEqual(report['results'][0]['selection_counts'], {'kept':3})
            self.assertTrue((output/'amplitude-estimates.csv').exists())
            for artifact in report['artifacts']:
                self.assertEqual(sha256(output/artifact['path']),artifact['sha256'])
            self.assertEqual(before,{p.name:sha256(p) for p in source.iterdir()})
            with self.assertRaises(FileExistsError): analyze_fixed(source, output, 4660, ['amplitude'])
            with self.assertRaises(ValueError): analyze_fixed(source, source/'out', 4660, ['amplitude'])

    def test_invalid_selection_clear_errors_and_no_completion_on_failure(self):
        c = capture_rows([(0,3,275)])
        for flag in (-1,65536,True,275.0):
            with self.assertRaises(ValueError): fixed_observations(c, FeatureConfig(), flag)
        with tempfile.TemporaryDirectory() as root:
            source = fixture(Path(root)/'input')
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(['joint-fixed','--input',str(source),'--output',root+'/none',
                                      '--rate-flags','275']),1)
            self.assertFalse(Path(root+'/none').exists())
            with patch('wifi_csi_breathing.fixed_configuration.save_figure',side_effect=OSError('Synthetic failure')):
                with self.assertRaises(OSError): analyze_fixed(source,Path(root)/'partial',4660,['amplitude'])
            self.assertFalse(Path(root+'/partial/fixed-configuration-report.json').exists())


if __name__ == '__main__':
    unittest.main()
