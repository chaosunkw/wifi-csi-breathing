"""One-minute comparisons using synthetic signals and transfer receipts only."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import json
import math
import tempfile
import unittest

from test_decoder import record
from test_joint import (SESSION, T0, fixture, refresh_receipt, save_json, save_table)
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import FeatureConfig, Observation, sha256
from wifi_csi_breathing.joint import EVENT_FIELDS, INDEX_FIELDS
from wifi_csi_breathing.rate_comparison import (analyze_rate, compare_reference,
    estimate_minute, manual_reference, mixed_rate_observations, window_bounds)


def signal_observations(seconds=90, hz=14/60, offset=0.0):
    return [Observation(offset+i/30, -.4 + .15*math.sin(2*math.pi*hz*(offset+i/30)), index=i+1)
            for i in range(round(seconds*30))]


def events_capture(times):
    events = [dict(zip(EVENT_FIELDS, (T0+round(t*1e9), 1, '0011223344556677',
              'aabbccddeeff0011', i, i*100, 'inhalation_onset_manual', 'ok', 0)))
              for i, t in enumerate(times, 1)]
    return SimpleNamespace(t0=T0, events=events)


def mixed_capture(seconds=90):
    observations = signal_observations(seconds)
    packets, index = [], []
    for i, observation in enumerate(observations):
        ntx = 1 if i % 79 == 10 else 2 if i % 113 == 11 else 3
        packet = {'ntx': ntx, 'nrx': 3, 'rate': 273+i % 5,
                  'csi': [[[complex(math.exp(observation.value))]*30, [1+0j]*30, [1+0j]*30]
                          for _ in range(ntx)]}
        packets.append(packet)
        index.append({'record_index': i+1, 'host_monotonic_ns': T0+round(observation.time*1e9)})
    return SimpleNamespace(t0=T0, packets=packets, index=index,
                           inside=lambda ns: T0 <= ns < T0+round(seconds*1e9))


def minute_fixture(folder):
    """Synthetic binary packets exercise the full parser/receipt/CLI path."""
    fixture(folder)
    rows, blobs, offset = [], [], 0
    for i in range(2700):
        seconds = i/30
        ntx = 1 if i % 79 == 10 else 2 if i % 113 == 11 else 3
        amplitude = round(45 * math.exp(.18*math.sin(2*math.pi*14/60*seconds)))
        blob = bytearray(record(ntx, 3, timestamp=i,
                               value=lambda g, r, t: (amplitude if r == 0 else 45, 0)))
        blob[21:23] = (273+i % 5).to_bytes(2, 'little')
        blobs.append(blob)
        rows.append(dict(zip(INDEX_FIELDS, (i+1, offset, len(blob), T0+round(seconds*1e9), 187, i))))
        offset += len(blob)
    (folder/'csi.dat').write_bytes(b''.join(blobs))
    save_table(folder/'csi-times.csv', INDEX_FIELDS, rows)
    timing = json.loads((folder/'timing.json').read_text())
    timing['t1_ns'] = str(T0+90_000_000_000)
    save_json(folder/'timing.json', timing)
    events = events_capture([6] + [10+k*60/14 for k in range(14)] + [70, 80]).events
    save_table(folder/'marker-events.csv', EVENT_FIELDS, events)
    (folder/'marker-records.jsonl').write_text(''.join(
        json.dumps({**e, 'accepted': True, 'simulation': False})+'\n' for e in events))
    save_json(folder/'marker-status.json', {'schemaVersion': 1, 'sessionId': SESSION,
        'status': 'finished', 'error': None, 'markerReady': True, 'csiReady': True,
        'accepted': len(events), 'rejected': 0, 'missing': 0, 'csiRecords': len(rows)})
    refresh_receipt(folder)
    return folder


class RateComparisonTests(unittest.TestCase):
    def test_known_fourteen_per_minute_and_manual_count(self):
        result, points = estimate_minute(signal_observations())
        self.assertEqual(result['status'], 'periodic_candidate')
        self.assertAlmostEqual(result['candidate_bpm'], 14)
        self.assertAlmostEqual(result['autocorrelation_bpm'], 14, delta=.5)
        self.assertEqual(result['settled_grid_samples'], 1200)
        self.assertEqual(result['supported_grid_fraction'], 1.0)
        self.assertTrue(all(p.available_at >= p.time-1e-8 for p in points))
        # Deliberately different manual count proves the CSI estimate is not fitted.
        manual = manual_reference(events_capture([11+k*4 for k in range(15)]))
        comparison = compare_reference(result, manual)
        self.assertEqual(comparison['manual_reference_bpm'], 15)
        self.assertAlmostEqual(comparison['absolute_error_bpm'], 1)
        self.assertEqual(compare_reference(result, manual_reference(events_capture([])))['candidate_bpm'], 14)

    def test_mixed_rates_are_kept_and_short_shape_gaps_do_not_reset(self):
        capture = mixed_capture()
        observations, audit = mixed_rate_observations(capture, FeatureConfig('log_ratio'))
        self.assertEqual({a['rate_flags'] for a in audit if a['status'] == 'kept'}, {273,274,275,276,277})
        self.assertTrue(any(a['ntx'] == 1 for a in audit))
        self.assertTrue(any(a['ntx'] == 2 for a in audit))
        self.assertTrue(all(a['value'] is None for a in audit if a['status'] == 'other_shape'))
        result, _ = estimate_minute(observations)
        self.assertEqual(result['filter_resets'], 0)
        self.assertAlmostEqual(result['candidate_bpm'], 14)

    def test_duplicate_unknown_and_invalid_packets_remain_barriers(self):
        for kind in ('duplicate', 'non_csi', 'invalid'):
            with self.subTest(kind=kind):
                capture = mixed_capture(1)
                if kind == 'duplicate':
                    capture.index[10]['host_monotonic_ns'] = capture.index[9]['host_monotonic_ns']
                elif kind == 'non_csi':
                    capture.packets[10] = None
                else:
                    capture.packets[9]['csi'][0][1][14] = 0j
                observations, audit = mixed_rate_observations(capture, FeatureConfig('log_ratio'))
                self.assertTrue(any(o.boundary for o in observations))
                self.assertTrue(any(a['status'] not in ('kept', 'other_shape') for a in audit))

    def test_prefix_invariance_and_no_late_window_search(self):
        observations = signal_observations()
        full, full_points = estimate_minute(observations)
        for count in (600, 1500, 2110):
            prefix, points = estimate_minute(observations[:count])
            cutoff = observations[count-1].time
            self.assertEqual(points, [p for p in full_points if p.available_at <= cutoff])
            if cutoff < 69:
                self.assertIsNone(prefix['candidate_bpm'])
            else:
                self.assertEqual(full, prefix)
        changed_future = observations[:2110] + [Observation(o.time, 999.) for o in observations[2110:]]
        self.assertEqual(full, estimate_minute(changed_future)[0])

    def test_long_gap_and_trailing_outage_do_not_get_a_full_minute_rate(self):
        observations = signal_observations(150)
        gap = [o for o in observations if not 30 <= o.time <= 31]
        result, points = estimate_minute(gap)
        self.assertEqual(result['status'], 'insufficient_data')
        self.assertIsNone(result['candidate_bpm'])
        self.assertEqual((result['start_s'], result['end_s']), (10, 70))
        self.assertFalse(any(30 < p.time < 31 for p in points))
        trailing, _ = estimate_minute([o for o in observations if o.time < 65])
        self.assertIsNone(trailing['candidate_bpm'])
        self.assertLess(trailing['supported_grid_fraction'], 1)

    def test_warmup_and_flat_signal_are_not_forced_estimates(self):
        warmup, _ = estimate_minute(signal_observations(), start_seconds=0)
        self.assertEqual(warmup['reason'], 'filter_warmup_inside_window')
        flat, _ = estimate_minute([Observation(i/30, -.4) for i in range(2700)])
        self.assertEqual(flat['status'], 'insufficient_variation')
        self.assertIsNone(flat['candidate_bpm'])
        self.assertIsNone(compare_reference(flat, manual_reference(events_capture([12])))['absolute_error_bpm'])

    def test_delayed_endpoint_is_audited_and_never_extrapolated(self):
        observations = signal_observations(offset=.04)
        before = [o for o in observations if o.time < 70]
        result, _ = estimate_minute(before)
        self.assertIsNone(result['candidate_bpm'])
        result, points = estimate_minute(observations)
        self.assertAlmostEqual(result['candidate_bpm'], 14)
        self.assertGreater(result['available_at_s'], 70)
        self.assertLessEqual(result['available_at_s'], 70.2)
        self.assertTrue(any(p.time < 70 < p.available_at for p in points))

    def test_manual_boundaries_use_integer_nanoseconds(self):
        capture = events_capture([9, 10, 12, 69, 70])
        capture.events[0]['host_monotonic_ns'] = T0+10_000_000_000-1
        capture.events[3]['host_monotonic_ns'] = T0+70_000_000_000-1
        reference = manual_reference(capture)
        self.assertEqual(reference['received_button_count'], 3)
        self.assertEqual(reference['manual_reference_bpm'], 3)

    def test_marker_quality_and_missing_events_withhold_comparison(self):
        for problem in ('gap', 'multiple', 'reboot', 'quality', 'empty', 'later_gap'):
            with self.subTest(problem=problem):
                capture = events_capture([] if problem == 'empty' else [12, 18, 22, 72])
                if problem == 'gap': capture.events[1]['missing_before'] = 1
                if problem == 'multiple': capture.events[1]['sender_address'] = '1122334455667788'
                if problem == 'reboot': capture.events[1]['boot_id'] = '1122334455667788'
                if problem == 'quality': capture.events[1]['quality'] = 'unknown'
                if problem == 'later_gap': capture.events[-1]['missing_before'] = 1
                reference = manual_reference(capture)
                self.assertEqual(reference['reference_status'], 'review_required')
                self.assertIsNone(reference['manual_reference_bpm'])
                self.assertIsNone(compare_reference({'candidate_bpm': 14}, reference)['absolute_error_bpm'])

    def test_cli_full_synthetic_transfer_export_and_read_only_integrity(self):
        with tempfile.TemporaryDirectory() as root:
            source = minute_fixture(Path(root)/'source')
            before = {p.name: sha256(p) for p in source.iterdir()}
            output = Path(root)/'output'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['joint-rate', '--input', str(source), '--output', str(output)]), 0)
            report = json.loads((output/'rate-report.json').read_text())
            self.assertTrue(report['complete'])
            self.assertEqual(report['policy']['rate_flags'], 'all_reported')
            self.assertAlmostEqual(report['comparison']['candidate_bpm'], 14)
            self.assertEqual(report['comparison']['received_button_count'], 14)
            self.assertEqual(report['comparison']['absolute_error_bpm'], 0)
            self.assertGreater(len(report['window_retained_rate_counts']), 1)
            for artifact in report['artifacts']:
                self.assertEqual(sha256(output/artifact['path']), artifact['sha256'])
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
            with self.assertRaises(FileExistsError): analyze_rate(source, output)
            with self.assertRaises(ValueError): analyze_rate(source, source/'out')
            with patch('wifi_csi_breathing.rate_comparison.save_figure', side_effect=OSError('Synthetic failure')):
                with self.assertRaises(OSError): analyze_rate(source, Path(root)/'partial')
            self.assertFalse((Path(root)/'partial/rate-report.json').exists())

    def test_invalid_window_and_short_recording_do_not_write_outputs(self):
        for start in (-1, float('inf'), float('nan'), 1e308):
            with self.assertRaises(ValueError): window_bounds(start)
        with tempfile.TemporaryDirectory() as root:
            source = fixture(Path(root)/'short')
            for start in ('10', '-1', 'nan'):
                with redirect_stderr(io.StringIO()):
                    self.assertEqual(main(['joint-rate','--input',str(source),'--output',root+'/out','--start',start]), 1)
            self.assertFalse(Path(root+'/out').exists())


if __name__ == '__main__':
    unittest.main()
