"""One-minute comparisons using synthetic signals and transfer receipts only."""
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import io
import csv
import json
import math
import tempfile
import unittest

from test_decoder import record
from test_joint import (SESSION, T0, fixture, refresh_receipt, save_json, save_table)
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import FeatureConfig, Observation, feature, sha256
from wifi_csi_breathing.joint import EVENT_FIELDS, INDEX_FIELDS
from wifi_csi_breathing.rate_comparison import (analyze_rate, compare_reference,
    draw_filtered, estimate_minute, gap_processing, manual_reference, mixed_rate_observations,
    observation_gaps, window_bounds)
from wifi_csi_breathing.fixed_configuration import segment_summary
from wifi_csi_breathing.subcarriers import new_figure


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


def minute_fixture(folder, gap=None):
    """Synthetic binary packets exercise the full parser/receipt/CLI path."""
    fixture(folder)
    rows, blobs, offset = [], [], 0
    for i in range(2700):
        seconds = i/30
        if gap and gap[0] < seconds < gap[1]:
            continue
        ntx = 1 if i % 79 == 10 else 2 if i % 113 == 11 else 3
        amplitude = round(45 * math.exp(.18*math.sin(2*math.pi*14/60*seconds)))
        blob = bytearray(record(ntx, 3, timestamp=i,
                               value=lambda g, r, t: (amplitude if r == 0 else 45, 0)))
        blob[21:23] = (273+i % 5).to_bytes(2, 'little')
        blobs.append(blob)
        rows.append(dict(zip(INDEX_FIELDS, (len(rows)+1, offset, len(blob), T0+round(seconds*1e9), 187, i))))
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
    def test_short_gap_preserves_state_and_true_time_without_ten_second_blackout(self):
        observations = [o for o in signal_observations() if not 37 < o.time < 37.3]
        old, _ = estimate_minute(observations)
        result, points = estimate_minute(observations, gap_policy='preserve-short')
        self.assertGreater(old['filter_resets'], 0)
        self.assertEqual(result['filter_resets'], 0)
        self.assertEqual(result['candidate_bpm'], 14)
        self.assertEqual(result['bridged_intervals'], 1)
        self.assertGreater(result['bridged_grid_samples'], 0)
        self.assertEqual(result['settled_grid_samples'], 1200)
        self.assertEqual(result['settled_unbridged_grid_samples'] + result['bridged_grid_samples'], 1200)
        self.assertTrue(all(math.isclose(b.time-a.time, .05, abs_tol=1e-8)
                            for a, b in zip(points, points[1:])))
        gaps = observation_gaps(observations, 70, 'preserve-short')
        self.assertEqual(len(gaps), 1)
        self.assertAlmostEqual(gaps[0]['duration_s'], .3)
        interior = [p for p in points if 37+1e-8 < p.time < 37.3-1e-8]
        self.assertTrue(all(p.available_at >= 37.3 for p in interior))
        # Missing internal grid values must never be drawn as received data.
        fig = new_figure((8, 4))
        ax = fig.subplots()
        draw_filtered(ax, points, segment_summary(points, gap_processing('preserve-short')),
                      10, 70, gaps, True)
        stable = ax.lines[0]
        pairs = list(zip(stable.get_xdata(), stable.get_ydata()))
        self.assertTrue(all(math.isnan(y) for x, y in pairs if 37+1e-8 < x < 37.3-1e-8))
        self.assertTrue(all(math.isfinite(y) for x, y in pairs if 37.35 < x < 47))
        self.assertFalse(ax.patches)
        fig.clear()

    def test_gap_policy_remains_causal_and_does_not_extrapolate(self):
        observations = [o for o in signal_observations() if not 37 < o.time < 37.3]
        full, full_points = estimate_minute(observations, gap_policy='preserve-short')
        for count in (1100, 1111, 1120, 2100):
            prefix, points = estimate_minute(observations[:count], gap_policy='preserve-short')
            cutoff = observations[count-1].time
            self.assertEqual(points, [p for p in full_points if p.available_at <= cutoff])
            if cutoff < 70:
                self.assertIsNone(prefix['candidate_bpm'])
            else:
                self.assertEqual(full, prefix)
        before = [o for o in observations if o.time <= 37]
        _, points = estimate_minute(before, gap_policy='preserve-short')
        self.assertLessEqual(max(p.time for p in points), 37+1e-8)
        # A bounded right endpoint can complete the minute, with explicit latency.
        delayed = [o for o in signal_observations(offset=.01) if not 69.8 < o.time < 70.1]
        result, _ = estimate_minute(delayed, gap_policy='preserve-short')
        self.assertEqual(result['candidate_bpm'], 14)
        self.assertGreater(result['available_at_s'], 70.1)
        self.assertLessEqual(result['available_at_s'], 70.5)
        self.assertEqual(result, estimate_minute(delayed+[Observation(90.1, 999.)],
                                               gap_policy='preserve-short')[0])

    def test_long_gap_and_invalid_records_keep_resets_but_settling_is_visible(self):
        for kind in ('long_gap', 'invalid', 'duplicate', 'boundary'):
            observations = signal_observations()
            if kind == 'long_gap':
                observations = [o for o in observations if not 37 < o.time < 38]
            elif kind == 'invalid':
                observations[1110] = Observation(37, None)
            elif kind == 'duplicate':
                observations.insert(1111, observations[1110])
            else:
                observations[1110] = Observation(37, -.4, True, 'non_csi_record')
            result, points = estimate_minute(observations, gap_policy='preserve-short')
            self.assertGreater(result['filter_resets'], 0)
            self.assertIsNone(result['candidate_bpm'])
            self.assertEqual(result['reason'], 'filter_reset_inside_window')
            if kind == 'long_gap':
                self.assertFalse(any(37+1e-8 < p.time < 38-1e-8 for p in points))
                fig = new_figure((8, 4))
                ax = fig.subplots()
                draw_filtered(ax, points, segment_summary(points, gap_processing('preserve-short')),
                              10, 70, observation_gaps(observations, 70, 'preserve-short'), True)
                dashed = [line for line in ax.lines if line.get_linestyle() == '--']
                self.assertTrue(any(math.isfinite(y) for line in dashed
                                    for x, y in zip(line.get_xdata(), line.get_ydata()) if 38 < x < 48))
                self.assertFalse(ax.patches)
                fig.clear()

    def test_many_short_gaps_are_not_reported_as_full_observed_coverage(self):
        observations = [o for o in signal_observations()
                        if not any(t < o.time < t+.3 for t in range(12, 66, 2))]
        result, _ = estimate_minute(observations, gap_policy='preserve-short')
        self.assertEqual(result['filter_resets'], 0)
        self.assertGreater(result['bridged_grid_fraction'], .05)
        self.assertEqual(result['reason'], 'excessive_gap_interpolation')
        self.assertIsNone(result['candidate_bpm'])
        with self.assertRaises(ValueError):
            estimate_minute(observations, gap_policy='anything')

    def test_preserved_gap_cli_exports_blanks_and_audits_internal_interpolation(self):
        with tempfile.TemporaryDirectory() as root:
            source = minute_fixture(Path(root)/'source', gap=(37, 37.3))
            before = {p.name: sha256(p) for p in source.iterdir()}
            output = Path(root)/'output'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['joint-rate', '--input', str(source), '--output', str(output),
                                       '--shape-policy', 'first-stream', '--gap-policy', 'preserve-short']), 0)
            report = json.loads((output/'rate-report.json').read_text())
            self.assertEqual(report['policy']['gap_policy'], 'preserve-short')
            self.assertEqual(report['processing']['max_gap_seconds'], .5)
            self.assertEqual(report['comparison']['filter_resets'], 0)
            self.assertEqual(report['comparison']['candidate_bpm'], 14)
            self.assertEqual(report['comparison']['received_button_count'], 14)
            with (output/'waveform.csv').open() as stream:
                points = list(csv.DictReader(stream))
            missing = [p for p in points if p['in_gap'] == 'True']
            self.assertTrue(missing)
            self.assertTrue(all(p['display_filtered'] == '' and p['filtered'] != '' for p in missing))
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
            for artifact in report['artifacts']:
                self.assertEqual(sha256(output/artifact['path']), artifact['sha256'])

    def test_first_stream_has_common_indices_and_common_gain_cancellation(self):
        capture = mixed_capture(3)
        expected = [o.value for o in signal_observations(3)]
        for i, packet in enumerate(capture.packets):
            gain = (2.3 - .7j) * packet['ntx']
            for rx in (0, 1):
                packet['csi'][0][rx][14] *= gain
            # Other streams must never influence selection or the ratio.
            for stream in packet['csi'][1:]:
                stream[0][14] = 100000j
                stream[1][14] = 0j
        config = FeatureConfig('log_ratio')
        observations, audit = mixed_rate_observations(capture, config, 'first-stream')
        self.assertEqual(len(observations), len(capture.packets))
        self.assertEqual({a['ntx'] for a in audit}, {1, 2, 3})
        self.assertTrue(all(a['status'] == 'kept' for a in audit))
        for observation, value in zip(observations, expected):
            self.assertAlmostEqual(observation.value, value)
        # The shared baseline feature extractor remains restricted to 3x3.
        self.assertIsNone(feature(capture.packets[10], config))

    def test_first_stream_retains_long_shape_bursts_without_reset(self):
        capture = mixed_capture()
        for i, packet in enumerate(capture.packets):
            if 20 <= i/30 < 22 or 30 <= i/30 < 31:
                packet['ntx'] = 1 if i/30 < 22 else 2
                packet['csi'] = packet['csi'][:1] * packet['ntx']
        config = FeatureConfig('log_ratio')
        baseline, _ = mixed_rate_observations(capture, config)
        selected, audit = mixed_rate_observations(capture, config, 'first-stream')
        old, _ = estimate_minute(baseline)
        new, _ = estimate_minute(selected)
        self.assertIsNone(old['candidate_bpm'])
        self.assertGreater(old['filter_resets'], 0)
        self.assertEqual(new['filter_resets'], 0)
        self.assertEqual(new['supported_grid_fraction'], 1)
        self.assertEqual(new['candidate_bpm'], 14)
        self.assertTrue(any(a['shape_changed'] for a in audit))
        # A genuine 1 s outage still prevents a complete minute estimate.
        outage, points = estimate_minute([o for o in selected if not 40 <= o.time <= 41])
        self.assertIsNone(outage['candidate_bpm'])
        self.assertGreater(outage['filter_resets'], 0)
        self.assertFalse(any(40 < p.time < 41 for p in points))

    def test_first_stream_does_not_silently_correct_configuration_offsets(self):
        capture = mixed_capture(3)
        expected = [o.value for o in signal_observations(3)]
        for packet in capture.packets:
            if packet['ntx'] == 1:
                packet['csi'][0][0][14] *= math.exp(.8)
        observations, audit = mixed_rate_observations(capture, FeatureConfig('log_ratio'), 'first-stream')
        for i, (observation, row) in enumerate(zip(observations, audit)):
            self.assertAlmostEqual(observation.value, expected[i] + (.8 if row['ntx'] == 1 else 0))
        self.assertFalse(any(o.boundary for o in observations))

    def test_first_stream_invalid_packets_remain_barriers(self):
        for kind in ('duplicate', 'unknown', 'zero', 'nan', 'infinite'):
            with self.subTest(kind=kind):
                capture = mixed_capture(1)
                self.assertEqual(capture.packets[10]['ntx'], 1)
                if kind == 'duplicate':
                    capture.index[10]['host_monotonic_ns'] = capture.index[9]['host_monotonic_ns']
                elif kind == 'unknown':
                    capture.packets[10] = None
                else:
                    capture.packets[10]['csi'][0][1][14] = {'zero': 0j, 'nan': complex(float('nan')),
                                                          'infinite': complex(float('inf'))}[kind]
                observations, audit = mixed_rate_observations(capture, FeatureConfig('log_ratio'), 'first-stream')
                self.assertTrue(observations[10].boundary)
                self.assertIsNone(observations[10].value)
                self.assertNotEqual(audit[10]['status'], 'kept')

    def test_first_stream_policy_is_explicit_and_prefix_invariant(self):
        capture = mixed_capture()
        config = FeatureConfig('log_ratio')
        for policy, feature_config in (('unknown', config), ('first-stream', FeatureConfig('log_ratio', tx=2))):
            with self.assertRaises(ValueError):
                mixed_rate_observations(capture, feature_config, policy)
        full, full_audit = mixed_rate_observations(capture, config, 'first-stream')
        full_result, full_points = estimate_minute(full)
        for count in (600, 1500, 2110):
            prefix = SimpleNamespace(t0=capture.t0, packets=capture.packets[:count],
                                     index=capture.index[:count], inside=capture.inside)
            observations, audit = mixed_rate_observations(prefix, config, 'first-stream')
            self.assertEqual(observations, full[:count])
            self.assertEqual(audit, full_audit[:count])
            result, points = estimate_minute(observations)
            self.assertEqual(points, [p for p in full_points if p.available_at <= observations[-1].time])
            if count == 2110:
                self.assertEqual(result, full_result)
        # This experiment accepts only the explicitly documented three RX chains.
        capture.packets[10]['nrx'] = 2
        capture.packets[10]['csi'] = [s[:2] for s in capture.packets[10]['csi']]
        _, audit = mixed_rate_observations(capture, config, 'first-stream')
        self.assertEqual(audit[10]['status'], 'other_shape')

    def test_first_stream_cli_export_and_invalid_stream_selection(self):
        with tempfile.TemporaryDirectory() as root:
            source = minute_fixture(Path(root)/'source')
            before = {p.name: sha256(p) for p in source.iterdir()}
            output = Path(root)/'output'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(['joint-rate', '--input', str(source), '--output', str(output),
                                       '--shape-policy', 'first-stream']), 0)
            report = json.loads((output/'rate-report.json').read_text())
            self.assertEqual(report['policy']['accepted_shapes'], [[1,3], [2,3], [3,3]])
            self.assertEqual(report['policy']['rate_flags'], 'all_reported')
            self.assertFalse(report['policy']['mapping_correction'])
            self.assertEqual(set(report['window_retained_shape_counts']), {'1x3', '2x3', '3x3'})
            self.assertTrue(report['window_shape_changes'])
            self.assertEqual(report['comparison']['retained_packet_fraction'], 1)
            self.assertEqual(report['comparison']['candidate_bpm'], 14)
            self.assertEqual(report['comparison']['absolute_error_bpm'], 0)
            for artifact in report['artifacts']:
                self.assertEqual(sha256(output/artifact['path']), artifact['sha256'])
            self.assertEqual(before, {p.name: sha256(p) for p in source.iterdir()})
            with redirect_stderr(io.StringIO()):
                self.assertEqual(main(['joint-rate', '--input', str(source), '--output', root+'/invalid',
                                       '--shape-policy', 'first-stream', '--tx', '2']), 1)
            self.assertFalse(Path(root+'/invalid').exists())

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
