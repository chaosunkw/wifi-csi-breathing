"""Synthetic receipts and records only; never commit participant recordings."""
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch
import csv
import io
import json
import tempfile
import unittest

import numpy as np

from test_decoder import record
from wifi_csi_breathing.analysis import process_observations
from wifi_csi_breathing.cli import main
from wifi_csi_breathing.data import FeatureConfig, sha256
from wifi_csi_breathing.joint import (EVENT_FIELDS, FILES, INDEX_FIELDS, JointCapture,
                                    draw_joint, joint_analyze)


T0 = 10**17 + 123
SESSION = 'synthetic-joint-session'


def save_json(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


def save_table(path, fields, rows):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refresh_receipt(folder):
    files = [{'name': name, 'bytes': (folder / name).stat().st_size,
              'sha256': sha256(folder / name), 'verified': True} for name in FILES]
    save_json(folder / 'transfer.json', {'status': 'verified', 'sessionId': SESSION,
                                        'captureStatus': 'completed', 'files': files})


def fixture(folder, empty=False):
    folder.mkdir()
    for name in FILES:
        (folder / name).write_bytes(b'')
    save_json(folder / 'metadata.json', {'schemaVersion': 1, 'sessionId': SESSION,
                                       'markerEnabled': True, 'status': 'completed'})
    save_json(folder / 'timing.json', {'schemaVersion': 1, 'sessionId': SESSION,
        'clock': 'CLOCK_MONOTONIC', 't0_ns': str(T0), 't1_ns': str(T0 + 10**9),
        'boot_id': 'synthetic-host-boot', 'alignment': 'host-receipt',
        'physicalSynchronization': False})
    rows, blobs, offset = [], [], 0
    for i, ns in enumerate((T0-1, T0, T0+100000001, T0+150000000, T0+10**9), 1):
        # A reset of the chipset clock must not rebase the host clock timeline.
        low = 100 if i % 2 else 0
        blob = record(3, 3, timestamp=low)
        blobs.append(blob)
        rows.append(dict(zip(INDEX_FIELDS, (i, offset, len(blob), ns, 187, low))))
        offset += len(blob)
    (folder / 'csi.dat').write_bytes(b''.join(blobs))
    save_table(folder / 'csi-times.csv', INDEX_FIELDS, rows)
    events = []
    for i, ns in enumerate(() if empty else (T0-1, T0, T0+250000001, T0+10**9), 1):
        events.append(dict(zip(EVENT_FIELDS, (ns, 1, '0011223344556677', 'aabbccddeeff0011',
            i, i*100, 'inhalation_onset_manual', 'first_marker' if i == 1 else 'ok', 0))))
    save_table(folder / 'marker-events.csv', EVENT_FIELDS, events)
    (folder / 'marker-records.jsonl').write_text(''.join(json.dumps({**e, 'accepted': True, 'simulation': False})+'\n'
                                                     for e in events), encoding='utf-8')
    save_json(folder / 'marker-status.json', {'schemaVersion': 1, 'sessionId': SESSION,
        'status': 'finished', 'error': None, 'markerReady': True, 'csiReady': True,
        'accepted': len(events), 'rejected': 0, 'missing': 0, 'csiRecords': len(rows)})
    refresh_receipt(folder)
    return folder


class JointTests(unittest.TestCase):
    def test_integer_clock_and_half_open_window_not_chipset_or_sender_zero(self):
        with tempfile.TemporaryDirectory() as root:
            capture = JointCapture(fixture(Path(root) / 'source'))
            obs = capture.observations(FeatureConfig())
            self.assertEqual([o.index for o in obs], [2, 3, 4])
            self.assertEqual([o.time for o in obs], [0, .100000001, .15])
            times = [(e['host_monotonic_ns']-capture.t0)/1e9 for e in capture.events
                     if capture.inside(e['host_monotonic_ns'])]
            self.assertEqual(times, [0, .250000001])

    def test_first_packet_is_not_rebased_and_shared_plot_positions(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            with (folder / 'csi-times.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            rows[1]['host_monotonic_ns'] = str(T0 + 10_000_000)
            save_table(folder / 'csi-times.csv', INDEX_FIELDS, rows)
            refresh_receipt(folder)
            capture = JointCapture(folder)
            obs = capture.observations(FeatureConfig())
            self.assertEqual(obs[0].time, .01)
            fig = draw_joint(capture, obs, FeatureConfig())
            a, b = fig.axes
            self.assertTrue(a.get_shared_x_axes().joined(a, b))
            self.assertEqual(a.get_xlim(), (0, 1))
            self.assertEqual(float(a.collections[0].get_offsets()[0][0]), .01)
            np.testing.assert_allclose([s[0][0] for s in b.collections[0].get_segments()], [0, .250000001])
            fig.clear()

    def test_receipt_hash_size_and_exact_allowlist(self):
        for mutation in ('hash', 'size', 'extra', 'duplicate', 'unverified'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as root:
                folder = fixture(Path(root) / 'source')
                receipt = json.loads((folder / 'transfer.json').read_text())
                if mutation == 'hash': receipt['files'][0]['sha256'] = '0'*64
                if mutation == 'size': receipt['files'][0]['bytes'] += 1
                if mutation == 'extra': receipt['files'].append(receipt['files'][0])
                if mutation == 'duplicate': receipt['files'][-1] = receipt['files'][0]
                if mutation == 'unverified': receipt['status'] = 'failed'
                save_json(folder / 'transfer.json', receipt)
                with self.assertRaises(ValueError): JointCapture(folder)

    def test_symlink_refused(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            original = folder / 'ping.log'
            original.rename(Path(root) / 'log')
            original.symlink_to(Path(root) / 'log')
            with self.assertRaises(ValueError): JointCapture(folder)

    def test_invalid_clock_session_status_and_window(self):
        changes = [('timing.json', 'clock', 'CLOCK_REALTIME'), ('timing.json', 't0_ns', 1.5),
                   ('timing.json', 't1_ns', str(T0)), ('timing.json', 'sessionId', 'other'),
                   ('timing.json', 'physicalSynchronization', True),
                   ('metadata.json', 'markerEnabled', False), ('metadata.json', 'status', 'failed'),
                   ('marker-status.json', 'accepted', 99), ('marker-status.json', 'csiRecords', 99),
                   ('marker-status.json', 'status', 'running')]
        for file, key, value in changes:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                folder = fixture(Path(root) / 'source')
                obj = json.loads((folder / file).read_text())
                obj[key] = value
                save_json(folder / file, obj)
                refresh_receipt(folder)
                with self.assertRaises(ValueError): JointCapture(folder)

    def test_index_corruption_missing_extra_and_backward_refused(self):
        for key, value in [('file_offset', 1), ('record_bytes', 1), ('record_index', 8),
                           ('record_type', 1), ('timestamp_low', 999), ('host_monotonic_ns', T0-2)]:
            with self.subTest(key=key), tempfile.TemporaryDirectory() as root:
                folder = fixture(Path(root) / 'source')
                with (folder / 'csi-times.csv').open() as stream: rows = list(csv.DictReader(stream))
                rows[1][key] = value
                save_table(folder / 'csi-times.csv', INDEX_FIELDS, rows)
                refresh_receipt(folder)
                with self.assertRaises(ValueError): JointCapture(folder)
        for kind in ('missing', 'extra', 'truncated'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as root:
                folder = fixture(Path(root) / 'source')
                if kind == 'truncated':
                    path = folder / 'csi.dat'
                    path.write_bytes(path.read_bytes()[:-1])
                else:
                    with (folder / 'csi-times.csv').open() as stream: rows = list(csv.DictReader(stream))
                    rows = rows[:-1] if kind == 'missing' else rows + [rows[-1]]
                    save_table(folder / 'csi-times.csv', INDEX_FIELDS, rows)
                refresh_receipt(folder)
                with self.assertRaises(ValueError): JointCapture(folder)

    def test_marker_journal_disagreement_and_simulation_refused(self):
        for field, value in [('sequence', 99), ('simulation', True)]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as root:
                folder = fixture(Path(root) / 'source')
                path = folder / 'marker-records.jsonl'
                records = [json.loads(s) for s in path.read_text().splitlines()]
                records[0][field] = value
                path.write_text(''.join(json.dumps(r)+'\n' for r in records))
                refresh_receipt(folder)
                with self.assertRaises(ValueError): JointCapture(folder)

    def test_marker_gap_and_reboot_preserve_identity_without_filling_events(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            with (folder / 'marker-events.csv').open() as stream:
                events = list(csv.DictReader(stream))
            for event in events:
                for key in ('host_monotonic_ns', 'host_unix_ns', 'sequence', 'sender_elapsed_ms', 'missing_before'):
                    event[key] = int(event[key])
            events[1].update(sequence=4, missing_before=2, quality='sequence_gap')
            for i, event in enumerate(events[2:], 1):
                event.update(boot_id='1122334455667788', sequence=i,
                             quality='new_sender_boot' if i == 1 else 'ok')
            save_table(folder / 'marker-events.csv', EVENT_FIELDS, events)
            (folder / 'marker-records.jsonl').write_text(''.join(json.dumps({**e, 'accepted': True, 'simulation': False})+'\n'
                                                               for e in events))
            status = json.loads((folder / 'marker-status.json').read_text())
            status['missing'] = 2
            save_json(folder / 'marker-status.json', status)
            refresh_receipt(folder)
            capture = JointCapture(folder)
            self.assertEqual(len(capture.events), 4)
            self.assertEqual([e['sequence'] for e in capture.events], [1, 4, 1, 2])

    def test_non_csi_records_are_indexed_but_not_feature_samples(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            blob = b'\x00\x02\x01\x02'
            data = folder / 'csi.dat'
            data.write_bytes(blob + data.read_bytes())
            with (folder / 'csi-times.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            for row in rows:
                row['record_index'] = int(row['record_index']) + 1
                row['file_offset'] = int(row['file_offset']) + len(blob)
            rows.insert(0, dict(zip(INDEX_FIELDS, (1, 0, len(blob), T0-2, 1, ''))))
            save_table(folder / 'csi-times.csv', INDEX_FIELDS, rows)
            status = json.loads((folder / 'marker-status.json').read_text())
            status['csiRecords'] = 6
            save_json(folder / 'marker-status.json', status)
            refresh_receipt(folder)
            capture = JointCapture(folder)
            self.assertEqual(len(capture.index), 6)
            self.assertEqual(len(capture.observations(FeatureConfig())), 3)

    def test_empty_markers_still_produce_valid_comparison(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source', empty=True)
            report = joint_analyze(folder, Path(root) / 'out', ['amplitude'])
            self.assertEqual(report['window_markers'], 0)
            self.assertTrue(report['complete'])

    def test_duplicates_and_dimension_changes_reset_without_time_shift(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            with (folder / 'csi-times.csv').open() as stream: rows = list(csv.DictReader(stream))
            rows[2]['host_monotonic_ns'] = str(T0)
            save_table(folder / 'csi-times.csv', INDEX_FIELDS, rows)
            refresh_receipt(folder)
            capture = JointCapture(folder)
            obs = capture.observations(FeatureConfig())
            self.assertIsNone(obs[1].value)
            self.assertEqual(obs[1].reason, 'duplicate_timestamp')
            capture.packets[3]['ntx'] = 2
            obs = capture.observations(FeatureConfig())
            self.assertEqual(obs[2].reason, 'dimension_change')
            self.assertIsNone(obs[2].value)
            _, full, _ = process_observations(obs, 'amplitude')
            _, prefix, _ = process_observations(obs[:2], 'amplitude')
            self.assertEqual([asdict(p) for p in full[:len(prefix)]], [asdict(p) for p in prefix])

    def test_end_to_end_cli_outputs_sources_unchanged_and_non_overwrite(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            before = {p.name: sha256(p) for p in folder.iterdir()}
            output = Path(root) / 'out'
            with redirect_stdout(io.StringIO()):
                code = main(['joint', '--input', str(folder), '--output', str(output)])
            self.assertEqual(code, 0)
            report = json.loads((output / 'joint-report.json').read_text())
            self.assertEqual(report['window_markers'], 2)
            self.assertEqual(report['excluded_markers'], 2)
            self.assertEqual(report['excluded_indexed_records'], 2)
            self.assertEqual(len(report['results']), 3)
            for item in report['artifacts']:
                self.assertEqual(sha256(output / item['path']), item['sha256'])
            self.assertEqual(before, {p.name: sha256(p) for p in folder.iterdir()})
            with self.assertRaises(FileExistsError): joint_analyze(folder, output)
            with self.assertRaises(ValueError): joint_analyze(folder, folder / 'out')
            with self.assertRaises(ValueError): joint_analyze(folder, Path(root) / 'dup', ['amplitude']*2)

    def test_no_completion_marker_after_render_or_integrity_failure(self):
        with tempfile.TemporaryDirectory() as root:
            folder = fixture(Path(root) / 'source')
            for i, target in enumerate(('draw_joint', 'JointCapture.verify')):
                output = Path(root) / f'partial{i}'
                if target == 'draw_joint':
                    with patch('wifi_csi_breathing.joint.' + target, side_effect=OSError('Synthetic failure')):
                        with self.assertRaises(OSError): joint_analyze(folder, output)
                else:
                    # Source mutation after loading, before final verification.
                    from wifi_csi_breathing.joint import draw_joint as original
                    def change_source(*args):
                        (folder / 'ping.log').write_text('changed')
                        return original(*args)
                    with patch('wifi_csi_breathing.joint.draw_joint', side_effect=change_source):
                        with self.assertRaises(ValueError): joint_analyze(folder, output, ['amplitude'])
                self.assertFalse((output / 'joint-report.json').exists())

    def test_cli_missing_source_is_clear_error(self):
        with tempfile.TemporaryDirectory() as root, redirect_stderr(io.StringIO()) as error:
            self.assertEqual(main(['joint', '--input', root, '--output', root+'/out']), 1)
            self.assertIn('source file', error.getvalue())


if __name__ == '__main__':
    unittest.main()
