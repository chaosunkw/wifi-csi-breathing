from pathlib import Path
from unittest.mock import patch
import json
import tempfile
import unittest

from test_joint import fixture, refresh_receipt, save_json, T0
from test_decoder import record
from wifi_csi_breathing.batch_diagnostics import (
    archive_batch, load_archive, corrected_labels, diagnose_archive,
    diagnostic_feature, packet_rows, figure_for, file_manifest,
)
from wifi_csi_breathing.data import FeatureConfig, feature, sha256
from wifi_csi_breathing.joint import JointCapture
from wifi_csi_breathing.vendor.intel5300 import decode_bfee


def six_sources(root):
    source = root / 'transfers'
    namespace = source / 'namespace'
    namespace.mkdir(parents=True)
    changes = []
    for i, (condition, trial) in enumerate((('Empty', 1), ('Empty', 2), ('F0', 1), ('F0', 2), ('R90', 1), ('R90', 2))):
        folder = fixture(namespace / f'synthetic-{i}', empty=condition == 'Empty')
        for name in ('metadata.json', 'timing.json', 'marker-status.json'):
            item = json.loads((folder / name).read_text())
            item['sessionId'] = folder.name
            if name == 'metadata.json':
                item.update(batchId='SYNTHETIC', positionId='L01_'+('F0' if condition == 'Empty' else condition),
                            activityLabel='normal_breathing', trialNumber=trial)
            save_json(folder / name, item)
        refresh_receipt(folder)
        receipt = json.loads((folder / 'transfer.json').read_text())
        receipt['sessionId'] = folder.name
        save_json(folder / 'transfer.json', receipt)
        if condition == 'Empty':
            changes.append({'session_id': folder.name, 'field': 'activityLabel',
                            'original': 'normal_breathing', 'corrected': 'Empty',
                            'reason': 'Synthetic operator correction', 'confirmed_by': 'test operator'})
    corrections = root / 'corrections.json'
    save_json(corrections, {'schema_version': 1, 'batch_id': 'SYNTHETIC', 'corrections': changes})
    return source, corrections


class BatchDiagnosticsTests(unittest.TestCase):
    def test_archive_copies_all_files_without_modifying_metadata_or_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, corrections = six_sources(root)
            before = {p.parent.name: file_manifest(p.parent) for p in source.glob('*/*/metadata.json')}
            archive = root / 'archive'
            report = archive_batch(source, 'SYNTHETIC', corrections, archive)
            self.assertEqual(len(report['sessions']), 6)
            manifest, digest, layout = load_archive(archive)
            self.assertEqual([c for c, _, _ in layout], ['Empty', 'Empty', 'F0', 'F0', 'R90', 'R90'])
            self.assertEqual(digest, sha256(archive / 'archive.json'))
            for entry in manifest['sessions']:
                self.assertEqual(entry['files'], before[entry['session_id']])
                self.assertEqual(file_manifest(Path(entry['source_directory'])), entry['files'])
                self.assertEqual(entry['original_metadata']['activityLabel'], 'normal_breathing')
            self.assertEqual(manifest['sessions'][0]['analysis_metadata']['activityLabel'], 'Empty')
            with self.assertRaises(FileExistsError): archive_batch(source, 'SYNTHETIC', corrections, archive)
            with self.assertRaises(ValueError): archive_batch(source, 'SYNTHETIC', corrections, source / 'archive')

    def test_bad_corrections_and_duplicate_trials_refused(self):
        for variant in ('missing', 'wrong_original', 'unknown', 'duplicate', 'wrong_field', 'unconfirmed'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, path = six_sources(root)
                corrections = json.loads(path.read_text())
                if variant == 'missing': corrections['corrections'] = []
                if variant == 'wrong_original': corrections['corrections'][0]['original'] = 'other'
                if variant == 'unknown': corrections['corrections'][0]['session_id'] = 'missing'
                if variant == 'duplicate': corrections['corrections'].append(corrections['corrections'][0])
                if variant == 'wrong_field': corrections['corrections'][0]['field'] = 'participantId'
                if variant == 'unconfirmed': corrections['corrections'][0]['confirmed_by'] = ''
                save_json(path, corrections)
                with self.assertRaises(ValueError): archive_batch(source, 'SYNTHETIC', path, root / 'out')
                self.assertFalse((root / 'out').exists())

    def test_corrupt_archive_copy_and_changed_correction_refused(self):
        for variant in ('source', 'correction'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as temp:
                root = Path(temp)
                source, path = six_sources(root)
                output = root / 'archive'
                archive_batch(source, 'SYNTHETIC', path, output)
                target = output / 'synthetic-0' / 'ping.log' if variant == 'source' else output / 'label-corrections.json'
                target.write_text('changed')
                with self.assertRaises(ValueError): load_archive(output)

    def test_failed_copy_has_no_completion_manifest(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, path = six_sources(root)
            with patch('wifi_csi_breathing.batch_diagnostics.shutil.copy2', side_effect=OSError('Synthetic failure')):
                with self.assertRaises(OSError): archive_batch(source, 'SYNTHETIC', path, root / 'archive')
            self.assertFalse((root / 'archive' / 'archive.json').exists())

    def test_shape_local_features_do_not_relax_existing_feature_policy(self):
        for ntx in (1, 2, 3):
            packet = decode_bfee(record(ntx, 3)[3:])
            for mode in ('amplitude', 'log_ratio', 'phase_difference'):
                cfg = FeatureConfig(mode)
                value = diagnostic_feature(packet, cfg)
                self.assertIsNotNone(value)
                if ntx == 3: self.assertAlmostEqual(value, feature(packet, cfg))
                else: self.assertIsNone(feature(packet, cfg))
        packet = decode_bfee(record(1, 1)[3:])
        self.assertIsNone(diagnostic_feature(packet, FeatureConfig('log_ratio')))
        self.assertIsNone(diagnostic_feature(packet, FeatureConfig(tx=2)))

    def test_shape_panels_never_join_lines_and_marker_axis_is_common(self):
        with tempfile.TemporaryDirectory() as temp:
            capture = JointCapture(fixture(Path(temp) / 'one'))
            capture.packets[1] = decode_bfee(record(1, 3)[3:])
            capture.packets[2] = decode_bfee(record(2, 3)[3:])
            rows = packet_rows(capture, FeatureConfig())
            self.assertEqual([r['ntx'] for r in rows], [1, 2, 3])
            self.assertEqual([r['seconds'] for r in rows], [0, .100000001, .15])
            fig = figure_for([('F0', 1, capture, rows)], FeatureConfig(), (0, 5), 1)
            self.assertEqual(len(fig.axes), 5)
            for i in range(3):
                self.assertEqual(len(fig.axes[i].lines), 0)
                self.assertEqual(len(fig.axes[i].collections[0].get_offsets()), 1)
            self.assertTrue(all(fig.axes[0].get_shared_x_axes().joined(fig.axes[0], a) for a in fig.axes[1:]))
            ticks = [segment[0][0] for segment in fig.axes[3].collections[0].get_segments()]
            self.assertEqual(ticks, [0, .250000001])
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            heading = fig.texts[0].get_window_extent(renderer)
            subtitle = fig.texts[1].get_window_extent(renderer)
            panel_title = fig.axes[0].get_title(loc='left')
            self.assertTrue(panel_title.startswith('F0'))
            self.assertLess(subtitle.y1, heading.y0)
            self.assertLess(fig.axes[0]._left_title.get_window_extent(renderer).y1, subtitle.y0)
            fig.clear()

    def test_diagnostics_end_to_end_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, path = six_sources(root)
            archive = root / 'archive'
            archive_batch(source, 'SYNTHETIC', path, archive)
            output = root / 'result'
            report = diagnose_archive(archive, output, ['amplitude'])
            self.assertTrue(report['complete'])
            self.assertEqual(len(report['summaries']), 6)
            self.assertEqual([s['window_markers'] for s in report['summaries']], [0, 0, 2, 2, 2, 2])
            for item in report['artifacts']:
                self.assertEqual(sha256(output / item['path']), item['sha256'])
            with self.assertRaises(FileExistsError): diagnose_archive(archive, output)
            with self.assertRaises(ValueError): diagnose_archive(archive, archive / 'out')
            with patch('wifi_csi_breathing.batch_diagnostics.figure_for', side_effect=OSError('Synthetic failure')):
                with self.assertRaises(OSError): diagnose_archive(archive, root / 'partial')
            self.assertFalse((root / 'partial' / 'diagnostics-report.json').exists())
            load_archive(archive)
