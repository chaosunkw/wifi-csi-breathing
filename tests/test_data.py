from dataclasses import replace
from pathlib import Path
import json
import tempfile
import unittest

from test_decoder import record
from wifi_csi_breathing.data import Dataset, FeatureConfig, RAW_FILES, feature, read_observations, sha256, contained
from wifi_csi_breathing.vendor.intel5300 import CSIFormatError, decode_bfee
from wifi_csi_breathing.analysis import analyze


def make_dataset(root, blob):
    root.mkdir(exist_ok=True)
    (root / 'csi.dat').write_bytes(blob)
    (root / 'metadata.json').write_text(json.dumps({'status': 'failed', 'durationSeconds': 120}), encoding='utf-8')
    for name in ('ping.log', 'receiver.log'):
        (root / name).write_text('Synthetic fixture\n', encoding='utf-8')
    files = tuple({'name': name, 'bytes': (root / name).stat().st_size, 'sha256': sha256(root / name)} for name in RAW_FILES)
    return Dataset('synthetic', root, files, 'Test fixture 🧪')


class DataTests(unittest.TestCase):
    def test_wrap_mixed_dimensions_and_failed_metadata_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            blob = (record(3, 3, (1 << 32) - 10000) + record(3, 3, 20000)
                    + record(1, 3, 50000) + record(3, 3, 80000))
            ds = make_dataset(Path(directory), blob)
            observations, audit = read_observations(ds, FeatureConfig())
            self.assertEqual(audit['counts']['timestampWraps'], 1)
            self.assertEqual(audit['original_status'], 'failed')
            self.assertEqual(audit['dimension_change_indices'], [3, 4])
            self.assertIsNone(observations[2].value)
            self.assertTrue(observations[3].boundary)
            self.assertAlmostEqual(observations[-1].time, 0.09)
            self.assertEqual(ds.verify()['status'], 'failed')

    def test_hash_mismatch_and_path_escape_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            ds = make_dataset(Path(directory), record(3, 3))
            (ds.directory / 'csi.dat').write_bytes(record(3, 3, 200))
            with self.assertRaisesRegex(ValueError, 'SHA-256 mismatch'):
                ds.verify()
            with self.assertRaises(ValueError):
                contained(ds.directory, '../escape')
            with self.assertRaises(ValueError):
                replace(ds, files=ds.files + (ds.files[0],)).verify()

    def test_truncated_and_backward_records_rejected(self):
        for blob in (record(3, 3) + b'\x00', record(3, 3, 50000) + record(3, 3, 20000)):
            with self.subTest(length=len(blob)), tempfile.TemporaryDirectory() as directory:
                ds = make_dataset(Path(directory), blob)
                with self.assertRaises(CSIFormatError):
                    read_observations(ds, FeatureConfig())

    def test_ratio_and_phase_difference_common_phase_cancels(self):
        packet = decode_bfee(record(3, 3, value=lambda g, rx, tx: ((3, 4) if rx == 0 else (0, 2)))[3:])
        ratio = feature(packet, FeatureConfig('log_ratio'))
        phase = feature(packet, FeatureConfig('phase_difference'))
        import cmath
        import math
        rotation = cmath.exp(1.5j)
        packet['csi'] = [[[z * rotation for z in groups] for groups in receivers] for receivers in packet['csi']]
        self.assertAlmostEqual(feature(packet, FeatureConfig('phase_difference')), phase)
        self.assertAlmostEqual(feature(packet, FeatureConfig('log_ratio')), ratio)
        self.assertAlmostEqual(ratio, math.log(2.5))
        with self.assertRaises(ValueError):
            FeatureConfig('phase_difference', rx=1, reference_rx=1)

    def test_export_refuses_overwrite_and_raw_destination(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ds = make_dataset(root / 'raw', record(3, 3))
            with self.assertRaises(ValueError):
                analyze([ds], ds.directory / 'results', ('amplitude',))
            report = analyze([ds], root / 'results', ('amplitude',))
            self.assertTrue(report['complete'])
            self.assertEqual(report['results'][0]['eligible_windows'], 0)
            with self.assertRaises(FileExistsError):
                analyze([ds], root / 'results', ('amplitude',))
            ds.verify()

    def test_operator_label_does_not_enter_signal_processing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            ds = make_dataset(root / 'raw', record(3, 3) + record(3, 3, 40000))
            a, _ = read_observations(ds, FeatureConfig())
            b, _ = read_observations(replace(ds, label='Different activity label'), FeatureConfig())
            self.assertEqual(a, b)
