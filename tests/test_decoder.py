import hashlib
import math
import pathlib
import struct
import sys
import tempfile
import unittest

from wifi_csi_breathing.vendor.intel5300 import CSIFormatError, PacketClock, create_preview, decode_bfee, preview_frame, scale_factor


def record(ntx=1, nrx=1, timestamp=0, permutation=(0, 1, 2), value=None, noise=-95, rssi=40):
    # Independent bit-list encoder: unlike the decoder, never reads a packed integer.
    bits = []
    for group in range(30):
        bits.extend([1, 0, 1])
        for rx in range(nrx):
            for tx in range(ntx):
                real, imag = value(group, rx, tx) if value else (3, 4)
                for number in (real, imag):
                    bits.extend(((number & 255) >> bit) & 1 for bit in range(8))
    packed = bytearray((len(bits) + 7) // 8)
    for index, bit in enumerate(bits):
        packed[index // 8] |= bit << (index % 8)
    header = bytearray(20)
    struct.pack_into('<IH', header, 0, timestamp, 42)
    header[8:16] = bytes([nrx, ntx, rssi, 0, 0, noise & 255, 10,
                          sum(p << (i * 2) for i, p in enumerate(permutation))])
    struct.pack_into('<HH', header, 16, len(packed), 0x1234)
    body = b'\xbb' + header + packed
    return struct.pack('>H', len(body)) + body


class ParserTests(unittest.TestCase):
    def preview(self, blob, **kwargs):
        with tempfile.TemporaryDirectory() as directory:
            filename = pathlib.Path(directory) / 'csi.dat'
            filename.write_bytes(blob)
            return create_preview(filename, **kwargs)

    def test_signed_bit_alignment_and_permutation_all_shapes(self):
        for ntx in (1, 2, 3):
            for nrx in (1, 2, 3):
                perm = tuple(reversed(range(nrx))) + tuple(range(nrx, 3))
                value = lambda group, rx, tx: (-128 + group + tx, 127 - group - rx)
                decoded = decode_bfee(record(ntx, nrx, permutation=perm, value=value)[3:])
                for group in range(30):
                    for rx in range(nrx):
                        for tx in range(ntx):
                            self.assertEqual(decoded['csi'][tx][perm[rx]][group], complex(*value(group, rx, tx)))
                self.assertEqual(decoded['sequence'], 42)
                self.assertEqual(decoded['rate'], 0x1234)

    def test_scaling_analytical_and_phase(self):
        for ntx in (1, 2, 3):
            decoded = decode_bfee(record(ntx, 3)[3:])
            # Constant 3+4j matrix; RSSI=40, AGC=10, noise=-95.
            scale = 10 ** (-14 / 10) / (25 * ntx * 3)
            correction = [1, 2, 10 ** 0.45][ntx - 1]
            expected_amplitude = 5 * math.sqrt(scale * correction / (10 ** -9.5 + scale * 3 * ntx))
            self.assertAlmostEqual(5 * scale_factor(decoded), expected_amplitude, places=12)
            frame = preview_frame(decoded, 0)
            self.assertAlmostEqual(frame['scaledAmplitude'][0][0][0], expected_amplitude, places=5)
            self.assertAlmostEqual(frame['rawPhaseRadians'][0][0][0], math.atan2(4, 3), places=5)

    def test_phase_quadrants_and_undefined_zero(self):
        pairs = [(1, 1), (-1, 1), (-1, -1), (1, -1), (0, 0)]
        frame = self.preview(record(value=lambda g, r, t: pairs[g % 5]))['frames'][0]
        for index, expected in enumerate([math.pi/4, 3*math.pi/4, -3*math.pi/4, -math.pi/4]):
            self.assertAlmostEqual(frame['rawPhaseRadians'][0][0][index], expected, places=5)
        self.assertIsNone(frame['rawPhaseRadians'][0][0][4])

    def test_noise_fallback_and_unscalable(self):
        preview = self.preview(record(noise=-127) + record(rssi=0))
        self.assertEqual(preview['summary']['noiseFallbackPackets'], 1)
        self.assertEqual(preview['summary']['unscaledPackets'], 1)
        self.assertIsNone(preview['frames'][1]['scaledAmplitude'][0][0][0])
        a = decode_bfee(record(noise=-127)[3:])
        b = decode_bfee(record(noise=-92)[3:])
        self.assertEqual(scale_factor(a), scale_factor(b))

    def test_sampling_mixed_dimensions_and_timestamp_span(self):
        result = self.preview(b''.join(record(ntx=i % 3 + 1, nrx=3, timestamp=i*100000) for i in range(150)))
        self.assertEqual(result['summary']['csiPackets'], 150)
        self.assertEqual(len(result['frames']), 120)
        self.assertEqual([result['frames'][i]['packetIndex'] for i in (0, -1)], [1, 150])
        self.assertEqual(result['summary']['durationSeconds'], 14.9)
        self.assertEqual([s['packets'] for s in result['summary']['shapes']], [50, 50, 50])

    def test_non_csi_and_truncated_tail(self):
        for tail in (b'\x00', b'\x00\x08\xbb\x01'):
            result = self.preview(b'\x00\x02\x01\x02' + record() + tail)
            self.assertEqual(result['summary']['nonCsiRecords'], 1)
            self.assertEqual(result['summary']['trailingBytes'], len(tail))
            self.assertTrue(any('trailing' in warning for warning in result['warnings']))

    def test_reject_bad_header_length_dimensions_and_permutation(self):
        for offset, value in [(8, 0), (9, 4), (16, 0), (15, 0)]:
            payload = bytearray(record(nrx=3)[3:])
            payload[offset] = value
            with self.assertRaises(CSIFormatError):
                decode_bfee(payload)
        with self.assertRaises(CSIFormatError):
            self.preview(b'\x00\x00')

    def test_empty_or_no_complete_csi_rejected(self):
        for blob in (b'', b'\x00\x02\x01\x02', record()[:-1]):
            with self.assertRaises(CSIFormatError): self.preview(blob)

    def test_hash_and_size_verification(self):
        blob = record()
        self.preview(blob, expected_sha256=hashlib.sha256(blob).hexdigest(), expected_bytes=len(blob))
        for options in ({'expected_sha256': '0'*64}, {'expected_bytes': len(blob)+1}):
            with self.assertRaises(CSIFormatError): self.preview(blob, **options)

    def test_timestamp_wrap_and_reset(self):
        result = self.preview(record(timestamp=2**32-100) + record(timestamp=100))
        self.assertEqual(result['summary']['durationSeconds'], .0002)
        self.assertEqual(result['summary']['timestampWraps'], 1)
        reset = self.preview(record(timestamp=10000) + record(timestamp=500))
        self.assertIsNone(reset['summary']['durationSeconds'])
        self.assertTrue(all(f['secondsFromFirst'] is None for f in reset['frames']))

    def test_symbolic_link_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            real = pathlib.Path(directory) / 'real.dat'
            real.write_bytes(record())
            link = pathlib.Path(directory) / 'link.dat'
            link.symlink_to(real)
            with self.assertRaises(OSError): create_preview(link)


if __name__ == '__main__':
    unittest.main()
