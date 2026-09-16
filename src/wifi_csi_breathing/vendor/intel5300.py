#!/usr/bin/env python3
"""Intel 5300 .dat decoding and bounded offline preview, Python 3.9+ stdlib.

Format/scaling adapted from Daniel Halperin's Linux 802.11n CSI Tool:
read_bfee.c, read_bf_file.m, get_scaled_csi.m and get_total_rss.m.
Copyright (c) 2008-2011 Daniel Halperin. See THIRD_PARTY_NOTICES.md.
Matrices use [TX stream][permuted RX antenna][30 subcarrier groups].
No spatial-mapping removal, phase sanitization, filtering or unwrapping.
"""

import argparse
import hashlib
import json
import math
import os
import stat
import struct
import sys
from collections import Counter

PARSER_VERSION = "intel5300-v1"
MAX_BYTES = 256 * 1024 * 1024
MAX_PACKETS = 250000


class CSIFormatError(ValueError):
    pass


def decode_bfee(payload):
    """Decode a CSI record after the 0xbb type byte (not a netlink header)."""
    if len(payload) < 20:
        raise CSIFormatError("CSI record has an incomplete 20-byte header.")
    timestamp, sequence = struct.unpack_from("<IH", payload)
    nrx, ntx = payload[8:10]
    if not 1 <= nrx <= 3 or not 1 <= ntx <= 3:
        raise CSIFormatError("Unsupported RX/TX count; Intel 5300 expects 1 to 3.")
    length, rate = struct.unpack_from("<HH", payload, 16)
    expected = (30 * (nrx * ntx * 16 + 3) + 7) // 8
    if length != expected or len(payload) != 20 + expected:
        raise CSIFormatError("CSI matrix length does not match RX/TX dimensions.")
    permutation = [(payload[15] >> shift) & 3 for shift in (0, 2, 4)]
    # With one receiver the official reader does not permute. For multiple
    # receivers enforce a bijection, stronger than the original sum-only check.
    receivers = [0] if nrx == 1 else permutation[:nrx]
    if sorted(receivers) != list(range(nrx)):
        raise CSIFormatError("Invalid receive-antenna permutation; refusing to mislabel antennas.")
    csi = [[[0j] * 30 for _ in range(nrx)] for _ in range(ntx)]
    packed = int.from_bytes(payload[20:], "little")
    bit = 0
    for group in range(30):
        bit += 3
        # MATLAB stores the first dimension (TX) fastest in each RX/group.
        for rx in range(nrx):
            for tx in range(ntx):
                real = (packed >> bit) & 255
                imag = (packed >> (bit + 8)) & 255
                csi[tx][receivers[rx]][group] = complex(
                    real - 256 if real >= 128 else real,
                    imag - 256 if imag >= 128 else imag,
                )
                bit += 16
    return {
        "timestampLow": timestamp, "sequence": sequence, "nrx": nrx, "ntx": ntx,
        "rssi": list(payload[10:13]), "noise": struct.unpack_from("b", payload, 13)[0],
        "agc": payload[14], "permutation": [rx + 1 for rx in receivers],
        "rate": rate, "csi": csi,
    }


def scale_factor(packet):
    """Match get_scaled_csi: output magnitude has sqrt(SNR) scaling, not dBm."""
    power = sum(z.real * z.real + z.imag * z.imag
                for tx in packet["csi"] for rx in tx for z in rx)
    rss = sum(10 ** (rssi / 10) for rssi in packet["rssi"] if rssi != 0)
    if power == 0 or rss == 0:
        return None
    rss_mw = rss * 10 ** ((-44 - packet["agc"]) / 10)
    scale = rss_mw / (power / 30)
    noise_db = -92 if packet["noise"] == -127 else packet["noise"]
    total_noise = 10 ** (noise_db / 10) + scale * packet["nrx"] * packet["ntx"]
    factor = math.sqrt(scale / total_noise)
    if packet["ntx"] == 2:
        factor *= math.sqrt(2)
    elif packet["ntx"] == 3:
        factor *= math.sqrt(10 ** (4.5 / 10))
    return factor


def iter_packets(stream, counts):
    while True:
        offset = stream.tell()
        header = stream.read(2)
        if not header:
            return
        if len(header) < 2:
            counts["trailingBytes"] = len(header)
            return
        length = int.from_bytes(header, "big")
        if length < 1:
            raise CSIFormatError("Invalid zero-length record at byte %d." % offset)
        record = stream.read(length)
        if len(record) != length:
            counts["trailingBytes"] = 2 + len(record)
            return
        if record[0] != 187:
            counts["nonCsiRecords"] += 1
            continue
        try:
            packet = decode_bfee(record[1:])
        except CSIFormatError as error:
            raise CSIFormatError("Record at byte %d: %s" % (offset, error)) from error
        counts["csiPackets"] += 1
        if counts["csiPackets"] > MAX_PACKETS:
            raise CSIFormatError("Offline preview is limited to 250,000 CSI packets.")
        packet["packetIndex"] = counts["csiPackets"]
        yield packet


class PacketClock:
    def __init__(self):
        self.previous = None
        self.elapsed = 0
        self.wraps = 0
        self.discontinuities = 0

    def update(self, timestamp):
        if self.previous is not None:
            delta = (timestamp - self.previous) % (1 << 32)
            if delta >= (1 << 31):
                self.discontinuities += 1
            else:
                self.elapsed += delta
                if timestamp < self.previous:
                    self.wraps += 1
        self.previous = timestamp
        return self.elapsed / 1_000_000


def hash_stream(stream):
    stream.seek(0)
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def preview_frame(packet, seconds):
    factor = scale_factor(packet)
    amplitude, phase = [], []
    for tx in packet["csi"]:
        amplitude.append([])
        phase.append([])
        for rx in tx:
            amplitude[-1].append([round(abs(z) * factor, 6) if factor is not None else None for z in rx])
            # Phase is undefined for zero magnitude; null produces a chart gap.
            phase[-1].append([round(math.atan2(z.imag, z.real), 6) if z != 0 else None for z in rx])
    return {
        **{key: value for key, value in packet.items() if key != "csi"},
        "secondsFromFirst": seconds, "scaledAmplitude": amplitude,
        "rawPhaseRadians": phase, "scaleValid": factor is not None,
    }


def create_preview(filename, expected_sha256=None, expected_bytes=None, max_frames=120):
    if not 2 <= max_frames <= 200:
        raise CSIFormatError("Preview frame limit must be between 2 and 200.")
    descriptor = os.open(filename, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        file_stat = os.fstat(stream.fileno())
        if not stat.S_ISREG(file_stat.st_mode) or not 0 < file_stat.st_size <= MAX_BYTES:
            raise CSIFormatError("CSI input must be a nonempty regular file no larger than 256 MiB.")
        if expected_bytes is not None and file_stat.st_size != expected_bytes:
            raise CSIFormatError("CSI file size no longer matches its verified transfer receipt.")
        digest = hash_stream(stream)
        if expected_sha256 and digest != expected_sha256:
            raise CSIFormatError("CSI SHA-256 no longer matches its verified transfer receipt.")
        counts, shapes, clock = Counter(), Counter(), PacketClock()
        for packet in iter_packets(stream, counts):
            shapes[(packet["ntx"], packet["nrx"])] += 1
            clock.update(packet["timestampLow"])
            if packet["noise"] == -127:
                counts["noiseFallbackPackets"] += 1
            if scale_factor(packet) is None:
                counts["unscaledPackets"] += 1
        total = counts["csiPackets"]
        if total == 0:
            raise CSIFormatError("No complete Intel 5300 CSI records were found.")
        sample_count = min(max_frames, total)
        indices = {round(i * (total - 1) / max(1, sample_count - 1)) + 1 for i in range(sample_count)}
        stream.seek(0)
        sample_clock, frames = PacketClock(), []
        for packet in iter_packets(stream, Counter()):
            seconds = sample_clock.update(packet["timestampLow"])
            if packet["packetIndex"] in indices:
                frames.append(preview_frame(packet, None if clock.discontinuities else seconds))
        if hash_stream(stream) != digest:
            raise CSIFormatError("CSI input changed during parsing. Retry with the verified copy.")
    warnings = [
        "Raw wrapped phase is uncalibrated. No phase sanitization, unwrapping or spatial-mapping removal is applied.",
        "Preview frames are evenly selected by packet index; this is not a full-rate signal for frequency analysis.",
    ]
    if counts["trailingBytes"]:
        warnings.append("Ignored %d trailing bytes from an incomplete final record." % counts["trailingBytes"])
    if counts["noiseFallbackPackets"]:
        warnings.append("%d packets used the official -92 dB noise fallback because noise was -127." % counts["noiseFallbackPackets"])
    if counts["unscaledPackets"]:
        warnings.append("%d packets have zero RSSI or CSI power; scaled amplitude is unavailable for them." % counts["unscaledPackets"])
    if clock.discontinuities:
        warnings.append("Timestamp discontinuities detected; relative time is unavailable. Packet indices remain valid.")
    return {
        "schemaVersion": 1, "parserVersion": PARSER_VERSION,
        "source": {"bytes": file_stat.st_size, "sha256": digest},
        "summary": {
            **{key: counts[key] for key in ("csiPackets", "nonCsiRecords", "trailingBytes", "noiseFallbackPackets", "unscaledPackets")},
            "sampledPackets": len(frames), "subcarrierGroups": 30,
            "durationSeconds": None if clock.discontinuities else clock.elapsed / 1_000_000,
            "timestampWraps": clock.wraps, "timestampDiscontinuities": clock.discontinuities,
            "shapes": [{"ntx": tx, "nrx": rx, "packets": count} for (tx, rx), count in sorted(shapes.items())],
        },
        "amplitudeUnit": "sqrt(SNR), official CSI Tool scaling",
        "phaseUnit": "radians, wrapped [-pi, pi], uncalibrated",
        "warnings": warnings, "frames": frames,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--expected-bytes", type=int)
    parser.add_argument("--frames", type=int, default=120)
    args = parser.parse_args()
    try:
        result = create_preview(args.input, args.expected_sha256, args.expected_bytes, args.frames)
        print(json.dumps(result, allow_nan=False, separators=(",", ":")))
    except (CSIFormatError, OSError) as error:
        print(json.dumps({"error": str(error)}), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
