"""Verified read-only dataset intake and chipset-specific feature extraction."""
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
import hashlib
import json
import math
import statistics

from .vendor.intel5300 import iter_packets, scale_factor, MAX_BYTES, CSIFormatError

RAW_FILES = ('csi.dat', 'metadata.json', 'ping.log', 'receiver.log')
MODES = ('amplitude', 'log_ratio', 'phase_difference')


class Cancelled(Exception):
    pass


def check_cancel(event):
    if event is not None and event.is_set():
        raise Cancelled('Operation cancelled')


def sha256(path, event=None):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(1024 * 1024):
            check_cancel(event)
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('JSON input exceeds 4 MiB')
    return json.loads(path.read_text(encoding='utf-8'))


def contained(root, relative):
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root.resolve()):
        raise ValueError('Dataset path escapes its root')
    return candidate


@dataclass(frozen=True)
class Dataset:
    id: str
    directory: Path
    files: tuple
    label: str

    def verify(self, event=None):
        if len(self.files) != 4 or {item['name'] for item in self.files} != set(RAW_FILES):
            raise ValueError('Expected exactly four raw source files')
        for item in self.files:
            check_cancel(event)
            path = contained(self.directory, item['name'])
            if not path.is_file() or path.stat().st_size != item['bytes']:
                raise ValueError(f"Size mismatch: {self.id}/{item['name']}")
            if sha256(path, event) != item['sha256']:
                raise ValueError(f"SHA-256 mismatch: {self.id}/{item['name']}")
        return read_json(self.directory / 'metadata.json')


def load_catalog(path):
    content = read_json(path)
    if content.get('schema_version') != 1:
        raise ValueError('Unsupported catalog schema')
    root = Path(content['source_root']).expanduser().resolve()
    datasets = [Dataset(item['id'], contained(root, item['folder']), tuple(item['files']), item['label'])
                for item in content['datasets']]
    if not datasets or len({d.id for d in datasets}) != len(datasets):
        raise ValueError('Catalog is empty or has duplicate IDs')
    return datasets


def import_archive(archive, output):
    archive = Path(archive).resolve()
    audit = read_json(archive / 'verification.json')
    datasets = []
    for item in audit['results']:
        m = item['metadata']
        label = f"{item['folder']} | {m['activityLabel']} | {m['positionId']} | Trial {m['trialNumber']}"
        dataset = Dataset(item['folder'], contained(archive, item['folder']), tuple(item['files']), label)
        metadata = dataset.verify()
        if metadata != m:
            raise ValueError('Manifest metadata disagrees with source metadata')
        datasets.append({'id': dataset.id, 'folder': item['folder'], 'label': label, 'files': item['files']})
    output = Path(output)
    if output.resolve().is_relative_to(archive):
        raise ValueError('The new catalog must not be written inside the source archive')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        json.dump({'schema_version': 1, 'source_root': str(archive), 'datasets': datasets}, stream, indent=2)
    return len(datasets)


@dataclass(frozen=True)
class FeatureConfig:
    mode: str = 'amplitude'
    tx: int = 1
    rx: int = 1
    reference_rx: int = 2
    subcarrier: int = 15

    def __post_init__(self):
        if self.mode not in MODES or not 1 <= self.subcarrier <= 30:
            raise ValueError('Invalid feature or subcarrier group')
        if not all(1 <= i <= 3 for i in (self.tx, self.rx, self.reference_rx)):
            raise ValueError('TX/RX must be 1..3')
        if self.mode != 'amplitude' and self.rx == self.reference_rx:
            raise ValueError('Ratio/phase difference requires two different RX channels')


@dataclass(frozen=True)
class Observation:
    time: float
    value: float | None
    boundary: bool = False
    reason: str = ''
    index: int = 0


def feature(packet, config):
    # Fixed 3x3 baseline: never silently splice a different spatial mapping.
    if (packet['ntx'], packet['nrx']) != (3, 3):
        return None
    a = packet['csi'][config.tx - 1][config.rx - 1][config.subcarrier - 1]
    if config.mode == 'amplitude':
        factor = scale_factor(packet)
        return abs(a) * factor if factor is not None else None
    b = packet['csi'][config.tx - 1][config.reference_rx - 1][config.subcarrier - 1]
    if abs(a) == 0 or abs(b) == 0:
        return None
    if config.mode == 'log_ratio':
        return math.log(abs(a)) - math.log(abs(b))
    product = a * b.conjugate()
    return math.atan2(product.imag, product.real)


def read_observations(dataset, config, event=None):
    metadata = dataset.verify(event)
    filename = dataset.directory / 'csi.dat'
    if not 0 < filename.stat().st_size <= MAX_BYTES:
        raise ValueError('CSI file must be nonempty and no larger than 256 MiB')
    counts, shapes = Counter(), Counter()
    observations, intervals, changes = [], [], []
    previous_timestamp = previous_shape = None
    elapsed = 0
    with filename.open('rb') as stream:
        for packet in iter_packets(stream, counts):
            check_cancel(event)
            timestamp = packet['timestampLow']
            delta = None if previous_timestamp is None else (timestamp - previous_timestamp) % (1 << 32)
            if delta is not None and delta >= (1 << 31):
                raise CSIFormatError('Backward timestamp discontinuity: refuse invented timing')
            if delta is not None:
                elapsed += delta
                if delta > 0:
                    intervals.append(delta / 1000)
                else:
                    counts['duplicateTimestamps'] += 1
                if timestamp < previous_timestamp:
                    counts['timestampWraps'] += 1
            shape = (packet['ntx'], packet['nrx'])
            changed = previous_shape is not None and shape != previous_shape
            shapes[str(shape)] += 1
            if changed:
                changes.append(packet['packetIndex'])
            value = feature(packet, config)
            reason = 'dimension_change' if changed else 'invalid_feature' if value is None else 'duplicate_timestamp' if delta == 0 else ''
            if value is None:
                counts['invalidFeaturePackets'] += 1
            if delta == 0:
                value = None
            observations.append(Observation(elapsed / 1e6, value, bool(reason), reason, packet['packetIndex']))
            previous_timestamp, previous_shape = timestamp, shape
    if not observations or counts['trailingBytes']:
        raise CSIFormatError('Empty CSI or incomplete trailing record; analysis refused')
    # Check again in case another process changed the file during decoding.
    expected = next(f['sha256'] for f in dataset.files if f['name'] == 'csi.dat')
    if sha256(filename, event) != expected:
        raise ValueError('CSI changed during read')
    duration = observations[-1].time
    audit = {'packets': len(observations), 'span_seconds': duration,
             'observed_hz': (len(observations) - 1) / duration if duration else None,
             'interval_median_ms': statistics.median(intervals) if intervals else None,
             'interval_max_ms': max(intervals) if intervals else None,
             'counts': dict(counts), 'shapes': dict(shapes), 'dimension_change_indices': changes,
             'original_status': metadata['status'], 'metadata': metadata}
    return observations, audit
