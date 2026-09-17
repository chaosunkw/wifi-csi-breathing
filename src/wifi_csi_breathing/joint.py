"""Read-only, receipt-verified CSI / manual-marker comparison on a host clock."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json
import re

from .analysis import process_observations, write_csv, write_json
from .data import FeatureConfig, MODES, Observation, feature, read_json, sha256
from .processing import ProcessingConfig
from .static_analysis import NAMES, UNITS
from .vendor.intel5300 import MAX_BYTES, MAX_PACKETS, decode_bfee


FILES = ('csi.dat', 'ping.log', 'receiver.log', 'metadata.json', 'csi-times.csv',
         'marker-records.jsonl', 'marker-events.csv', 'marker-status.json', 'timing.json')
INDEX_FIELDS = ('record_index', 'file_offset', 'record_bytes', 'host_monotonic_ns',
                'record_type', 'timestamp_low')
EVENT_FIELDS = ('host_monotonic_ns', 'host_unix_ns', 'sender_address', 'boot_id',
                'sequence', 'sender_elapsed_ms', 'event', 'quality', 'missing_before')
LIMITATION = ('Host-receipt alignment, not physical synchronization. Manual button events '
              'are self-reported inhalation onsets, not physiological ground truth. '
              'Human reaction, radio/USB buffering and host scheduling add unmeasured delay.')


def integer(value, name):
    # Subtract integer nanoseconds before converting to seconds (never float epochs).
    if isinstance(value, bool) or not isinstance(value, (str, int)) or not re.fullmatch(r'[0-9]+', str(value)):
        raise ValueError(f'Invalid nonnegative integer: {name}')
    return int(value)


def regular(path, limit=MAX_BYTES):
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise ValueError(f'Expected a bounded regular source file: {path.name}')


def object_json(path):
    content = read_json(path)
    if not isinstance(content, dict):
        raise ValueError(f'Expected a JSON object: {path.name}')
    return content


def table(path, fields):
    with path.open(encoding='utf-8', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(fields):
            raise ValueError(f'Unexpected CSV columns: {path.name}')
        for index, row in enumerate(reader):
            if index >= MAX_PACKETS or None in row or any(v is None for v in row.values()):
                raise ValueError(f'Malformed or oversized CSV: {path.name}')
            yield row


class JointCapture:
    """Require a verified Monitor transfer, not an unverified folder of CSV files."""

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        receipt_path = self.directory / 'transfer.json'
        regular(receipt_path)
        self.receipt_hash = sha256(receipt_path)
        self.receipt = object_json(receipt_path)
        files = self.receipt.get('files', [])
        if (self.receipt.get('status') != 'verified' or not isinstance(files, list)
                or len(files) != len(FILES) or not all(isinstance(f, dict) for f in files)
                or {f.get('name') for f in files} != set(FILES)):
            raise ValueError('Expected a verified nine-file joint transfer receipt')
        self.files = files
        self.verify()
        self.metadata = object_json(self.directory / 'metadata.json')
        self.timing = object_json(self.directory / 'timing.json')
        self.status = object_json(self.directory / 'marker-status.json')
        self.session_id = self.receipt.get('sessionId')
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError('Missing capture session identity')
        for item in (self.metadata, self.timing, self.status):
            if item.get('schemaVersion') != 1 or item.get('sessionId') != self.session_id:
                raise ValueError('Joint artifact schema/session identity mismatch')
        if self.metadata.get('markerEnabled') is not True:
            raise ValueError('Session does not declare Marker capture')
        if (self.timing.get('clock') != 'CLOCK_MONOTONIC'
                or self.timing.get('alignment') != 'host-receipt'
                or self.timing.get('physicalSynchronization') is not False
                or not self.timing.get('boot_id')):
            raise ValueError('Unsupported or missing common host clock')
        self.t0 = integer(self.timing.get('t0_ns'), 't0_ns')
        self.t1 = integer(self.timing.get('t1_ns'), 't1_ns')
        if self.t1 <= self.t0:
            raise ValueError('Joint recording requires a positive closed time window')
        self.duration = (self.t1 - self.t0) / 1e9
        if self.metadata.get('status') not in ('completed', 'stopped'):
            raise ValueError('Failed or unfinished captures are not supported by this comparison')
        if self.receipt.get('captureStatus') != self.metadata['status']:
            raise ValueError('Capture status differs from transfer receipt')
        if (self.status.get('status') != 'finished' or self.status.get('error') is not None
                or self.status.get('markerReady') is not True or self.status.get('csiReady') is not True):
            raise ValueError('Marker capture did not finish successfully')
        self.events = self.read_events()
        self.packets, self.index = self.read_csi()
        self.verify()

    def verify(self):
        regular(self.directory / 'transfer.json')
        if sha256(self.directory / 'transfer.json') != self.receipt_hash:
            raise ValueError('Transfer receipt changed during analysis')
        for item in self.files:
            path = self.directory / item['name']
            regular(path)
            if (item.get('verified') is not True or path.stat().st_size != item.get('bytes')
                    or sha256(path) != item.get('sha256')):
                raise ValueError(f'Source differs from transfer receipt: {path.name}')

    def inside(self, ns):
        return self.t0 <= ns < self.t1

    def read_events(self):
        events, previous, seen = [], None, set()
        for row in table(self.directory / 'marker-events.csv', EVENT_FIELDS):
            for key in ('host_monotonic_ns', 'host_unix_ns', 'sequence', 'sender_elapsed_ms', 'missing_before'):
                row[key] = integer(row[key], key)
            ns = row['host_monotonic_ns']
            identity = (row['sender_address'], row['boot_id'], row['sequence'])
            if (previous is not None and ns < previous) or identity in seen:
                raise ValueError('Unordered or duplicate accepted Marker event')
            if (row['event'] != 'inhalation_onset_manual' or row['sequence'] < 1
                    or not re.fullmatch(r'[0-9a-fA-F]{16}', row['sender_address'])
                    or not re.fullmatch(r'[0-9a-fA-F]{16}', row['boot_id'])):
                raise ValueError('Invalid manual event identity or type')
            previous = ns
            seen.add(identity)
            events.append(row)
        # Cross-check the accepted CSV against the raw journal, including counters.
        accepted, rejected, missing = [], 0, 0
        with (self.directory / 'marker-records.jsonl').open(encoding='utf-8') as stream:
            for index, line in enumerate(stream):
                if index >= MAX_PACKETS or len(line) > 65536:
                    raise ValueError('Marker journal exceeds limits')
                record = json.loads(line)
                if (not isinstance(record, dict) or record.get('simulation') is not False
                        or type(record.get('accepted')) is not bool):
                    raise ValueError('Expected hardware Marker journal records')
                if record['accepted']:
                    accepted.append({key: record[key] for key in EVENT_FIELDS})
                    missing += integer(record['missing_before'], 'missing_before')
                else:
                    rejected += 1
        if accepted != events:
            raise ValueError('Marker CSV disagrees with raw journal')
        for key, value in (('accepted', len(events)), ('rejected', rejected), ('missing', missing)):
            if integer(self.status.get(key), key) != value:
                raise ValueError(f'Marker final counter mismatch: {key}')
        return events

    def read_csi(self):
        packets, rows, previous = [], [], None
        with (self.directory / 'csi.dat').open('rb') as stream:
            for index, row in enumerate(table(self.directory / 'csi-times.csv', INDEX_FIELDS), 1):
                offset = stream.tell()
                header = stream.read(2)
                if len(header) != 2:
                    raise ValueError('CSI index exceeds data length')
                length = int.from_bytes(header, 'big')
                body = stream.read(length)
                if not length or len(body) != length:
                    raise ValueError('Incomplete CSI record')
                for key in INDEX_FIELDS[:-1]:
                    row[key] = integer(row[key], key)
                if (row['record_index'], row['file_offset'], row['record_bytes'], row['record_type']) != (
                        index, offset, length + 2, body[0]):
                    raise ValueError('CSI index does not match record framing')
                ns = row['host_monotonic_ns']
                if previous is not None and ns < previous:
                    raise ValueError('Backward CSI host clock')
                previous = ns
                packet = decode_bfee(body[1:]) if body[0] == 187 else None
                expected = str(packet['timestampLow']) if packet else ''
                if row['timestamp_low'] != expected:
                    raise ValueError('CSI index chipset timestamp mismatch')
                rows.append(row)
                packets.append(packet)
            if stream.read(1) or not rows or not any(p is not None for p in packets):
                raise ValueError('Unindexed trailing data or no complete CSI packets')
        if integer(self.status.get('csiRecords'), 'csiRecords') != len(rows):
            raise ValueError('CSI final counter disagrees with index')
        return packets, rows

    def observations(self, config):
        output, previous_shape, previous_ns = [], None, None
        for row, packet in zip(self.index, self.packets):
            ns = row['host_monotonic_ns']
            if packet is None or not self.inside(ns):
                continue
            shape = (packet['ntx'], packet['nrx'])
            value = feature(packet, config)
            reason = ('dimension_change' if previous_shape is not None and shape != previous_shape
                      else 'invalid_feature' if value is None else '')
            if previous_ns == ns:
                value, reason = None, 'duplicate_timestamp'
            output.append(Observation((ns - self.t0) / 1e9, value, bool(reason), reason, row['record_index']))
            previous_shape, previous_ns = shape, ns
        if not output:
            raise ValueError('No CSI packets inside the formal recording window')
        return output


def draw_joint(capture, observations, config):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(11, 6.6))
    FigureCanvasAgg(fig)
    top, bottom = fig.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    top.set_title(f'{NAMES[config.mode]} and manual inhalation markers', loc='left', fontsize=13)
    # Packet-level values are deliberately not tuned, shifted or filtered toward markers.
    # A scatter plot cannot draw a misleading line across a gap or shape transition.
    valid = [o for o in observations if o.value is not None]
    top.scatter([o.time for o in valid], [o.value for o in valid], s=3, color='#0072B2',
                label='Packet feature (phase wrapped)' if config.mode == 'phase_difference' else 'Packet feature')
    top.set_ylabel(UNITS[config.mode])
    top.legend(loc='upper right', fontsize=8)
    if not valid:
        top.text(0.5, 0.5, 'No valid feature values for this fixed channel policy',
                 transform=top.transAxes, ha='center', fontsize=10)
    bottom.set_title('Manual button events — host receipt time', loc='left', fontsize=10)
    selected = [e for e in capture.events if capture.inside(e['host_monotonic_ns'])]
    times = [(e['host_monotonic_ns'] - capture.t0) / 1e9 for e in selected]
    bottom.vlines(times, 0, 0.75, color='#008678', linewidth=1.2)
    if len(selected) <= 40:
        for t, event in zip(times, selected):
            bottom.text(t, 0.82, str(event['sequence']), ha='center', fontsize=8)
    if not selected:
        bottom.text(0.5, 0.5, 'No accepted markers in this recording window', transform=bottom.transAxes, ha='center')
    bottom.set_ylim(0, 1.1)
    bottom.set_yticks([])
    bottom.set_xlabel('Seconds from common Ubuntu T0 (host monotonic clock)')
    for ax in (top, bottom):
        ax.set_xlim(0, capture.duration)
        ax.grid(axis='x', color='#e5e7eb', linewidth=0.6)
        ax.spines[['top', 'right']].set_visible(False)
    fig.text(0.09, 0.94, f'TX {config.tx} / RX {config.rx} / reference RX {config.reference_rx} / group {config.subcarrier}'
             f' | {len(observations)} CSI packets | {len(selected)} manual events | original status: {capture.metadata["status"]}', fontsize=8)
    fig.text(0.09, 0.915, f'Unavailable feature packets: {len(observations)-len(valid)}'
             f' | rejected markers (whole capture): {capture.status["rejected"]}'
             f' | observed missing sequences (whole capture): {capture.status["missing"]}', fontsize=8, color='#555555')
    fig.text(0.09, 0.045, 'Host-receipt alignment only; reaction and transport delays are not corrected. Not physiological ground truth.', fontsize=8)
    fig.text(0.09, 0.02, 'Fixed feature settings; no marker-driven channel selection, phase shift, or breathing-rate claim.', fontsize=8)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.87, bottom=0.14, hspace=0.4)
    return fig


def joint_analyze(directory, output, modes=MODES, tx=1, rx=1, reference_rx=2, subcarrier=15):
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Select at least one non-duplicate mode')
    configs = [FeatureConfig(m, tx, rx, reference_rx, subcarrier) for m in modes]
    capture = JointCapture(directory)
    output = Path(output).resolve()
    if output.is_relative_to(capture.directory):
        raise ValueError('Output must not be inside the source recording')
    output.mkdir(parents=True, exist_ok=False)
    processing = ProcessingConfig()
    event_rows = [{**event, 'relative_seconds': (event['host_monotonic_ns'] - capture.t0) / 1e9,
                   'in_window': capture.inside(event['host_monotonic_ns'])} for event in capture.events]
    write_csv(output / 'markers.csv', event_rows, (*EVENT_FIELDS, 'relative_seconds', 'in_window'))
    results = []
    for config in configs:
        observations = capture.observations(config)
        processor, points, _ = process_observations(observations, config.mode, processing)
        write_csv(output / f'{config.mode}-observations.csv', (asdict(o) for o in observations),
                  ('time', 'value', 'boundary', 'reason', 'index'))
        write_csv(output / f'{config.mode}-waveform.csv', (asdict(p) for p in points),
                  ('time', 'available_at', 'raw', 'filtered', 'segment'))
        fig = draw_joint(capture, observations, config)
        for extension in ('png', 'pdf', 'svg'):
            fig.savefig(output / f'{config.mode}-joint.{extension}', dpi=180)
        fig.clear()
        results.append({'feature': asdict(config), 'window_csi_packets': len(observations),
                        'invalid_feature_packets': sum(o.value is None for o in observations),
                        'filter_resets': processor.resets})
    images = ''.join(f'<h2>{NAMES[m]}</h2><img src="{m}-joint.png" alt="{m} and manual markers">'
                     f'<p><a href="{m}-joint.pdf">PDF</a> · <a href="{m}-joint.svg">SVG</a></p>' for m in modes)
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>CSI and manual markers</title>'
        '<style>body{font:16px system-ui;max-width:1150px;margin:30px auto;padding:0 16px}img{width:100%}</style>'
        f'<h1>CSI and manual markers</h1><p>{LIMITATION}</p>'
        '<p>Packet features above; manual events below. Both share the stored T0; neither is independently zeroed. '
        'Feature settings are fixed. Waveform CSVs also contain the unchanged causal processor output; '
        'the figure shows packet features, not bandpass output.</p>' + images, encoding='utf-8')
    capture.verify()
    import matplotlib
    import numpy
    import scipy
    report = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'session_id': capture.session_id, 'source_directory': str(capture.directory),
              'original_status': capture.metadata['status'],
              'source_files': capture.files, 'receipt_sha256': capture.receipt_hash,
              'timing': capture.timing, 'duration_seconds': capture.duration,
              'relative_time_formula': '(host_monotonic_ns - t0_ns) / 1e9',
              'window': '[t0_ns, t1_ns)', 'marker_status': capture.status,
              'window_markers': sum(row['in_window'] for row in event_rows),
              'excluded_markers': sum(not row['in_window'] for row in event_rows),
              'total_indexed_records': len(capture.index),
              'excluded_indexed_records': sum(not capture.inside(row['host_monotonic_ns']) for row in capture.index),
              'processing': asdict(processing), 'results': results, 'limitation': LIMITATION,
              'raw_data_modified': False, 'physiological_validation': False,
              'runtime': {'numpy': numpy.__version__, 'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
              'implementation_hashes': [{'path': str(p.relative_to(Path(__file__).parent)), 'sha256': sha256(p)}
                                        for p in sorted(Path(__file__).parent.rglob('*.py'))],
              'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / 'joint-report.json', report)
    return report
