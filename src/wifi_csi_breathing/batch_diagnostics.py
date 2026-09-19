"""Immutable joint-session archives and shape-separated packet diagnostics."""
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import html
import math
import re
import shutil

from .analysis import write_json
from .data import FeatureConfig, MODES, contained, sha256
from .joint import FILES, JointCapture, LIMITATION, object_json, regular
from .static_analysis import NAMES, UNITS, shared_limits
from .vendor.intel5300 import scale_factor

SOURCE_FILES = (*FILES, 'transfer.json')
CONDITIONS = ('Empty', 'F0', 'R90')
COLORS = {1: '#B66700', 2: '#0072B2', 3: '#008675'}


def file_manifest(directory):
    return [{'name': name, 'bytes': (directory / name).stat().st_size,
             'sha256': sha256(directory / name)} for name in SOURCE_FILES]


def corrected_labels(captures, corrections, batch):
    if corrections.get('schema_version') != 1 or corrections.get('batch_id') != batch:
        raise ValueError('Correction schema or batch does not match')
    mapping = {c.session_id: c for c in captures}
    if len(mapping) != len(captures):
        raise ValueError('Duplicate source session identity')
    effective = {key: dict(c.metadata) for key, c in mapping.items()}
    seen = set()
    for change in corrections.get('corrections', []):
        key = change['session_id']
        if key not in mapping or key in seen or change['field'] != 'activityLabel':
            raise ValueError('Unknown/duplicate correction or unsupported field')
        if effective[key].get('activityLabel') != change['original']:
            raise ValueError('Correction original value disagrees with source')
        if change['corrected'] != 'Empty' or not change.get('reason') or not change.get('confirmed_by'):
            raise ValueError('An Empty correction needs explicit confirmation and reason')
        effective[key]['activityLabel'] = change['corrected']
        seen.add(key)
    placements = []
    for key, m in effective.items():
        if m.get('batchId') != batch:
            raise ValueError('Source batch does not match')
        activity, position = m.get('activityLabel', '').lower(), m.get('positionId', '')
        if activity == 'empty':
            condition = 'Empty'
        elif activity == 'normal_breathing' and position.endswith(('_F0', '_R90')):
            condition = position.rsplit('_', 1)[1]
        else:
            raise ValueError('Unknown six-session condition')
        trial = m.get('trialNumber')
        if type(trial) is not int or trial <= 0:
            raise ValueError('Trial number must be a positive integer')
        placements.append((condition, trial, key))
    if len(placements) != 6 or any(len({t for cond, t, _ in placements if cond == c}) != 2 for c in CONDITIONS):
        raise ValueError('Expected two distinct trials for each of Empty, F0 and R90')
    return sorted(placements, key=lambda row: (CONDITIONS.index(row[0]), row[1])), effective


def archive_batch(source_root, batch, correction_path, output):
    """Copy exactly the verified artifacts; keep a separate label amendment."""
    source_root, output = Path(source_root).resolve(), Path(output).resolve()
    correction_path = Path(correction_path).resolve()
    regular(correction_path)
    correction_hash = sha256(correction_path)
    corrections = object_json(correction_path)
    selected = []
    for path in sorted(source_root.glob('*/*/metadata.json')):
        if object_json(path).get('batchId') == batch:
            selected.append(JointCapture(path.parent))
    placements, effective = corrected_labels(selected, corrections, batch)
    if output.is_relative_to(source_root):
        raise ValueError('Archive must be outside the original transfer root')
    snapshots = {c.session_id: file_manifest(c.directory) for c in selected}
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for c in selected:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', c.session_id):
            raise ValueError('Unsafe session folder name')
        destination = output / c.session_id
        destination.mkdir()
        for name in SOURCE_FILES:
            regular(c.directory / name)
            shutil.copy2(c.directory / name, destination / name)
        copied = JointCapture(destination)
        c.verify()
        if file_manifest(c.directory) != snapshots[c.session_id] or file_manifest(destination) != snapshots[c.session_id]:
            raise ValueError('Source or archive copy changed during copying')
        copied.verify()
        entries.append({'session_id': c.session_id, 'folder': destination.name,
                        'source_directory': str(c.directory), 'files': snapshots[c.session_id],
                        'original_metadata': c.metadata, 'analysis_metadata': effective[c.session_id]})
    if sha256(correction_path) != correction_hash:
        raise ValueError('Correction instructions changed during archiving')
    shutil.copy2(correction_path, output / 'label-corrections.json')
    if sha256(output / 'label-corrections.json') != correction_hash:
        raise ValueError('Correction copy hash mismatch')
    for c in selected:
        c.verify()
        if file_manifest(c.directory) != snapshots[c.session_id]:
            raise ValueError('Original files changed during archiving')
    report = {'schema_version': 1, 'complete': True, 'batch_id': batch,
              'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'corrections_sha256': correction_hash, 'source_files_modified': False,
              'layout': placements, 'sessions': entries}
    write_json(output / 'archive.json', report)
    return report


def load_archive(directory):
    directory = Path(directory).resolve()
    regular(directory / 'archive.json')
    archive_hash = sha256(directory / 'archive.json')
    manifest = object_json(directory / 'archive.json')
    if manifest.get('schema_version') != 1 or manifest.get('complete') is not True:
        raise ValueError('Archive is not complete')
    regular(directory / 'label-corrections.json')
    if sha256(directory / 'label-corrections.json') != manifest['corrections_sha256']:
        raise ValueError('Archive corrections have changed')
    captures = []
    for entry in manifest['sessions']:
        folder = contained(directory, entry['folder'])
        c = JointCapture(folder)
        if (c.session_id != entry['session_id'] or file_manifest(folder) != entry['files']
                or c.metadata != entry['original_metadata']):
            raise ValueError('Archived source disagrees with archive manifest')
        captures.append(c)
    placements, effective = corrected_labels(captures, object_json(directory / 'label-corrections.json'), manifest['batch_id'])
    for entry in manifest['sessions']:
        if effective[entry['session_id']] != entry['analysis_metadata']:
            raise ValueError('Stored analysis labels disagree with correction log')
    by_id = {c.session_id: c for c in captures}
    return manifest, archive_hash, [(cond, trial, by_id[key]) for cond, trial, key in placements]


def diagnostic_feature(packet, config):
    """Shape-local packet view only; do not relax the strict replay processor."""
    if (config.tx > packet['ntx'] or config.rx > packet['nrx']
            or (config.mode != 'amplitude' and config.reference_rx > packet['nrx'])):
        return None
    a = packet['csi'][config.tx-1][config.rx-1][config.subcarrier-1]
    if config.mode == 'amplitude':
        factor = scale_factor(packet)
        return abs(a) * factor if factor is not None else None
    b = packet['csi'][config.tx-1][config.reference_rx-1][config.subcarrier-1]
    if abs(a) == 0 or abs(b) == 0:
        return None
    if config.mode == 'log_ratio':
        return math.log(abs(a)) - math.log(abs(b))
    z = a * b.conjugate()
    return math.atan2(z.imag, z.real)


def packet_rows(capture, config):
    rows = []
    for indexed, packet in zip(capture.index, capture.packets):
        if packet is None or not capture.inside(indexed['host_monotonic_ns']):
            continue
        value = diagnostic_feature(packet, config)
        rows.append({'record_index': indexed['record_index'],
                     'seconds': (indexed['host_monotonic_ns']-capture.t0)/1e9,
                     'ntx': packet['ntx'], 'nrx': packet['nrx'], 'rate': packet['rate'],
                     'value': value if value is None or math.isfinite(value) else None})
    if not rows:
        raise ValueError('No in-window CSI packets')
    return rows


def diagnostic_summary(capture, rows):
    times = [r['seconds'] for r in rows]
    intervals = [b-a for a, b in zip(times, times[1:])]
    markers = [e for e in capture.events if capture.inside(e['host_monotonic_ns'])]
    return {'session_id': capture.session_id, 'duration_seconds': capture.duration,
            'window_csi_packets': len(rows), 'shape_counts': dict(Counter(f"{r['ntx']}x{r['nrx']}" for r in rows)),
            'shape_transitions': sum((a['ntx'], a['nrx']) != (b['ntx'], b['nrx']) for a, b in zip(rows, rows[1:])),
            'rate_transitions': sum(a['rate'] != b['rate'] for a, b in zip(rows, rows[1:])),
            'first_csi_seconds': times[0], 'last_csi_seconds': times[-1],
            'max_gap_seconds': max(intervals, default=0), 'gaps_over_200ms': sum(d > .2 for d in intervals),
            'window_markers': len(markers), 'excluded_markers': len(capture.events)-len(markers),
            'rejected_markers': capture.status['rejected'], 'observed_missing_sequences': capture.status['missing'],
            'timing': capture.timing}


def panel(fig, spec, condition, trial, capture, rows, config, limits, end, compact=False):
    grid = spec.subgridspec(5, 1, height_ratios=[1, 1, 1, .55, .7], hspace=.17)
    axes = []
    for i in range(5):
        ax = fig.add_subplot(grid[i], sharex=axes[0] if axes else None)
        axes.append(ax)
        ax.set_xlim(0, end)
        ax.grid(axis='x', color='#e7e7e7', linewidth=.5)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=7 if compact else 9)
        if i != 4:
            ax.tick_params(labelbottom=False)
        if capture.duration < end:
            ax.axvspan(capture.duration, end, color='#e8e8e8', lw=0)
    count = sum(capture.inside(e['host_monotonic_ns']) for e in capture.events)
    axes[0].set_title(f'{condition} | Trial {trial} | {len(rows):,} packets | {count} markers',
                      loc='left', fontsize=10 if compact else 13)
    for ntx, ax in zip((1, 2, 3), axes):
        selected = [r for r in rows if r['ntx'] == ntx and r['nrx'] == 3 and r['value'] is not None]
        ax.scatter([r['seconds'] for r in selected], [r['value'] for r in selected],
                   s=1.5 if compact else 3, color=COLORS[ntx], rasterized=True)
        ax.set_ylim(*limits)
        ax.set_ylabel(f'{ntx} x 3', fontsize=8 if compact else 10)
        if not selected:
            ax.text(.5, .5, 'No valid samples', transform=ax.transAxes, ha='center', fontsize=7)
    markers = [(e['host_monotonic_ns']-capture.t0)/1e9 for e in capture.events if capture.inside(e['host_monotonic_ns'])]
    axes[3].vlines(markers, 0, 1, color='#404040', linewidth=.7)
    axes[3].set_ylim(0, 1.1)
    axes[3].set_yticks([])
    axes[3].set_ylabel('Marker', fontsize=8 if compact else 10)
    if not markers:
        label = 'No button events (empty-room condition)' if condition == 'Empty' else 'No accepted button events'
        axes[3].text(.5, .35, label, transform=axes[3].transAxes, ha='center', fontsize=7 if compact else 9)
    for ntx in (1, 2, 3):
        chosen = [r for r in rows if r['ntx'] == ntx and r['nrx'] == 3]
        axes[4].scatter([r['seconds'] for r in chosen], [ntx]*len(chosen),
                        s=1.5 if compact else 3, color=COLORS[ntx], rasterized=True)
    other = [r for r in rows if r['nrx'] != 3]
    if other:
        axes[4].scatter([r['seconds'] for r in other], [0]*len(other), s=3, color='#aa0044')
    axes[4].set_yticks([0, 1, 2, 3] if other else [1, 2, 3])
    axes[4].set_ylim(-.2 if other else .6, 3.4)
    axes[4].set_ylabel('TX streams', fontsize=8 if compact else 10)
    axes[4].set_xlabel('Seconds from session common T0', fontsize=8 if compact else 10)
    return axes


def figure_for(items, config, limits, end):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    overview = len(items) > 1
    fig = Figure(figsize=(15, 18) if overview else (12, 10))
    FigureCanvasAgg(fig)
    grid = fig.add_gridspec(3, 2, hspace=.38, wspace=.2) if overview else fig.add_gridspec(1, 1)
    for i, (condition, trial, capture, rows) in enumerate(items):
        panel(fig, grid[i//2, i%2] if overview else grid[0, 0], condition, trial, capture, rows,
              config, limits, end, compact=overview)
    fig.suptitle(f'{NAMES[config.mode]} | shape-separated CSI + manual markers', fontsize=15, y=.983)
    fig.text(.08, .961 if overview else .94, f'Stream {config.tx} / RX {config.rx} / reference RX {config.reference_rx} / group {config.subcarrier}'
             f' | {UNITS[config.mode]} | shared feature scale | packet samples, no filtering', fontsize=9)
    fig.text(.08, .035, 'Shapes are separate effective channels, not interchangeable physical TX antennas. No lines bridge samples or shape changes.', fontsize=8)
    fig.text(.08, .02, 'Host-receipt alignment only. Manual events are not physiological ground truth; no rate estimate or causal attribution is made.', fontsize=8)
    fig.subplots_adjust(left=.08, right=.975, bottom=.085 if overview else .105,
                        top=.925 if overview else .89)
    return fig


def diagnose_archive(directory, output, modes=MODES):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError('Results must be outside the raw archive')
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Select unique feature modes')
    configs = [FeatureConfig(mode) for mode in modes]
    manifest, archive_hash, layout = load_archive(directory)
    output.mkdir(parents=True, exist_ok=False)
    end = max(c.duration for _, _, c in layout)
    summaries, image_sections = [], []
    for config in configs:
        items = [(cond, trial, c, packet_rows(c, config)) for cond, trial, c in layout]
        limits = shared_limits([[r['value'] for r in rows if r['value'] is not None and r['nrx'] == 3]
                                for _, _, _, rows in items])
        outputs = [('overview', items)] + [(f'{cond.lower()}-trial-{trial}', [item]) for item in items for cond, trial, _, _ in [item]]
        links = []
        for slug, chosen in outputs:
            stem = f'{config.mode}-{slug}'
            fig = figure_for(chosen, config, limits, end)
            for ext in ('png', 'pdf', 'svg'):
                fig.savefig(output / f'{stem}.{ext}', dpi=160)
            fig.clear()
            links.append(f'<h3>{html.escape(slug)}</h3><a href="{stem}.png"><img loading="lazy" src="{stem}.png" alt="{stem}"></a>'
                         f'<p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a></p>')
        image_sections.append(f'<section><h2>{NAMES[config.mode]}</h2>'+''.join(links)+'</section>')
        for condition, trial, c, rows in items:
            write_json(output / f'{config.mode}-{condition.lower()}-trial-{trial}-samples.json',
                       {'feature': config.__dict__, 'window': '[t0_ns, t1_ns)', 'packets': rows})
            if config.mode == configs[0].mode:
                summaries.append({'condition': condition, 'trial': trial, **diagnostic_summary(c, rows)})
    summary_rows = ''.join('<tr>'+''.join(f'<td>{html.escape(str(s[k]))}</td>' for k in
                            ('condition', 'trial', 'window_csi_packets', 'window_markers', 'shape_transitions'))+'</tr>' for s in summaries)
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>Shape-separated CSI diagnostics</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:32px auto;padding:0 20px;color:#222}img{width:100%}'
        'td,th{padding:8px;border-bottom:1px solid #ddd;text-align:left}p{line-height:1.6}</style>'
        f'<h1>{html.escape(manifest["batch_id"])} — shape-separated CSI diagnostics</h1><p>{LIMITATION}</p>'
        '<p>Three feature rows separate 1x3, 2x3 and 3x3 packets; button events and stream-count observations share the same host time axis. '
        'Missing dots are not interpolated. NRX other than 3 is shown as state 0 and omitted from feature rows. '
        'Separate sessions share axis limits, not an absolute clock or breath phase. '
        'The same stream index need not represent the same physical antenna across shapes or rate changes. '
        'No filtering, normalization, channel search or automatic cause diagnosis is performed.</p>'
        '<p>Empty-room labels use the separately confirmed correction log; all original metadata and receipt bytes are preserved. '
        'The existing fixed-3x3 analysis and replay remain unchanged.</p>'
        '<table><tr><th>Condition</th><th>Trial</th><th>CSI packets</th><th>Markers</th><th>Shape changes</th></tr>'
        +summary_rows+'</table>'+''.join(image_sections), encoding='utf-8')
    checked, final_hash, _ = load_archive(directory)
    if archive_hash != final_hash or checked != manifest:
        raise ValueError('Archive changed during diagnostics')
    report = {'schema_version': 1, 'complete': True, 'archive_directory': str(directory),
              'archive_sha256': archive_hash, 'corrections_sha256': manifest['corrections_sha256'],
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'summaries': summaries,
              'features': [c.__dict__ for c in configs], 'raw_data_modified': False,
              'limitation': LIMITATION, 'diagnostic_only': True,
              'implementation_hashes': [{'path': str(p.relative_to(Path(__file__).parent)), 'sha256': sha256(p)}
                                        for p in sorted(Path(__file__).parent.rglob('*.py'))],
              'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / 'diagnostics-report.json', report)
    return report
