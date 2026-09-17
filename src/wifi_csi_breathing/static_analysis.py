"""Shared-scale pilot figures; labels control layout, never signal processing."""
from dataclasses import asdict
from datetime import datetime, timezone
from itertools import groupby
from pathlib import Path
import csv
import html
import math
import platform

import numpy as np
from scipy import signal
import scipy

from .analysis import analyze, write_csv, write_json
from .data import MODES, sha256
from .processing import Point, ProcessingConfig


CONDITIONS = ('Empty', 'F0', 'R90')
COLORS = ('#666666', '#0072B2', '#D55E00')
NAMES = {'amplitude': 'Scaled CSI amplitude', 'log_ratio': 'Log amplitude ratio',
         'phase_difference': 'Inter-RX phase difference'}
UNITS = {'amplitude': 'Relative amplitude', 'log_ratio': 'Log ratio (unitless)',
         'phase_difference': 'Phase difference (rad)'}
PSD_UNITS = {'amplitude': 'Amplitude squared / Hz', 'log_ratio': 'Log-ratio squared / Hz',
             'phase_difference': 'Rad squared / Hz'}
KINDS = ('feature', 'filtered', 'normalized', 'spectrum', 'periodicity')


def pilot_layout(datasets):
    """Require two distinct trials per condition; preserve actual trial numbers."""
    if len(datasets) != 6 or len({dataset.id for dataset in datasets}) != 6:
        raise ValueError('Static pilot comparison requires six unique recording IDs')
    groups = {condition: [] for condition in CONDITIONS}
    for dataset in datasets:
        metadata = dataset.verify()
        activity = str(metadata.get('activityLabel', '')).lower()
        position = str(metadata.get('positionId', '')).upper()
        if activity == 'empty':
            condition = 'Empty'
        elif activity == 'normal_breathing' and position.endswith(('_F0', '_R90')):
            condition = position.rsplit('_', 1)[1]
        else:
            raise ValueError(f'Unsupported pilot condition: {dataset.id}')
        trial = metadata.get('trialNumber')
        if isinstance(trial, bool) or not isinstance(trial, int) or trial < 1:
            raise ValueError('Pilot trial numbers must be positive integers')
        groups[condition].append((trial, dataset, metadata))
    if any(len(group) != 2 or len({item[0] for item in group}) != 2 for group in groups.values()):
        raise ValueError('Static pilot comparison requires two distinct trials per condition (six recordings)')
    return [(condition, trial, dataset, metadata)
            for condition in CONDITIONS for trial, dataset, metadata in sorted(groups[condition], key=lambda item: item[0])]


def segment_arrays(points, end, config):
    """Split first, then clip. Settling is relative to each original segment start."""
    segments = []
    for segment_id, group in groupby(points, key=lambda point: point.segment):
        group = list(group)
        start = group[0].time
        chosen = [point for point in group if 0 <= point.time <= end + 1e-9
                  and point.available_at <= end + 1e-9]
        if not chosen:
            continue
        t = np.array([p.time for p in chosen])
        segments.append({'segment': segment_id, 'time': t,
                         'raw': np.array([p.raw for p in chosen]),
                         'filtered': np.array([p.filtered for p in chosen]),
                         'settled': t - start >= config.warmup_seconds - 1e-9})
    return segments


def spectrum(segments, config):
    """Equal 30 s Hann periodograms, 50% overlap, no cross-segment windows.

    Use pre-bandpass smoothed features, with per-window linear detrending.
    Every window has identical resolution; no zero padding or short-window fallback.
    """
    n = round(config.window_seconds * config.sample_hz)
    step = n // 2
    powers = []
    frequencies = np.fft.rfftfreq(n, 1 / config.sample_hz)
    for part in segments:
        values = part['raw'][part['settled']]
        for start in range(0, len(values) - n + 1, step):
            _, power = signal.periodogram(values[start:start + n], fs=config.sample_hz,
                                           window='hann', detrend='linear', scaling='density')
            powers.append(power)
    return frequencies, np.mean(powers, axis=0) if powers else None, len(powers)


def normalization(segments):
    chunks = [part['filtered'][part['settled']] for part in segments]
    values = np.concatenate(chunks) if chunks else np.array([])
    if not len(values):
        return None, None
    mean, std = float(values.mean()), float(values.std())
    return (mean, std) if std > 1e-12 else (mean, None)


def shared_limits(arrays, positive=False):
    finite = [np.asarray(a)[np.isfinite(a)] for a in arrays if a is not None]
    finite = [a for a in finite if a.size]
    if not finite:
        return (0, 1) if positive else (-1, 1)
    lo, hi = min(float(a.min()) for a in finite), max(float(a.max()) for a in finite)
    if positive:
        return 0, max(hi * 1.08, 1e-12)
    margin = max((hi - lo) * 0.08, max(abs(lo), abs(hi), 1) * 1e-6)
    return lo - margin, hi + margin


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def make_item(summary, folder, condition, trial, end, config):
    points = [Point(float(r['time']), float(r['available_at']), float(r['raw']),
                    float(r['filtered']), int(r['segment'])) for r in read_rows(folder / 'waveform.csv')]
    segments = segment_arrays(points, end, config)
    frequency, power, windows = spectrum(segments, config)
    mean, std = normalization(segments)
    estimates = [r for r in read_rows(folder / 'estimates.csv') if float(r['available_at']) <= end + 1e-9]
    observations = [r for r in read_rows(folder / 'observations.csv') if float(r['time']) <= end + 1e-9]
    times = np.array([float(r['time']) for r in observations])
    dt = np.diff(times)
    valid = sum(bool(r['value']) and math.isfinite(float(r['value'])) for r in observations)
    settled_seconds = sum(int(part['settled'].sum()) for part in segments) / config.sample_hz
    candidates = [float(r['candidate_bpm']) for r in estimates if r['candidate_bpm']]
    peak = band_fraction = None
    if power is not None and float(power[1:].sum()) > 1e-20:
        band = (frequency >= config.low_hz - 1e-9) & (frequency <= config.high_hz + 1e-9)
        peak = float(frequency[band][np.argmax(power[band])] * 60)
        band_fraction = float(power[band].sum() / power[1:].sum())
    metrics = {
        'dataset_id': summary['dataset_id'], 'condition': condition, 'trial': trial,
        'mode': summary['feature']['mode'], 'original_status': summary['input']['original_status'],
        'full_span_seconds': summary['input']['span_seconds'],
        'full_packet_count': summary['input']['packets'],
        'full_observed_hz': summary['input']['observed_hz'],
        'comparison_end_seconds': end, 'comparison_packets': len(observations),
        'comparison_observed_hz': float((len(times) - 1) / (times[-1] - times[0])) if len(times) > 1 and times[-1] > times[0] else None,
        'invalid_feature_packets': len(observations) - valid,
        'duplicate_timestamps': int((dt == 0).sum()),
        'long_gap_count': int((dt > config.max_gap_seconds).sum()),
        'max_interval_ms': float(dt.max() * 1000) if len(dt) else None,
        'dimension_changes': sum(r['reason'] == 'dimension_change' for r in observations),
        'segments': len(segments), 'settled_grid_seconds': settled_seconds,
        'settled_coverage_fraction': min(settled_seconds / end, 1.0),
        'filtered_std': std, 'normalization_mean': mean,
        'psd_windows': windows, 'psd_bin_hz': config.sample_hz / round(config.window_seconds * config.sample_hz),
        'input_band_peak_cycles_min': peak, 'input_band_power_fraction': band_fraction,
        'eligible_candidate_windows': len(estimates), 'periodic_candidate_windows': len(candidates),
        'candidate_median_cycles_min': float(np.median(candidates)) if candidates else None,
    }
    return {'metrics': metrics, 'segments': segments, 'frequency': frequency,
            'power': power, 'estimates': estimates, 'mean': mean, 'std': std}


def shade_segments(ax, parts, end, config):
    """Mark unavailable time and filter settling; never join disjoint traces."""
    cursor = 0.0
    for part in parts:
        t = part['time']
        if t[0] > cursor:
            ax.axvspan(cursor, t[0], color='#d5d5d5', lw=0, zorder=0)
        stop = min(float(t[-1]) + 1 / config.sample_hz, end)
        unsettled = t[~part['settled']]
        if len(unsettled):
            ax.axvspan(float(t[0]), min(float(unsettled[-1]) + 1 / config.sample_hz, stop),
                       facecolor='#f5f5f5', edgecolor='#c7c7c7', hatch='////', lw=0, zorder=0)
        cursor = stop
    if cursor < end:
        ax.axvspan(cursor, end, color='#d5d5d5', lw=0, zorder=0)


def draw_figure(items, mode, kind, end, config, feature):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(10.4, 7.9))
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 2, sharex=True, sharey=True)
    values = []
    for item in items:
        for part in item['segments']:
            if kind == 'feature':
                values.append(part['raw'])
            elif kind == 'filtered':
                values.append(part['filtered'][part['settled']])
            elif kind == 'normalized' and item['std'] is not None:
                values.append((part['filtered'][part['settled']] - item['mean']) / item['std'])
        if kind == 'spectrum' and item['power'] is not None:
            values.append(item['power'][item['frequency'] <= 1.0])
    limits = (0, 36) if kind == 'periodicity' else shared_limits(values, positive=kind == 'spectrum')
    subtitles = {'feature': 'Smoothed feature before bandpass', 'filtered': 'Causal 0.1–0.5 Hz component',
                 'normalized': 'Normalized component — shape only', 'spectrum': 'Pre-bandpass feature spectrum',
                 'periodicity': 'Trailing-window periodicity diagnostics'}
    fig.suptitle(f'{NAMES[mode]} | {subtitles[kind]}', fontsize=14, y=0.974)
    fig.text(0.5, 0.937, f"TX {feature['tx']} / RX {feature['rx']} / reference RX {feature['reference_rx']} / group {feature['subcarrier']}"
             f' · common interval 0–{end:g} s · fixed parameters', ha='center', fontsize=9, color='#444444')
    for i, (ax, item) in enumerate(zip(axes.flat, items)):
        m, color = item['metrics'], COLORS[i // 2]
        label = f"({chr(97 + i)}) {m['condition']} · Trial {m['trial']}"
        if m['original_status'] != 'completed':
            label += ' [original: failed]'
        ax.set_title(label, loc='left', fontsize=10, pad=6)
        ax.set_ylim(*limits)
        ax.grid(True, color='#e7e7e7', linewidth=0.45)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(labelsize=8)
        if kind == 'spectrum':
            ax.set_xlim(0, 1)
            ax.axvspan(config.low_hz, config.high_hz, color='#eaf2f7', zorder=0)
            if item['power'] is not None:
                ax.plot(item['frequency'], item['power'], color=color, lw=1)
                peak = m['input_band_peak_cycles_min']
                text = f"{m['psd_windows']} windows"
                if peak is not None:
                    text += f' · band peak {peak:.1f} cycles/min'
            else:
                text = 'Insufficient continuous data for a 30 s spectrum'
            ax.text(0.98, 0.96, text, transform=ax.transAxes, ha='right', va='top', fontsize=7.5)
        else:
            ax.set_xlim(0, end)
            shade_segments(ax, item['segments'], end, config)
            if kind == 'periodicity':
                for _, group in groupby(item['estimates'], key=lambda r: r['segment']):
                    group = list(group)
                    times = [float(r['time']) for r in group]
                    for field, linestyle, linecolor in [('spectral_peak_bpm', '-', color),
                                                        ('autocorrelation_bpm', '--', '#888888')]:
                        ys = [float(r[field]) if r[field] else np.nan for r in group]
                        ax.plot(times, ys, linestyle, color=linecolor, lw=0.8)
                    good = [r for r in group if r['candidate_bpm']]
                    ax.scatter([float(r['time']) for r in good], [float(r['candidate_bpm']) for r in good],
                               s=12, color=color, zorder=3)
                ax.text(0.98, 0.96, f"Candidates: {m['periodic_candidate_windows']}/{m['eligible_candidate_windows']}",
                        transform=ax.transAxes, ha='right', va='top', fontsize=8)
            else:
                for part in item['segments']:
                    mask = np.ones(len(part['time']), dtype=bool) if kind == 'feature' else part['settled']
                    ys = part['raw'] if kind == 'feature' else part['filtered']
                    if kind == 'normalized':
                        if item['std'] is None:
                            continue
                        ys = (ys - item['mean']) / item['std']
                    ax.plot(part['time'][mask], ys[mask], color=color, lw=0.8)
                if kind == 'normalized' and item['std'] is None:
                    ax.text(0.5, 0.5, 'No nonconstant settled signal', transform=ax.transAxes, ha='center', fontsize=8)
        if i >= 4:
            ax.set_xlabel('Frequency (Hz)' if kind == 'spectrum' else 'Time from first CSI packet (s)', fontsize=9)
        if i % 2 == 0:
            unit = PSD_UNITS[mode] if kind == 'spectrum' else 'Cycles/min' if kind == 'periodicity' else 'Standard deviations' if kind == 'normalized' else UNITS[mode]
            ax.set_ylabel(unit, fontsize=9)
    footnotes = {
        'feature': 'Shared axes; independent recordings, not phase-aligned. Phase unwrap is segment-local; offsets are not displacement.',
        'filtered': 'Shared axes; first 10 s of every segment omitted. Filtered oscillation is not proof of breathing.',
        'normalized': 'Per-recording mean/SD from settled samples only; relative amplitude differences are deliberately removed.',
        'spectrum': '30 s Hann windows, 50% overlap; segment-safe mean PSD. Blue band: 0.1–0.5 Hz. Bin spacing: 2 cycles/min.',
        'periodicity': 'Solid: spectral peak; dashed: autocorrelation peak; dots: heuristic agreement. Overlapping windows are not independent.',
    }
    fig.text(0.08, 0.058, footnotes[kind], fontsize=8)
    fig.text(0.08, 0.032, 'Research preview; no reference breathing measurement.' +
             ('' if kind == 'spectrum' else ' Gray: unavailable interval; hatched: provisional filter settling.'), fontsize=8, color='#555555')
    fig.subplots_adjust(left=0.095, right=0.975, bottom=0.13, top=0.875, hspace=0.43, wspace=0.17)
    return fig


def render_all(items, destination, end, config, feature):
    import matplotlib
    style = {'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.linewidth': 0.7,
             'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white',
             'pdf.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False}
    output = []
    with matplotlib.rc_context(style):
        for mode in MODES:
            chosen = [item for item in items if item['metrics']['mode'] == mode]
            for kind in KINDS:
                fig = draw_figure(chosen, mode, kind, end, config, feature)
                name = f'{mode}-{kind}'
                for extension in ('png', 'pdf', 'svg'):
                    file = destination / f'{name}.{extension}'
                    fig.savefig(file, dpi=300)
                    output.append(file)
                fig.clear()
    return output


def report_text(items, end):
    lines = ['# Six-session static comparison', '',
             'Research preview, not a validated breathing or occupancy detector. No independent breathing reference is available.', '',
             f'Common display and analysis interval: 0–{end:g} s from each recording\'s first CSI packet. '
             'Records are not synchronized or breath-phase aligned. Original full-span exports are retained in `numerical/`.', '',
             '## Recording quality', '',
             '| Condition / trial | Original status | Full span (s) | Full packets | Common packet rate (Hz) | Shape changes | Max interval (ms) |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    def number(value, places=2):
        return '—' if value is None else f'{value:.{places}f}'
    for item in items:
        m = item['metrics']
        if m['mode'] == 'amplitude':
            lines.append(f"| {m['condition']} / {m['trial']} | {m['original_status']} | {m['full_span_seconds']:.3f} | "
                         f"{m['full_packet_count']} | {number(m['comparison_observed_hz'])} | {m['dimension_changes']} | {number(m['max_interval_ms'])} |")
    lines += ['', 'The recovered first empty recording keeps its original failed status and approximately 125 s span. '
              'Only the common interval is compared; no source is trimmed or relabeled. Packet rate is not packet-loss rate. '
              'Settled coverage below measures available processed samples after exclusions, not probability of usable respiration.', '']
    for mode in MODES:
        lines += [f'## {NAMES[mode]}', '',
                  '| Condition / trial | Settled coverage | Filtered SD | PSD windows | Input-band peak (cycles/min) | Input-band power fraction | Candidate / eligible windows |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for item in items:
            m = item['metrics']
            if m['mode'] == mode:
                lines.append(f"| {m['condition']} / {m['trial']} | {m['settled_coverage_fraction']:.1%} | {number(m['filtered_std'], 4)} | "
                             f"{m['psd_windows']} | {number(m['input_band_peak_cycles_min'], 1)} | {number(m['input_band_power_fraction'], 3)} | "
                             f"{m['periodic_candidate_windows']} / {m['eligible_candidate_windows']} |")
        lines += ['', 'Band peaks are descriptive, including in empty-room records; they are not confirmed respiratory rates. '
                  'Different channels/features use different units. Phase offsets across reset segments are arbitrary.', '']
        for kind in KINDS:
            name = f'{mode}-{kind}'
            lines += [f'### {kind.capitalize()}', '', f'![{name}](figures/{name}.png)', '',
                      f'[Vector PDF](figures/{name}.pdf) · [Vector SVG](figures/{name}.svg)', '']
    lines += ['## Interpretation and limitations', '',
              '- All six panels share limits within each figure; no panel is independently autoscaled.',
              '- Normalization is for shape inspection only. Use the non-normalized filtered figures for relative magnitude comparisons.',
              '- Pre-bandpass spectra use identical 30 s windows, not concatenated segments. Short segments are excluded, not zero-padded.',
              '- Candidate gates, channel selection, and the streaming processor are unchanged from the replay baseline.',
              '- The 10 s settling exclusion is provisional. Residual filter transients, motion, or environmental periodicity may remain.',
              '- A plausible peak or a larger filtered SD does not establish breathing, occupancy, or superiority of an orientation.',
              '- No statistical significance, clinical normality, apnea decision, or rate accuracy can be inferred from this six-record pilot.',
              '- Numerical exports and source metadata are local research artifacts; do not publish them without a privacy review.', '']
    return '\n'.join(lines)


def gallery(items, end):
    """Small self-contained index; figure files are local and no network is used."""
    sections = []
    for mode in MODES:
        images = ''.join(f'<h3>{kind.capitalize()}</h3><a href="figures/{mode}-{kind}.pdf">PDF</a> · '
                         f'<a href="figures/{mode}-{kind}.svg">SVG</a><br>'
                         f'<img loading="lazy" src="figures/{mode}-{kind}.png" alt="{mode} {kind}">'
                         for kind in KINDS)
        sections.append(f'<section id="{mode}"><h2>{NAMES[mode]}</h2>{images}</section>')
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(value))}</td>' for value in (
        item['metrics']['condition'], item['metrics']['trial'], item['metrics']['mode'],
        f"{item['metrics']['settled_coverage_fraction']:.1%}",
        f"{item['metrics']['periodic_candidate_windows']}/{item['metrics']['eligible_candidate_windows']}")) + '</tr>' for item in items)
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            '<title>Six-session CSI static comparison</title><style>body{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 18px;color:#222;background:white}'
            'img{width:100%;height:auto}section{margin-top:48px}table{border-collapse:collapse}td,th{padding:6px 14px;border-bottom:1px solid #ddd;text-align:left}'
            'a{color:#0072B2}p{line-height:1.6}</style><h1>Six-session CSI static comparison</h1>'
            f'<p>Common interval: 0–{end:g} s. Three conditions × two repeats. Shared axes within each figure.</p>'
            '<p><strong>Research preview, not confirmed breathing.</strong> No ground truth; band peaks may also occur in empty-room data. '
            'Gray denotes unavailable time, hatching denotes provisional settling. Normalized plots compare shape, not strength. '
            'The recovered empty trial retains its original failed status.</p>'
            '<p><a href="SUMMARY.md">Full methods and metric summary</a> · <a href="quality.csv">Quality CSV</a> · '
            '<a href="static-report.json">Provenance JSON</a></p><nav>'
            + ' · '.join(f'<a href="#{mode}">{NAMES[mode]}</a>' for mode in MODES)
            + '</nav><h2>Processed coverage and periodic candidates</h2><p>Candidate counts refer to overlapping windows, not independent trials or accuracy.</p>'
            '<table><tr><th>Condition</th><th>Trial</th><th>Feature</th><th>Settled coverage</th><th>Candidate/eligible</th></tr>'
            + rows + '</table>' + ''.join(sections) + '</html>')


def static_analyze(datasets, output, end=120.0, tx=1, rx=1, reference_rx=2, subcarrier=15):
    if not math.isfinite(end) or end < 40:
        raise ValueError('Comparison duration must be finite and at least 40 seconds')
    # Validate settings and layout before creating output directories.
    from .data import FeatureConfig
    for mode in MODES:
        FeatureConfig(mode, tx, rx, reference_rx, subcarrier)
    layout = pilot_layout(datasets)
    output = Path(output).resolve()
    if any(output.is_relative_to(dataset.directory.resolve()) for dataset in datasets):
        raise ValueError('Output must not be inside a raw recording')
    output.mkdir(parents=True, exist_ok=False)
    config = ProcessingConfig()
    baseline = analyze([entry[2] for entry in layout], output / 'numerical', MODES, tx, rx, reference_rx, subcarrier)
    placement = {dataset.id: (condition, trial) for condition, trial, dataset, _ in layout}
    items = [make_item(summary, output / 'numerical' / summary['output_folder'],
                       *placement[summary['dataset_id']], end, config) for summary in baseline['results']]
    destination = output / 'figures'
    destination.mkdir()
    files = render_all(items, destination, end, config, baseline['results'][0]['feature'])
    metrics = [item['metrics'] for item in items]
    write_csv(output / 'quality.csv', metrics, list(metrics[0]))
    psd_rows = []
    for item in items:
        if item['power'] is not None:
            for hz, power in zip(item['frequency'], item['power']):
                psd_rows.append({'dataset_id': item['metrics']['dataset_id'], 'mode': item['metrics']['mode'],
                                 'frequency_hz': float(hz), 'power_density': float(power)})
    write_csv(output / 'spectra.csv', psd_rows, ('dataset_id', 'mode', 'frequency_hz', 'power_density'))
    (output / 'SUMMARY.md').write_text(report_text(items, end), encoding='utf-8')
    (output / 'index.html').write_text(gallery(items, end), encoding='utf-8')
    for dataset in datasets:
        dataset.verify()
    import matplotlib
    artifact_paths = sorted(path for path in output.rglob('*') if path.is_file())
    report = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'software_version': baseline['software_version'],
              'runtime': {'python': platform.python_version(), 'numpy': np.__version__,
                          'scipy': scipy.__version__, 'matplotlib': matplotlib.__version__},
              'implementation_hashes': [{'path': str(path.relative_to(Path(__file__).parent)),
                                         'sha256': sha256(path)}
                                        for path in sorted(Path(__file__).parent.rglob('*.py'))],
              'comparison_end_seconds': end, 'processing': asdict(config),
              'psd_method': 'Mean 30 s Hann periodograms, linear detrend, 50% overlap, settled pre-bandpass samples, no cross-segment windows',
              'normalization': 'Per-recording population mean/SD of settled filtered samples in common interval; shape only',
              'feature': {k: v for k, v in baseline['results'][0]['feature'].items() if k != 'mode'},
              'raw_data_modified': False, 'physiological_validation': False,
              'source_records': [{'dataset_id': r['dataset_id'], 'source_hashes': r['source_hashes'],
                                  'summary': 'numerical/' + r['output_folder'] + '/summary.json'}
                                 for r in baseline['results'] if r['feature']['mode'] == 'amplitude'],
              'metrics': metrics, 'figures': [str(path.relative_to(output)) for path in files],
              'artifacts': [{'path': str(path.relative_to(output)), 'sha256': sha256(path)} for path in artifact_paths]}
    # Only this final marker indicates that the entire static analysis completed.
    write_json(output / 'static-report.json', report)
    return report
