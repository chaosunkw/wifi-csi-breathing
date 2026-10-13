"""Read-only, retrospective views of all 30 Intel 5300 subcarrier groups.

These packet diagnostics do not change the causal, fixed-3x3 replay pipeline.
No interpolation, respiratory filtering, channel ranking or rate estimation.
"""
from datetime import datetime, timezone
from pathlib import Path
import html
import math

import numpy as np

from .analysis import write_json
from .batch_diagnostics import load_archive
from .data import FeatureConfig, MODES, sha256
from .joint import LIMITATION
from .static_analysis import NAMES, UNITS
from .vendor.intel5300 import scale_factor

DEFAULT_MODES = ('amplitude', 'phase_difference')
GROUPS = np.arange(1, 31)


def all_features(packet, config):
    """Return 30 packet features using the same definitions as single-group views."""
    if (config.tx > packet['ntx'] or config.rx > packet['nrx']
            or (config.mode != 'amplitude' and config.reference_rx > packet['nrx'])):
        return np.full(30, np.nan)
    a = np.asarray(packet['csi'][config.tx - 1][config.rx - 1], dtype=complex)
    if config.mode == 'amplitude':
        factor = scale_factor(packet)
        return np.abs(a) * factor if factor is not None else np.full(30, np.nan)
    b = np.asarray(packet['csi'][config.tx - 1][config.reference_rx - 1], dtype=complex)
    result = np.full(30, np.nan)
    valid = (np.abs(a) > 0) & (np.abs(b) > 0)
    if config.mode == 'log_ratio':
        result[valid] = np.log(np.abs(a[valid])) - np.log(np.abs(b[valid]))
    else:
        result[valid] = np.angle(a[valid] * b[valid].conjugate())
    result[~np.isfinite(result)] = np.nan
    return result


def extract(capture, config, max_gap=.2):
    """Keep the original session axis and break lines at every known boundary."""
    times, shapes, rates, records, values, segments = [], [], [], [], [], []
    previous = None
    segment = 0
    for indexed, packet in zip(capture.index, capture.packets):
        if packet is None or not capture.inside(indexed['host_monotonic_ns']):
            continue
        stamp = indexed['host_monotonic_ns']
        shape = (packet['ntx'], packet['nrx'])
        identity = (shape, packet['rate'])
        record = indexed['record_index']
        if (previous is None or identity != previous[1] or record != previous[2] + 1
                or not 0 < stamp - previous[0] <= round(max_gap * 1e9)):
            segment += 1
        times.append((stamp - capture.t0) / 1e9)
        shapes.append(shape)
        rates.append(packet['rate'])
        records.append(record)
        segments.append(segment)
        values.append(all_features(packet, config))
        previous = (stamp, identity, record)
    if not times:
        raise ValueError('No CSI packets in the recording window')
    return dict(times=np.asarray(times), shapes=np.asarray(shapes),
                rates=np.asarray(rates), record_indices=np.asarray(records),
                segments=np.asarray(segments), raw=np.asarray(values),
                markers=np.asarray([(e['host_monotonic_ns'] - capture.t0) / 1e9
                                    for e in capture.events if capture.inside(e['host_monotonic_ns'])]))


def centered_features(data, mode):
    """Whole-session baseline removal by shape and group, never a causal filter.

Phase uses a circular mean and wrapped residual, not linear mean or unwrapping.
Undefined circular centers remain NaN. No standard-deviation normalization.
"""
    raw = data['raw']
    centered = np.full(raw.shape, np.nan)
    centers = np.full((3, 30), np.nan)
    resultant = np.full((3, 30), np.nan)
    for ntx in (1, 2, 3):
        selected = (data['shapes'][:, 0] == ntx) & (data['shapes'][:, 1] == 3)
        for k in range(30):
            mask = selected & np.isfinite(raw[:, k])
            if not mask.any():
                continue
            vals = raw[mask, k]
            if mode == 'phase_difference':
                vector = np.mean(np.exp(1j * vals))
                resultant[ntx - 1, k] = abs(vector)
                if abs(vector) < 1e-8:
                    continue
                center = np.angle(vector)
                centered[mask, k] = np.angle(np.exp(1j * (vals - center)))
            else:
                center = np.median(vals)
                centered[mask, k] = vals - center
            centers[ntx - 1, k] = center
    return centered, centers, resultant


def packet_cells(times, duration, cell_seconds=.05):
    """Packet glyph cells capped in real time, with explicit unpainted gaps.

Neighbor midpoints clip each cell, even when neighbors have a different shape.
Duplicate timestamps get zero width rather than fabricated ordering in time.
"""
    times = np.asarray(times, dtype=float)
    if not len(times) or np.any(np.diff(times) < 0):
        raise ValueError('Expected nonempty, nondecreasing packet times')
    mid = (times[:-1] + times[1:]) / 2
    left = np.maximum(times - cell_seconds / 2, np.r_[0., mid])
    right = np.minimum(times + cell_seconds / 2, np.r_[mid, duration])
    duplicates = np.r_[False, np.diff(times) == 0] | np.r_[np.diff(times) == 0, False]
    left[duplicates] = times[duplicates]
    right[duplicates] = times[duplicates]
    edges = np.unique(np.r_[0., left, right, duration])
    # Each unpainted bin retains -1. Width is display support, not an observation.
    owner = np.full(len(edges) - 1, -1, dtype=int)
    for i, (a, b) in enumerate(zip(left, right)):
        if b > a:
            owner[np.searchsorted(edges, a):np.searchsorted(edges, b)] = i
    return edges, owner


def heat_values(data, owner, ntx, view):
    matrix = np.full((30, len(owner)), np.nan)
    available = np.flatnonzero(owner >= 0)
    packet = owner[available]
    selected = (data['shapes'][packet, 0] == ntx) & (data['shapes'][packet, 1] == 3)
    matrix[:, available[selected]] = data[view][packet[selected]].T
    return np.ma.masked_invalid(matrix)


def line_arrays(data, ntx, view):
    """Insert NaN separators without discarding a segment's first observation."""
    boundaries = np.r_[False, np.diff(data['segments']) != 0]
    indices = np.flatnonzero(boundaries)
    times = np.insert(data['times'], indices, np.nan)
    values = data[view].copy()
    values[(data['shapes'][:, 0] != ntx) | (data['shapes'][:, 1] != 3)] = np.nan
    # A wrapped jump in any group conservatively breaks all group lines.
    if data['mode'] == 'phase_difference':
        jumps = np.abs(np.diff(values, axis=0)) > math.pi
        extra = np.flatnonzero(jumps.any(axis=1)) + 1
        indices = np.union1d(indices, extra)
        times = np.insert(data['times'], indices, np.nan)
    return times, np.insert(values, indices, np.nan, axis=0)


def limits_for(items, mode, percentile):
    raw = np.concatenate([d['raw'][(d['shapes'][:, 1] == 3)].ravel() for d in items])
    raw = raw[np.isfinite(raw)]
    if mode == 'phase_difference':
        raw_limits = [-math.pi, math.pi]
    elif not len(raw):
        raw_limits = [0., 1.]
    else:
        lo, hi = float(raw.min()), float(raw.max())
        raw_limits = [lo, hi] if hi > lo else [lo - .5, hi + .5]
    residual = []
    for ntx in (1, 2, 3):
        vals = np.concatenate([d['centered'][(d['shapes'][:, 0] == ntx)
                                           & (d['shapes'][:, 1] == 3)].ravel() for d in items])
        vals = np.abs(vals[np.isfinite(vals)])
        bound = math.pi if mode == 'phase_difference' else float(np.percentile(vals, percentile)) if len(vals) else 1.
        residual.append([-bound, bound] if bound > 1e-12 else [-1., 1.])
    return {'raw': raw_limits, 'centered_by_shape': residual,
            'centered_color_percentile': percentile if mode != 'phase_difference' else 100.}


def new_figure(size):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=size)
    FigureCanvasAgg(fig)
    return fig


def caption(fig, text):
    fig.text(.055, .033, text, fontsize=8)
    fig.text(.055, .017, 'Host-receipt timeline. Manual button events are approximate references, not physiological ground truth.', fontsize=8)


def marker_axis(ax, data, end):
    ax.vlines(data['markers'], 0, 1, color='#383838', linewidth=.7)
    if not len(data['markers']):
        ax.text(.5, .4, 'No button events', transform=ax.transAxes, ha='center', fontsize=9)
    ax.set(xlim=(0, end), ylim=(0, 1.1), yticks=[], ylabel='Marker', xlabel='Time from common T0 (s)')
    ax.spines[['top', 'right']].set_visible(False)


def draw_heat(ax, data, ntx, view, limits, end):
    from matplotlib import colormaps
    cmap = 'twilight' if view == 'raw' and data['mode'] == 'phase_difference' else 'viridis' if view == 'raw' else 'RdBu_r'
    palette = colormaps[cmap].copy()
    palette.set_bad('#d9dde2')
    ax.set_facecolor('#d9dde2')
    mesh = ax.pcolormesh(data['edges'], np.arange(.5, 31.5),
                         heat_values(data, data['owner'], ntx, view),
                         cmap=palette, vmin=limits[0], vmax=limits[1], shading='flat', rasterized=True)
    ax.set(xlim=(0, end), ylim=(.5, 30.5), yticks=[1, 15, 30], ylabel=f'{ntx} x 3\nGroup')
    if not np.isfinite(data[view][(data['shapes'][:, 0] == ntx) & (data['shapes'][:, 1] == 3)]).any():
        ax.text(.5, .5, 'No valid samples', transform=ax.transAxes, ha='center', fontsize=9)
    return mesh


def heatmap_figure(data, limits, end):
    fig = new_figure((15, 10))
    outer = fig.add_gridspec(1, 2, wspace=.25)
    # Separate colorbar columns keep heatmaps and markers on identical time widths.
    grids = [outer[0, col].subgridspec(4, 2, height_ratios=[1, 1, 1, .25],
                                      width_ratios=[1, .035], hspace=.28, wspace=.05)
             for col in range(2)]
    first = None
    for row, ntx in enumerate((1, 2, 3)):
        for col, view in enumerate(('raw', 'centered')):
            ax = fig.add_subplot(grids[col][row, 0], sharex=first)
            if first is None:
                first = ax
            color_limits = limits['raw'] if view == 'raw' else limits['centered_by_shape'][row]
            mesh = draw_heat(ax, data, ntx, view, color_limits, end)
            ax.tick_params(labelbottom=False)
            cb = fig.colorbar(mesh, cax=fig.add_subplot(grids[col][row, 1]),
                              extend='both' if view == 'centered' and data['mode'] != 'phase_difference' else 'neither')
            cb.ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title('Original packet feature' if view == 'raw' else data['center_label'])
    for col in range(2):
        marker_axis(fig.add_subplot(grids[col][3, 0], sharex=first), data, end)
    fig.suptitle(f"{data['label']} | {NAMES[data['mode']]} | all 30 groups", fontsize=16, y=.98)
    fig.text(.055, .937, f"Stream {data['config'].tx}, RX {data['config'].rx}, reference RX {data['config'].reference_rx}. "
             f"{UNITS[data['mode']]}. Gray cells: no valid sample. No interpolation or filtering.", fontsize=10)
    scaling = ('Phase colors span [-pi, pi].' if data['mode'] == 'phase_difference'
               else f"Centered colors saturate at the {limits['centered_color_percentile']:g}th absolute percentile; values are unchanged.")
    caption(fig, 'Color scales are shared across sessions; centered scales are per-shape. ' + scaling)
    fig.subplots_adjust(left=.07, right=.95, top=.88, bottom=.105)
    return fig


def overview_figure(items, limits, end, mode):
    fig = new_figure((16, 17))
    outer = fig.add_gridspec(3, 2, hspace=.32, wspace=.25)
    for i, data in enumerate(items):
        grid = outer[i // 2, i % 2].subgridspec(4, 2, height_ratios=[1, 1, 1, .25],
                                             width_ratios=[1, .03], hspace=.12, wspace=.05)
        first = None
        for j, ntx in enumerate((1, 2, 3)):
            ax = fig.add_subplot(grid[j, 0], sharex=first)
            if first is None:
                first = ax
            mesh = draw_heat(ax, data, ntx, 'centered', limits['centered_by_shape'][j], end)
            ax.tick_params(labelbottom=False, labelsize=8)
            if j == 0:
                ax.set_title(data['label'], loc='left', fontsize=11)
            cb = fig.colorbar(mesh, cax=fig.add_subplot(grid[j, 1]),
                              extend='both' if mode != 'phase_difference' else 'neither')
            cb.ax.tick_params(labelsize=7)
        marker_axis(fig.add_subplot(grid[3, 0], sharex=first), data, end)
    fig.suptitle(f'Six-session comparison | {NAMES[mode]} | 30 groups', fontsize=17, y=.983)
    fig.text(.065, .957, items[0]['center_label'] + '. Same color scale across sessions within each shape. No respiratory filtering.', fontsize=10)
    caption(fig, 'Gray regions preserve missing observations. Whole-session centering is retrospective; it does not establish respiratory origin.')
    fig.subplots_adjust(left=.065, right=.95, top=.923, bottom=.085)
    return fig


def waterfall_figure(data, end):
    from matplotlib import colormaps
    fig = new_figure((18, 7))
    for col, ntx in enumerate((1, 2, 3)):
        ax = fig.add_subplot(1, 3, col + 1, projection='3d')
        times, values = line_arrays(data, ntx, 'centered')
        for k in range(30):
            ax.plot(times, np.full(len(times), k + 1), values[:, k],
                    linewidth=.45, marker='.', markersize=.8,
                    color=colormaps['viridis'](k / 29), rasterized=True)
        ax.set(xlim=(0, end), ylim=(1, 30), yticks=[1, 10, 20, 30],
               xlabel='Time (s)', ylabel='Group', zlabel=UNITS[data['mode']], title=f'{ntx} x 3')
        ax.view_init(elev=28, azim=-65)
        ax.tick_params(labelsize=8)
        valid = values[np.isfinite(values)]
        if not len(valid):
            ax.text2D(.25, .55, 'No valid samples', transform=ax.transAxes)
        elif valid.min() == valid.max():
            ax.set_zlim(float(valid[0]) - .5, float(valid[0]) + .5)
    fig.suptitle(f"{data['label']} | {NAMES[data['mode']]} | 3D waterfall", fontsize=17, y=.965)
    fig.text(.055, .88, data['center_label'] + '. All 30 groups and all packet samples retained. Z scales are local to each panel.', fontsize=10)
    caption(fig, 'Lines break at shape/rate changes, non-increasing time, gaps and invalid values. No interpolation, smoothing or 3D surface filling.')
    fig.subplots_adjust(left=.02, right=.94, top=.84, bottom=.13, wspace=.05)
    return fig


def trace_figure(data, limits, end, groups):
    fig = new_figure((15, 10))
    grid = fig.add_gridspec(4, len(groups), height_ratios=[1, 1, 1, .25], hspace=.28, wspace=.22)
    first = None
    for row, ntx in enumerate((1, 2, 3)):
        times, values = line_arrays(data, ntx, 'centered')
        # Unlike heatmap colors, waveform axes include the full selected-group range.
        chosen = values[:, np.asarray(groups) - 1]
        valid = chosen[np.isfinite(chosen)]
        bound = max(float(np.max(np.abs(valid))), 1e-9) if len(valid) else 1.
        for col, group in enumerate(groups):
            ax = fig.add_subplot(grid[row, col], sharex=first)
            if first is None:
                first = ax
            ax.plot(times, values[:, group - 1], color='#0072B2', linewidth=.6, marker='.', markersize=1.5)
            ax.set(xlim=(0, end), ylim=(-bound * 1.05, bound * 1.05))
            ax.grid(alpha=.2)
            ax.tick_params(labelbottom=False)
            if col == 0:
                ax.set_ylabel(f'{ntx} x 3\n{UNITS[data["mode"]]}')
            if row == 0:
                ax.set_title(f'Group {group}')
    for col in range(len(groups)):
        marker_axis(fig.add_subplot(grid[3, col], sharex=first), data, end)
    fig.suptitle(f"{data['label']} | {NAMES[data['mode']]} | fixed example groups", fontsize=16, y=.98)
    fig.text(.065, .936, data['center_label'] + '. Groups chosen in advance, not ranked by apparent periodicity. No respiratory filtering.', fontsize=10)
    caption(fig, 'Full values shown without color saturation. Lines never cross known configuration boundaries or long gaps.')
    fig.subplots_adjust(left=.075, right=.97, top=.88, bottom=.105)
    return fig


def save_figure(fig, output, stem):
    for ext in ('png', 'pdf', 'svg'):
        fig.savefig(output / f'{stem}.{ext}', dpi=140)
    fig.clear()


def gallery_link(stem, title):
    return (f'<h3>{html.escape(title)}</h3><a href="{stem}.png"><img loading="lazy" '
            f'src="{stem}.png" alt="{html.escape(title)}"></a>'
            f'<p><a href="{stem}.pdf">PDF</a> / <a href="{stem}.svg">SVG</a></p>')


def implementation_hashes():
    package = Path(__file__).parent
    return [{'path': str(p.relative_to(package)), 'sha256': sha256(p)}
            for p in sorted(package.rglob('*.py'))]


def analyze_subcarriers(directory, output, modes=DEFAULT_MODES, tx=1, rx=1, reference_rx=2,
                        max_gap=.2, cell_seconds=.05, color_percentile=99., groups=(1, 15, 30)):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    if output.is_relative_to(directory):
        raise ValueError('Results must be outside the raw archive')
    if not modes or len(set(modes)) != len(modes) or any(m not in MODES for m in modes):
        raise ValueError('Select unique supported modes')
    if (not math.isfinite(max_gap) or not 0 < max_gap <= 1
            or not math.isfinite(cell_seconds) or not 0 < cell_seconds <= max_gap
            or not math.isfinite(color_percentile) or not 90 <= color_percentile <= 100):
        raise ValueError('Require 0 < cell width <= max gap <= 1 s and color percentile 90..100')
    if not 1 <= len(groups) <= 4 or len(set(groups)) != len(groups) or any(type(g) is not int or not 1 <= g <= 30 for g in groups):
        raise ValueError('Select one to four unique example groups in 1..30')
    configs = [FeatureConfig(m, tx, rx, reference_rx) for m in modes]
    manifest, archive_hash, layout = load_archive(directory)
    implementation = implementation_hashes()
    output.mkdir(parents=True, exist_ok=False)
    end = max(c.duration for _, _, c in layout)
    sections, summaries, scale_report = [], [], {}
    for config in configs:
        items = []
        for condition, trial, capture in layout:
            data = extract(capture, config, max_gap)
            centered, centers, resultant = centered_features(data, config.mode)
            edges, owner = packet_cells(data['times'], capture.duration, cell_seconds)
            data.update(centered=centered, centers=centers, phase_resultant=resultant,
                        edges=edges, owner=owner, config=config, mode=config.mode,
                        label=f'{condition} / Trial {trial}', condition=condition, trial=trial,
                        center_label='Circular-centered phase difference' if config.mode == 'phase_difference' else 'Per-group, per-shape median-centered feature')
            items.append(data)
        scales = limits_for(items, config.mode, color_percentile)
        scale_report[config.mode] = scales
        stem = config.mode + '-overview'
        print(f'Exporting {config.mode}: six-session overview', flush=True)
        save_figure(overview_figure(items, scales, end, config.mode), output, stem)
        sections.append(f'<section id="{config.mode}"><h2>{NAMES[config.mode]}</h2>' + gallery_link(stem, 'Six-session overview'))
        for data, (_, _, capture) in zip(items, layout):
            stem = f'{config.mode}-{data["condition"].lower()}-trial-{data["trial"]}'
            print(f'Exporting {stem}', flush=True)
            for suffix, fig in [('heatmap', heatmap_figure(data, scales, end)),
                                ('waterfall', waterfall_figure(data, end)),
                                ('traces', trace_figure(data, scales, end, groups))]:
                save_figure(fig, output, stem + '-' + suffix)
            arrays = {key: data[key] for key in ('times', 'shapes', 'rates', 'record_indices', 'segments',
                       'raw', 'centered', 'centers', 'phase_resultant', 'markers', 'edges', 'owner')}
            np.savez_compressed(output / (stem + '-samples.npz'), **arrays)
            counts, saturated = {}, {}
            for ntx in (1, 2, 3):
                selected = (data['shapes'][:, 0] == ntx) & (data['shapes'][:, 1] == 3)
                counts[f'{ntx}x3'] = int(selected.sum())
                vals = data['centered'][selected]
                lo, hi = scales['centered_by_shape'][ntx - 1]
                saturated[f'{ntx}x3'] = {'valid_values': int(np.isfinite(vals).sum()),
                    'saturated_color_values': int(((vals < lo) | (vals > hi)).sum())}
            summaries.append({'condition': data['condition'], 'trial': data['trial'],
                'session_id': capture.session_id,
                'feature': {'mode': config.mode, 'tx': tx, 'rx': rx, 'reference_rx': reference_rx,
                            'subcarrier_groups': GROUPS.tolist()},
                'packets': len(data['times']), 'duration_seconds': capture.duration,
                'shape_counts': counts, 'markers': len(data['markers']), 'saturation': saturated,
                'other_nrx_packets': int((data['shapes'][:, 1] != 3).sum()),
                'samples': stem + '-samples.npz', 'timing': capture.timing})
            sections.append(f'<details><summary>{html.escape(data["label"])}</summary>'
                            + ''.join(gallery_link(stem + '-' + s, title) for s, title in
                                      [('heatmap', 'Original and centered heatmaps'),
                                       ('waterfall', 'All-group 3D waterfall'),
                                       ('traces', 'Fixed example groups with manual markers')]) + '</details>')
        sections.append('</section>')
    header = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
              '<title>All-subcarrier CSI diagnostics</title><style>body{font:16px system-ui;max-width:1350px;'
              'margin:32px auto;padding:0 20px;color:#222}img{width:100%;height:auto}p{line-height:1.6}'
              'summary{cursor:pointer;padding:14px;background:#eef4f7;margin:12px 0}section{margin-top:40px}</style>')
    (output / 'index.html').write_text(header + f'<h1>{html.escape(manifest["batch_id"])} all-subcarrier diagnostics</h1>'
        '<p>All 30 measured Intel 5300 subcarrier groups, separated by reported 1x3, 2x3 and 3x3 configuration. '
        'Stream indices do not identify fixed physical TX antennas across configurations. '
        'Default views use stream 1, RX 1 and reference RX 2 unless configured otherwise.</p>'
        f'<p>{LIMITATION}</p><p>Heatmaps show capped packet glyphs, not interpolated measurements. '
        f'Each glyph is at most {cell_seconds:g} s wide and is clipped by neighboring packet midpoints. '
        'Gray areas mean no valid displayed sample, not absent respiration. Centering uses the entire session '
        'within each shape and group, so it is retrospective, not a deployable causal preprocessing step. '
        'Phase residuals use circular centering without unwrapping. No respiratory filtering or rate estimation is applied.</p>'
        f'<p>Original heatmaps share full-range scales. Centered nonphase heatmaps use the {color_percentile:g}th '
        'percentile of absolute residuals across all six sessions separately for each shape. Values outside '
        'these limits saturate in color only. Saved arrays, waterfall plots and example traces retain full values. '
        'Waterfall Z scales are panel-local. Coincident periodicity does not establish respiratory origin.</p>'
        '<p>Each overview is followed by expandable session figures. These are exploratory diagnostics, not a trained detector. '
        'The existing fixed-3x3 replay processor and original capture files are unchanged.</p>'
        + ''.join(sections) + '</html>', encoding='utf-8')
    checked, final_hash, _ = load_archive(directory)
    if checked != manifest or final_hash != archive_hash:
        raise ValueError('Archive changed during export')
    if implementation_hashes() != implementation:
        raise ValueError('Implementation changed during export; rerun with a stable source tree')
    report = {'schema_version': 1, 'complete': True, 'diagnostic_version': 'all-subcarriers-v1',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'archive_directory': str(directory),
        'archive_sha256': archive_hash, 'corrections_sha256': manifest['corrections_sha256'],
        'raw_data_modified': False, 'diagnostic_only': True, 'causal': False,
        'parameters': {'max_gap_seconds': max_gap, 'cell_seconds': cell_seconds,
                       'color_percentile': color_percentile, 'example_groups': list(groups)},
        'color_scales': scale_report, 'summaries': summaries, 'limitation': LIMITATION,
        'implementation_hashes': implementation,
        'artifacts': [{'path': p.name, 'bytes': p.stat().st_size, 'sha256': sha256(p)}
                      for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / 'subcarrier-report.json', report)
    return report
