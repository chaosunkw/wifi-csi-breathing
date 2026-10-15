"""Third-batch comparison with unchanged FFT/ACF, FarSense and WiResP runs."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import html
import json

import numpy as np

from .analysis import write_csv, write_json
from .baselines.vitalcsi import VitalCSIConfig, amplitudes, analyze_interval
from .data import sha256
from .farsense_comparison import save_plot, table_html, validate_manifest
from .joint import JointCapture
from .subcarriers import implementation_hashes, new_figure
from .wiresp_comparison import compare_wiresp


METHODS = ('LogRatio-FFT', 'LogRatio-ACF', 'FarSense-core', 'WiResP-adapted', 'VitalCSI-adapted')
KEYS = ('fft_bpm', 'acf_bpm', 'farsense_bpm', 'wiresp_bpm', 'vitalcsi_bpm')
COLORS = ('#0072B2', '#AD6700', '#8055B3', '#008675', '#C44E52')


def metrics_for(rows):
    eligible = [r for r in rows if not r['empty_room'] and r['manual_reference_bpm'] is not None]
    common = [r for r in eligible if all(r[k] is not None for k in KEYS)]
    empty = [r for r in rows if r['empty_room']]
    metrics = []
    for name, key in zip(METHODS, KEYS):
        available = [r for r in eligible if r[key] is not None]
        errors = [abs(r[key]-r['manual_reference_bpm']) for r in available]
        paired = [abs(r[key]-r['manual_reference_bpm']) for r in common]
        metrics.append({'method': name, 'eligible_minutes': len(eligible), 'available_minutes': len(available),
                        'mae_available_bpm': float(np.mean(errors)) if errors else None,
                        'common_minutes': len(common), 'mae_common_bpm': float(np.mean(paired)) if paired else None,
                        'empty_minutes': len(empty), 'empty_candidate_minutes': sum(r[key] is not None for r in empty)})
    return metrics


def overview(rows):
    occupied = [r for r in rows if not r['empty_room']]
    fig = new_figure((15.5, 8.5))
    rates, errors = fig.subplots(2, 1, sharex=True)
    fig.subplots_adjust(left=.08, right=.98, top=.83, bottom=.24, hspace=.2)
    fig.suptitle('Third batch | VitalCSI adaptation alongside four existing baselines', fontsize=17, y=.98)
    fig.text(.08, .92, 'Identical predefined minute intervals. All unavailable results are retained; manual counts are comparison only.', fontsize=10)
    x = np.arange(len(occupied))
    refs = np.array([r['manual_reference_bpm'] if r['manual_reference_bpm'] is not None else np.nan for r in occupied])
    rates.scatter(x, refs, marker='_', s=460, color='#222222', label='Manual / min', zorder=5)
    for shift, name, key, color in zip(np.linspace(-.32, .32, 5), METHODS, KEYS, COLORS):
        values = np.array([r[key] if r[key] is not None else np.nan for r in occupied])
        rates.scatter(x+shift, values, s=40, color=color, label=name)
        errors.bar(x+shift, abs(values-refs), width=.145, color=color)
        for i in np.flatnonzero(~np.isfinite(values)):
            rates.text(i+shift, .02, 'N/A', rotation=90, color=color, fontsize=7, transform=rates.get_xaxis_transform())
    rates.set_ylabel('Rate estimate (/min)')
    errors.set_ylabel('Absolute difference\nfrom manual (/min)')
    errors.set_xticks(x, [r['label']+'\n'+f"{r['start_s']:g}-{r['end_s']:g} s" for r in occupied], rotation=25, ha='right', fontsize=9)
    rates.legend(ncol=3, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.29))
    for ax in (rates, errors):
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    errors.set_ylim(bottom=0)
    fig.text(.08, .025, 'VitalCSI: mean of all 31 overlapping 30 s SQI/Kalman-fused estimates; every frame required. Missing frames are not filled by prediction.\n'
             'FFT/ACF and WiResP: candidates before their final gates. FarSense: five 12 s estimates. Different pipelines/latencies; exploratory comparison.', fontsize=9)
    return fig


def pack_arrays(frames, data, cfg):
    nf, n, k = len(frames), round(cfg.sample_hz*cfg.window_seconds), cfg.top_k
    packed = {name: np.full((nf, n, 30), np.nan) for name in ('amplitudes', 'pcs')}
    packed.update({name: np.full((nf, n, k), np.nan) for name in ('filtered', 'ssf', 'bssf', 'thresholds')})
    packed.update({name: np.full((nf, 30), np.nan) for name in ('variance_fraction', 'snr', 'cqi', 'peak_hz')})
    packed.update(loadings=np.full((nf, 30, 30), np.nan), raw_power=np.full((nf, n//2+1, 30), np.nan),
                  filtered_power=np.full((nf, n//2+1, k), np.nan), selected_indices=np.full((nf, k), -1, int),
                  group_mask=np.zeros((nf, 30), bool), sample_times=np.full((nf, n), np.nan),
                  frequencies_hz=np.fft.rfftfreq(n, 1/cfg.sample_hz), frame_end_s=np.array([f['end_s'] for f in frames]))
    for i, a in enumerate(data):
        if a is None:
            continue
        nc, ns = len(a['cqi']), len(a['selected_indices'])
        packed['amplitudes'][i], packed['sample_times'][i] = a['amplitudes'], a['times']
        packed['group_mask'][i, a['group_indices']] = True
        packed['pcs'][i, :, :nc], packed['raw_power'][i, :, :nc] = a['pcs'], a['raw_power']
        packed['loadings'][i, :nc] = a['loadings']
        for key in ('variance_fraction', 'snr', 'cqi', 'peak_hz'):
            packed[key][i, :nc] = a[key]
        packed['selected_indices'][i, :ns] = a['selected_indices']
        if ns:
            for key in ('filtered', 'ssf', 'bssf', 'thresholds', 'filtered_power'):
                packed[key][i, :, :ns] = a[key]
    return packed


def stages_figure(row, frames, data, markers, cfg):
    """Always show the first frame, never a post-hoc best-looking waveform."""
    fig = new_figure((14, 17))
    axes = fig.subplots(6, 1)
    fig.subplots_adjust(left=.09, right=.97, top=.91, bottom=.055, hspace=.82)
    fig.suptitle(row['label']+f" | {row['start_s']:g}-{row['end_s']:g} s | VitalCSI adapted", fontsize=17, y=.985)
    fig.text(.09, .95, 'First four panels: first completed 30 s window. Last two panels: all 31 frames. No waveform fitting to manual events.', fontsize=10)
    a, frame = data[0], frames[0]
    for ax in axes:
        ax.grid(alpha=.16)
    titles = ['1. Raw CSI magnitude | stream 1, fixed RX, predefined groups 1 / 15 / 30',
              '2. Selected PCA components after filtering | standardized for display only',
              '3. Hann spectra of the selected components | no zero padding',
              '4. BSSF onset detector | highest-CQI component of this first window',
              '5. Per-component estimates and Kalman fusion | trailing-window end time',
              '6. Component signal quality | SNR x estimator agreement x spectral purity']
    for ax, title in zip(axes, titles):
        ax.set_title(title, loc='left', fontsize=11)
    axes[0].set_ylabel('Raw magnitude')
    axes[1].set_ylabel('Standardized\ncomponent')
    axes[2].set_ylabel('Power / maximum')
    axes[3].set_ylabel('Slope sum')
    axes[4].set_ylabel('Cycles/min')
    axes[5].set_ylabel('SQI (not probability)')
    if a is not None:
        for group in (0, 14, 29):
            axes[0].plot(a['times'], a['amplitudes'][:, group], linewidth=.8, label=f'Group {group+1}')
        axes[0].legend(fontsize=8, ncol=3)
        for column, component in enumerate(frame['components']):
            pc, x = component['pc_index']+1, a['filtered'][:, column]
            standardized = (x-x.mean())/max(float(x.std()), 1e-20)
            axes[1].plot(a['times'], standardized, linewidth=.9, label=f'PC {pc}, CQI {component["cqi"]:.2f}')
            p = a['filtered_power'][:, column]
            axes[2].plot(a['frequencies_hz']*60, p/max(float(p.max()), 1e-20), label=f'PC {pc}')
        if frame['components']:
            c = frame['components'][0]
            axes[1].legend(fontsize=8, ncol=3)
            axes[2].legend(fontsize=8, ncol=5)
            axes[3].plot(a['times'], a['ssf'][:, 0], label='Slope sum', linewidth=1)
            axes[3].step(a['times'], a['bssf'][:, 0], label='Boxed slope sum', alpha=.65, linewidth=.9)
            axes[3].plot(a['times'], a['thresholds'][:, 0], '--', color='#a05e00', label='Adaptive threshold')
            for onset in c['onsets_s']:
                axes[3].axvline(onset, color=COLORS[4], alpha=.55, linewidth=.8)
            axes[3].legend(fontsize=8, ncol=3, loc='upper right')
            fft_text = f"{c['fft_bpm']:.2f}" if c['fft_bpm'] is not None else 'N/A'
            breath_text = f"{c['breath_bpm']:.2f}" if c['breath_bpm'] is not None else 'N/A'
            axes[3].set_title(titles[3]+f"\nPC {c['pc_index']+1}: FFT {fft_text}, BSSF {breath_text} /min; red lines = onsets",
                              loc='left', fontsize=10)
    if a is None or not frame['components']:
        for ax in axes[1:4] if a is not None else axes[:4]:
            ax.text(.5, .5, frame['reason'].replace('_', ' '), transform=ax.transAxes, ha='center')
    for ax in (axes[0], axes[1], axes[3]):
        ax.set_xlim(row['start_s'], row['start_s']+cfg.window_seconds)
        ax.set_xlabel('Seconds from common Ubuntu T0')
    for ax in (axes[0], axes[1]):
        for marker in markers:
            if row['start_s'] <= marker < row['start_s']+cfg.window_seconds:
                ax.axvline(marker, color='#008675', alpha=.2, linewidth=.8)
    axes[2].set_xlim(0, 60)
    axes[2].axvspan(cfg.low_bpm, cfg.high_bpm, color='#008675', alpha=.06)
    axes[2].set_xlabel('Frequency (cycles/min)')
    for f in frames:
        for c in f['components']:
            if c['combined_bpm'] is not None:
                axes[4].scatter(f['end_s'], c['combined_bpm'], s=8, color='#929ba3', alpha=.5)
            axes[5].scatter(f['end_s'], c['sqi'], s=12, color=COLORS[4], alpha=.55)
    axes[4].plot([f['end_s'] for f in frames], [f['fused_bpm'] if f['fused_bpm'] is not None else np.nan for f in frames],
                 color=COLORS[4], linewidth=1.8, marker='o', markersize=3, label='Kalman fusion')
    if row['manual_reference_bpm'] is not None:
        axes[4].axhline(row['manual_reference_bpm'], color='#333333', linestyle='--', label='Whole-minute manual count')
    axes[4].legend(fontsize=8, ncol=2)
    for ax in axes[4:]:
        ax.set_xlim(row['start_s']+cfg.window_seconds, row['end_s'])
        ax.set_xlabel('Trailing-window end time from common Ubuntu T0 (s)')
    axes[5].set_ylim(-.03, 1.03)
    fig.text(.09, .015, 'Within-window filtering is retrospective; each estimate is available only when its window and interpolation endpoints are complete.\n'
             'Green lines: manual presses, never estimator inputs. PCA sign does not identify inhalation. Missing estimates are not silently held.', fontsize=9)
    return fig


def compare_vitalcsi(manifest_path, output, cfg=VitalCSIConfig()):
    if cfg.window_seconds != 30 or cfg.hop_seconds != 1:
        raise ValueError('The minute comparison requires 30 s windows and 1 s hops')
    records = validate_manifest(manifest_path)
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(r['input']).resolve()) for r in records):
        raise ValueError('Output must not be inside a source recording')
    implementation, manifest_hash = implementation_hashes(), sha256(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/'input-manifest.json', json.loads(Path(manifest_path).read_text()))
    reference = compare_wiresp(manifest_path, output/'reference-methods')
    old_rows = {r['key']: r for r in reference['results']}
    rows, details = [], []
    for record in records:
        capture = JointCapture(record['input'])
        times, values, audit = amplitudes(capture, cfg)
        for start in record['starts_s']:
            key = f"{record['key']}-{start:g}-{start+60:g}s"
            row = dict(old_rows[key])
            summary, frames, data = analyze_interval(times, values, start, start+60, cfg)
            row.update(vitalcsi_bpm=summary['fused_mean_bpm'], vitalcsi_available_frames=summary['available_frames'],
                       vitalcsi_total_frames=summary['total_frames'], vitalcsi_partial_mean_bpm=summary['available_only_mean_bpm'],
                       vitalcsi_raw_component_mean_bpm=summary['raw_component_mean_bpm'])
            row['vitalcsi_absolute_difference'] = (abs(row['vitalcsi_bpm']-row['manual_reference_bpm'])
                if row['vitalcsi_bpm'] is not None and row['manual_reference_bpm'] is not None else None)
            rows.append(row)
            dest = output/key
            dest.mkdir()
            np.savez_compressed(dest/'intermediates.npz', **pack_arrays(frames, data, cfg))
            write_json(dest/'comparison.json', row)
            write_json(dest/'vitalcsi-summary.json', summary)
            write_json(dest/'frames.json', frames)
            fields = ('start_s', 'end_s', 'available_at_s', 'fused_bpm', 'raw_component_mean_bpm',
                      'selected_components', 'usable_components', 'valid_groups', 'bridged_grid_fraction', 'reason')
            write_csv(dest/'frames.csv', ({k: f.get(k) for k in fields} for f in frames), fields)
            component_fields = ('end_s', 'pc_index', 'variance_fraction', 'cqi', 'snr', 'spi', 'agreement', 'sqi',
                                'fft_bpm', 'breath_bpm', 'combined_bpm')
            write_csv(dest/'components.csv', ({k: f['end_s'] if k == 'end_s' else c[k] for k in component_fields}
                      for f in frames for c in f['components']), component_fields)
            write_csv(dest/'packet-audit.csv', audit, tuple(audit[0]) if audit else
                      ('record_index', 'time_s', 'status', 'ntx', 'nrx', 'rate_flags', 'valid_groups'))
            markers = [(e['host_monotonic_ns']-capture.t0)/1e9 for e in capture.events
                       if start <= (e['host_monotonic_ns']-capture.t0)/1e9 < start+60]
            save_plot(stages_figure(row, frames, data, markers, cfg), dest, 'vitalcsi-stages')
            details.append(f'<details><summary>{html.escape(row["label"])} | {start:g}-{start+60:g} s</summary>'
                f'<img src="{key}/vitalcsi-stages.png" alt="VitalCSI amplitude PCA spectra BSSF and fusion stages">'
                f'<p><a href="{key}/frames.csv">Frame results</a> | <a href="{key}/components.csv">Component estimates</a> | '
                f'<a href="{key}/frames.json">Full decisions and Kalman weights</a> | <a href="{key}/intermediates.npz">Numerical arrays</a> | '
                f'<a href="{key}/vitalcsi-stages.svg">Vector figure</a> | <a href="reference-methods/{key}/wiresp-stages.png">WiResP stages</a></p></details>')
            print(f"{key}: VitalCSI={row['vitalcsi_bpm']}, frames={summary['available_frames']}/{summary['total_frames']}, manual={row['manual_reference_bpm']}", flush=True)
        capture.verify()
    metrics = metrics_for(rows)
    save_plot(overview(rows), output, 'comparison-overview')
    write_csv(output/'summary.csv', rows, tuple(rows[0]))
    write_csv(output/'metrics.csv', metrics, tuple(metrics[0]))
    fields = [('label', 'Recording'), ('start_s', 'Start (s)')]+list(zip(KEYS, METHODS))+[('manual_reference_bpm', 'Manual /min')]
    availability = [('label', 'Recording'), ('start_s', 'Start (s)'), ('vitalcsi_available_frames', 'Available frames'),
                    ('vitalcsi_total_frames', 'Total frames'), ('vitalcsi_bpm', 'Complete-minute mean'),
                    ('vitalcsi_raw_component_mean_bpm', 'Pre-SQI component mean'),
                    ('vitalcsi_partial_mean_bpm', 'Partial mean (not used in minute MAE)')]
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>VitalCSI: third-batch comparison</title><style>'
        'body{font:16px system-ui;color:#183249;background:#f7f9fb;max-width:1450px;margin:32px auto;padding:0 24px;line-height:1.6}'
        'h1{line-height:1.2}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}'
        'td,th{padding:9px;text-align:left;border-bottom:1px solid #d8e1e8}.scroll{overflow:auto}'
        'section,details{background:white;border:1px solid #d8e1e8;border-radius:8px;padding:18px;margin:20px 0}'
        'summary{font-size:20px;cursor:pointer;font-weight:650}a{color:#006c9a}.note{border-left:4px solid #b47727;padding-left:14px}'
        '</style><h1>VitalCSI adaptation vs. four existing baselines</h1>'
        '<p>Third batch: independent implementation of amplitude PCA, component selection, FFT/BSSF estimation, SQI and Kalman fusion from '
        '<a href="https://doi.org/10.3390/s26010225">Michaelis et al., VitalCSI</a>.</p>'
        '<p class="note">Adapted reproduction, not the authors\' exact implementation or reported accuracy. '
        'The paper has an ambiguous variance rule and omits several filter/BSSF/Kalman settings. '
        'Our choices are fixed and exported in the audit. Input is stream 1, RX1 and 30 measured groups, resampled at 20 Hz. '
        'No reference events or activity labels enter any estimator.</p>'
        '<section><h2>Five-method comparison</h2><img src="comparison-overview.png" alt="Five-method rate estimates and errors">'
        '<p>All methods use the same predefined minute intervals. VitalCSI uses 31 overlapping 30 s frames at 1 s hops; '
        'its complete-minute mean requires all 31 SQI/Kalman estimates. FFT/ACF and WiResP values remain before their final gates. '
        'FarSense retains five 12 s blocks. Features, windows, bands and quality handling differ by method; '
        'this is an exploratory end-to-end comparison, not an estimator-only ablation.</p><div class="scroll">'
        +table_html([r for r in rows if not r['empty_room']], fields)+'</div></section>'
        '<section><h2>Availability and error against manual counts</h2><div class="scroll">'
        +table_html(metrics, [('method', 'Method'), ('available_minutes', 'Available'), ('eligible_minutes', 'Eligible'),
          ('mae_available_bpm', 'MAE on available'), ('common_minutes', 'Common to all five'), ('mae_common_bpm', 'MAE on common'),
          ('empty_candidate_minutes', 'Empty-room minute candidates')])+'</div>'
        '<p>Different available sets cannot be ranked by their MAEs alone. Manual presses are self-reports, not a physiological sensor. '
        'Overlapping windows and repeated recordings are not independent subjects.</p></section>'
        '<section><h2>VitalCSI coverage, including unavailable estimates</h2><div class="scroll">'+table_html(rows, availability)
        +'</div><p>SQI-zero components do not contribute to fusion. Missing results are never replaced by a held Kalman prediction. '
        'Partial means are shown for inspection only. Fusion restarts at each compared minute.</p></section>'
        '<section><h2>Empty-room controls</h2><div class="scroll">'+table_html([r for r in rows if r['empty_room']], fields)
        +'</div><p>No zero-rate ground truth is assigned. An estimate in an empty room can arise from periodic noise; '
        'neither a high SQI nor Kalman fusion establishes breathing or person presence.</p></section>'
        '<h2>Inspect PCA, frequency, breath counting and fusion</h2>'+''.join(details)
        +'<section><h2>Reproducibility</h2><p>Parameters fixed before this run. Original timestamps and configurations are audited. '
        'Bounded gap interpolation and required window support remain explicit. Source files and receipts verified unchanged.</p>'
        '<p><a href="reference-methods/index.html">Previous four methods</a> | <a href="summary.csv">Comparison CSV</a> | '
        '<a href="metrics.csv">Metrics CSV</a> | <a href="report.json">Audit and settings</a> | '
        '<a href="comparison-overview.svg">Vector overview</a></p></section></html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    if implementation_hashes() != implementation or sha256(manifest_path) != manifest_hash:
        raise ValueError('Code or manifest changed during comparison')
    for source in reference['sources']:
        if sha256(Path(source['directory'])/'transfer.json') != source['receipt_sha256']:
            raise ValueError('Source receipt changed during comparison')
        for item in source['files']:
            if sha256(Path(source['directory'])/item['name']) != item['sha256']:
                raise ValueError('Source recording changed during comparison')
    report = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'VitalCSI-adapted-v1', 'paper_doi': '10.3390/s26010225', 'configuration': asdict(cfg),
              'source_manifest_sha256': manifest_hash, 'sources': reference['sources'],
              'results': rows, 'metrics': metrics, 'implementation_hashes': implementation,
              'raw_data_modified': False, 'manual_events_used_in_estimation': False,
              'full_system_reproduction': False, 'author_code_verified': False,
              'artifacts': [{'path': str(p.relative_to(output)), 'sha256': sha256(p)}
                            for p in sorted(output.rglob('*')) if p.is_file()]}
    write_json(output/'report.json', report)
    return report
