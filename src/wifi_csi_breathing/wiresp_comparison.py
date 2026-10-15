"""Verified second-batch comparison; previous three methods are rerun unchanged."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import html
import json

import numpy as np

from .analysis import write_csv, write_json
from .baselines.wiresp import WiResPConfig, analyze_interval, normalized_amplitudes
from .data import sha256
from .farsense_comparison import compare_farsense, save_plot, table_html, validate_manifest
from .joint import JointCapture
from .subcarriers import implementation_hashes, new_figure


METHODS = ('LogRatio-FFT', 'LogRatio-ACF', 'FarSense-core', 'WiResP-adapted')
KEYS = ('fft_bpm', 'acf_bpm', 'farsense_bpm', 'wiresp_bpm')
COLORS = ('#0072B2', '#AD6700', '#8055B3', '#008675')


def metrics_for(rows):
    eligible = [r for r in rows if not r['empty_room'] and r['manual_reference_bpm'] is not None]
    common = [r for r in eligible if all(r[k] is not None for k in KEYS)]
    empty = [r for r in rows if r['empty_room']]
    result = []
    for name, key in zip(METHODS, KEYS):
        available = [r for r in eligible if r[key] is not None]
        errors = [abs(r[key]-r['manual_reference_bpm']) for r in available]
        paired = [abs(r[key]-r['manual_reference_bpm']) for r in common]
        result.append({'method': name, 'eligible_minutes': len(eligible), 'available_minutes': len(available),
                       'mae_available_bpm': float(np.mean(errors)) if errors else None,
                       'common_minutes': len(common), 'mae_common_bpm': float(np.mean(paired)) if paired else None,
                       'empty_minutes': len(empty), 'empty_candidate_minutes': sum(r[key] is not None for r in empty)})
    return result


def overview(rows):
    occupied = [r for r in rows if not r['empty_room']]
    fig = new_figure((15, 8.5))
    rates, errors = fig.subplots(2, 1, sharex=True)
    fig.subplots_adjust(left=.08, right=.98, top=.84, bottom=.23, hspace=.20)
    fig.suptitle('Second batch | WiResP adaptation alongside three existing baselines', fontsize=17, y=.98)
    fig.text(.08, .915, 'Same predefined minute intervals. Candidate rates before quality gates; unavailable minutes stay unavailable.', fontsize=11)
    x = np.arange(len(occupied))
    refs = np.array([r['manual_reference_bpm'] if r['manual_reference_bpm'] is not None else np.nan for r in occupied])
    rates.scatter(x, refs, marker='_', s=450, color='#222222', label='Manual / min', zorder=5)
    for shift, name, key, color in zip((-.30, -.10, .10, .30), METHODS, KEYS, COLORS):
        values = np.array([r[key] if r[key] is not None else np.nan for r in occupied])
        rates.scatter(x+shift, values, s=45, color=color, label=name)
        errors.bar(x+shift, abs(values-refs), width=.18, color=color)
        for i in np.flatnonzero(~np.isfinite(values)):
            rates.text(i+shift, .03, 'N/A', rotation=90, color=color, fontsize=7,
                       transform=rates.get_xaxis_transform())
    rates.set_ylabel('Candidate rate (/min)')
    errors.set_ylabel('Absolute difference\nfrom manual (/min)')
    errors.set_xticks(x, [r['label']+'\n'+f"{r['start_s']:g}-{r['end_s']:g} s" for r in occupied],
                      rotation=25, ha='right', fontsize=9)
    rates.legend(ncol=5, fontsize=9, loc='upper center', bbox_to_anchor=(.5, 1.18))
    for ax in (rates, errors):
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    errors.set_ylim(bottom=0)
    fig.text(.08, .025, 'WiResP: mean of all 46 overlapping 15 s candidates, only if all available. Its adapted continuity gate is reported separately.\n'
             'FarSense: mean of five 12 s candidates. FFT/ACF: one 60 s estimate. Manual reference is self-report; this is reused pilot data.', fontsize=9)
    return fig


def pack_arrays(frames, data, cfg):
    """Export reproducible numeric inputs and intermediate stages, no pickle."""
    nf, n = len(frames), round(cfg.window_seconds*cfg.sample_hz)
    packed = {name: np.full((nf, n, 90), np.nan) for name in ('normalized_amplitudes', 'median_amplitudes', 'acf')}
    packed.update({name: np.full((n, nf), np.nan) for name in ('combined', 'smoothed', 'enhanced_latest', 'binary_latest')})
    packed.update(bnr=np.full((nf, 90), np.nan), selected=np.zeros((nf, 90), bool),
                  lag_s=np.arange(n)/cfg.sample_hz, frame_end_s=np.array([f['end_s'] for f in frames]),
                  sample_times=np.full((nf, n), np.nan))
    for i, a in enumerate(data):
        if a is None:
            continue
        indices = a['channel_indices']
        packed['sample_times'][i] = a['times']
        packed['normalized_amplitudes'][i] = a['normalized_amplitudes']
        packed['median_amplitudes'][i][:, indices] = a['median_amplitudes']
        packed['acf'][i][:, indices] = a['acf']
        packed['bnr'][i, indices] = a['bnr']
        packed['selected'][i, indices[a['selected_indices']]] = True
        packed['combined'][:, i], packed['smoothed'][:, i] = a['combined_acf'], a['smoothed_acf']
        if 'enhanced' in a:
            packed['enhanced_latest'][:, i] = a['enhanced'][:, -1]
            packed['binary_latest'][:, i] = a['binary'][:, -1]
    return packed


def diagnostics_figure(row, frames, data, packed, cfg):
    fig = new_figure((14, 15))
    axes = fig.subplots(5, 1)
    fig.subplots_adjust(left=.09, right=.92, top=.90, bottom=.07, hspace=.65)
    fig.suptitle(row['label']+f" | {row['start_s']:g}-{row['end_s']:g} s | WiResP adapted", fontsize=17, y=.98)
    fig.text(.09, .94, 'The images below show autocorrelation versus lag and frame-end time. They are not breathing waveforms or FFT spectra.', fontsize=10)
    x = packed['frame_end_s']
    extent = [x[0]-cfg.hop_seconds/2, x[-1]+cfg.hop_seconds/2, 0, cfg.window_seconds]
    for ax, key, title in zip(axes[:3], ('smoothed', 'enhanced_latest', 'binary_latest'),
            ('1. BNR-weighted amplitude ACF after LOESS (before enhancement)',
             '2. Enhanced latest ACF column from each trailing 30-snapshot image',
             '3. Otsu binary latest column; blank = history unavailable')):
        image = packed[key]
        kwargs = {'vmin': 0, 'vmax': 1}
        if key == 'smoothed':
            visible = image[packed['lag_s'] <= 8]
            finite = visible[np.isfinite(visible)]
            kwargs = {'vmin': float(finite.min()), 'vmax': float(finite.max())} if finite.size else {}
        rendered = ax.imshow(image, origin='lower', aspect='auto', interpolation='nearest', extent=extent, cmap='viridis', **kwargs)
        ax.set_title(title, loc='left', fontsize=11)
        ax.set_ylabel('Lag (s)')
        ax.set_xlabel('Frame end from common T0 (s)')
        ax.set_ylim(0, 8)
        bar = fig.colorbar(rendered, ax=ax, fraction=.025, pad=.015)
        bar.ax.tick_params(labelsize=8)
        ax.set_facecolor('#eceff2')
        if key != 'smoothed':
            ax.text(.23, .5, 'Building trailing image history\nRaw rate candidates remain visible below',
                    transform=ax.transAxes, ha='center', va='center', color='#566574', fontsize=10)
    rates = axes[3]
    raw = [f['candidate_bpm'] if f['candidate_bpm'] is not None else np.nan for f in frames]
    accepted = [f['accepted_bpm'] if f['accepted_bpm'] is not None else np.nan for f in frames]
    rates.plot(x, raw, color=COLORS[3], label='Raw ACF candidate', linewidth=1.1)
    rates.scatter(x, accepted, color='#17384d', s=20, label='Passes adapted continuity rule')
    if row['manual_reference_bpm'] is not None:
        rates.axhline(row['manual_reference_bpm'], color='#555555', linestyle='--', label='Manual whole-minute count')
    rates.set(title='4. Rate candidates and adapted continuity decisions', ylabel='Cycles/min', xlabel='Frame end from common T0 (s)')
    rates.legend(fontsize=9, loc='upper right')
    rates.grid(alpha=.2)
    trend_ax = axes[4]
    last = next((i for i in range(len(data)-1, -1, -1) if data[i] is not None and 'binary_average' in data[i]), None)
    if last is None:
        trend_ax.text(.5, .5, 'No complete enhancement history', transform=trend_ax.transAxes, ha='center')
    else:
        a, f = data[last], frames[last]
        lag = packed['lag_s']
        trend_ax.plot(lag, a['binary_average'], color=COLORS[3], label='Time average of binary image')
        trend_ax.step(lag, a['binary_trend'], where='mid', color='#333333', linewidth=1, label='Majority trend')
        trend_ax.plot(lag, a['template'], color='#AD6700', linestyle=':', label='Adapted reference template')
        trend_ax.set_title(f"5. Last available history ending {f['end_s']:g} s | {f['continuity_reason'].replace('_', ' ')}", loc='left', fontsize=11)
        trend_ax.legend(fontsize=9, loc='upper right')
    trend_ax.set(xlabel='Lag (s)', ylabel='Binary fraction', xlim=(0, cfg.window_seconds), ylim=(-.1, 1.25))
    motion_note = 'Motion threshold supplied.' if cfg.motion_threshold is not None else 'Motion gating is disabled.'
    fig.text(.09, .018, 'Continuity thresholds are uncalibrated adaptation choices. '+motion_note+' Raw ACF colors span the displayed 0-8 s lags.\n'
             'Raw candidates remain visible while continuity history builds. Histogram enhancement is monotonic and cannot establish breathing by itself.', fontsize=9)
    return fig


def compare_wiresp(manifest_path, output, cfg=WiResPConfig()):
    records = validate_manifest(manifest_path)
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(r['input']).resolve()) for r in records):
        raise ValueError('Output must not be inside a source recording')
    implementation, manifest_hash = implementation_hashes(), sha256(manifest_path)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(manifest_path).read_text())
    write_json(output/'input-manifest.json', manifest)
    # Actual unchanged implementations, not values copied from an old report.
    reference = compare_farsense(manifest_path, output/'reference-methods')
    old_rows = {r['key']: r for r in reference['results']}
    rows, details = [], []
    for record in records:
        capture = JointCapture(record['input'])
        times, amplitudes, audit = normalized_amplitudes(capture)
        for start in record['starts_s']:
            key = f"{record['key']}-{start:g}-{start+60:g}s"
            row = dict(old_rows[key])
            summary, frames, data = analyze_interval(times, amplitudes, start, start+60, cfg)
            row.update(wiresp_bpm=summary['candidate_mean_bpm'], wiresp_candidate_frames=summary['candidate_frames'],
                       wiresp_total_frames=summary['total_frames'], wiresp_continuity_frames=summary['continuity_evaluated_frames'],
                       wiresp_accepted_frames=summary['accepted_frames'], wiresp_partial_after_check_bpm=summary['accepted_frames_mean_bpm'])
            row['wiresp_absolute_difference'] = (abs(row['wiresp_bpm']-row['manual_reference_bpm'])
                if row['wiresp_bpm'] is not None and row['manual_reference_bpm'] is not None else None)
            rows.append(row)
            dest = output/key
            dest.mkdir()
            packed = pack_arrays(frames, data, cfg)
            np.savez_compressed(dest/'intermediates.npz', **packed)
            write_json(dest/'comparison.json', row)
            write_json(dest/'wiresp-summary.json', summary)
            write_json(dest/'frames.json', frames)
            fields = ('start_s', 'end_s', 'available_at_s', 'candidate_bpm', 'enhanced_candidate_bpm', 'accepted_bpm',
                      'continuity_pass', 'continuity_reason', 'motion_statistic', 'motion_pass',
                      'valid_channels', 'bridged_grid_fraction', 'reason')
            write_csv(dest/'frames.csv', ({k: f.get(k) for k in fields} for f in frames), fields)
            write_csv(dest/'packet-audit.csv', audit, tuple(audit[0]) if audit else
                      ('record_index', 'time_s', 'status', 'ntx', 'nrx', 'rate_flags', 'valid_channels'))
            save_plot(diagnostics_figure(row, frames, data, packed, cfg), dest, 'wiresp-stages')
            details.append(f'<details><summary>{html.escape(row["label"])} | {start:g}-{start+60:g} s</summary>'
                f'<img src="{key}/wiresp-stages.png" alt="WiResP ACF enhancement and continuity stages">'
                f'<p><a href="{key}/frames.csv">Frame decisions</a> | <a href="{key}/frames.json">Full diagnostics</a> | '
                f'<a href="{key}/intermediates.npz">Numerical arrays</a> | <a href="{key}/wiresp-stages.svg">Vector figure</a> | '
                f'<a href="reference-methods/{key}/comparison.png">Previous three-method waveforms</a></p></details>')
            print(f"{key}: WiResP={row['wiresp_bpm']}, continuity={summary['accepted_frames']}/{summary['continuity_evaluated_frames']}, manual={row['manual_reference_bpm']}", flush=True)
        capture.verify()
    metrics = metrics_for(rows)
    save_plot(overview(rows), output, 'comparison-overview')
    write_csv(output/'summary.csv', rows, tuple(rows[0]))
    write_csv(output/'metrics.csv', metrics, tuple(metrics[0]))
    fields = [('label', 'Recording'), ('start_s', 'Start (s)'), ('fft_bpm', 'FFT'), ('acf_bpm', 'ACF'),
              ('farsense_bpm', 'FarSense'), ('wiresp_bpm', 'WiResP candidate'), ('manual_reference_bpm', 'Manual /min')]
    gate_fields = [('label', 'Recording'), ('start_s', 'Start (s)'), ('wiresp_candidate_frames', 'Candidate frames'),
                   ('wiresp_total_frames', 'Total frames'), ('wiresp_continuity_frames', 'Continuity evaluated'),
                   ('wiresp_accepted_frames', 'Passes adapted check'), ('wiresp_partial_after_check_bpm', 'Partial passing-frame mean')]
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>WiResP: second-batch comparison</title><style>'
        'body{font:16px system-ui;color:#183249;background:#f7f9fb;max-width:1450px;margin:32px auto;padding:0 24px;line-height:1.6}'
        'h1{line-height:1.2}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}'
        'td,th{padding:9px;text-align:left;border-bottom:1px solid #d8e1e8}.scroll{overflow:auto}'
        'section,details{background:white;border:1px solid #d8e1e8;border-radius:8px;padding:18px;margin:20px 0}'
        'summary{font-size:20px;cursor:pointer;font-weight:650}a{color:#006c9a}.note{border-left:4px solid #b47727;padding-left:14px}'
        '</style><h1>WiResP adaptation vs. FarSense, FFT and ACF</h1>'
        '<p>Second batch: independent implementation of the amplitude ACF, BNR combining, histogram enhancement and binary continuity pipeline from '
        '<a href="https://doi.org/10.1109/JSEN.2024.3399110">Wang et al., WiResP (2024)</a>.</p>'
        '<p class="note">This is an adapted reproduction, not the authors\' complete calibrated system. '
        'Several parameters are absent from the paper. Our pulse/template thresholds are explicit experimental choices. '
        f'Motion gate enabled: {cfg.motion_threshold is not None}; threshold: {cfg.motion_threshold}. '
        'No manual events enter any estimator. These gates are not calibrated on held-out activity data.</p>'
        '<section><h2>Four-method candidate comparison</h2><img src="comparison-overview.png" alt="Four-method rate comparison">'
        '<p>All four use the same predefined 60 s intervals. FFT/ACF keep the original 60 s LogRatio pipeline; '
        'FarSense keeps five 12 s complex-ratio estimates. WiResP uses normalized amplitude, 46 overlapping 15 s windows '
        'at 1 s intervals, and the mean only when all 46 yield candidates. Different features, window lengths and search bands '
        'are preserved. This compares adapted end-to-end methods, not just estimators.</p><div class="scroll">'
        +table_html([r for r in rows if not r['empty_room']], fields)+'</div></section>'
        '<section><h2>Availability and difference from manual counts</h2><div class="scroll">'
        +table_html(metrics, [('method', 'Method'), ('available_minutes', 'Available'), ('eligible_minutes', 'Eligible'),
            ('mae_available_bpm', 'MAE available /min'), ('common_minutes', 'Common'), ('mae_common_bpm', 'MAE common /min'),
            ('empty_candidate_minutes', 'Empty candidates'), ('empty_minutes', 'Empty minutes')])
        +'</div><p>Manual presses are self-reports. These reused pilot recordings are not independent subjects or held-out validation. '
        'Missing values are included in availability; available-only and common-window errors are both shown.</p></section>'
        '<section><h2>WiResP adapted continuity check, separately</h2><div class="scroll">'+table_html(rows, gate_fields)
        +'</div><p>Thirty trailing snapshots are required, so the first check is at minute-start +44 s. '
        'Earlier raw candidates remain visible. Passing-frame means cover only the shown passing frames and are not substituted '
        'for complete-minute estimates. These uncalibrated gates do not establish respiration or person presence.</p></section>'
        '<section><h2>Empty-room controls</h2><div class="scroll">'+table_html([r for r in rows if r['empty_room']], fields)
        +'</div><p>Empty-room candidates expose periodic noise. No zero-rate reference or validated false-alarm rate is assigned.</p></section>'
        '<h2>Inspect the intermediate stages</h2>'+''.join(details)
        +'<section><h2>Reproducibility</h2><p>Parameters were fixed before this run. Original timestamps and configuration codes remain in the audits. '
        'Bounded interpolation, unavailable frames, and continuity-history requirements are explicit. '
        'Source files and transfer receipts were verified unchanged.</p>'
        '<p><a href="reference-methods/index.html">Unchanged first-batch methods</a> | <a href="summary.csv">Comparison CSV</a> | '
        '<a href="metrics.csv">Metrics CSV</a> | <a href="report.json">Audit report</a> | <a href="input-manifest.json">Run manifest</a> | '
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
              'method': 'WiResP-adapted-v1', 'paper_doi': '10.1109/JSEN.2024.3399110', 'configuration': asdict(cfg),
              'source_manifest_sha256': manifest_hash, 'sources': reference['sources'],
              'results': rows, 'metrics': metrics, 'implementation_hashes': implementation,
              'raw_data_modified': False, 'manual_events_used_in_estimation': False,
              'motion_gate_enabled': cfg.motion_threshold is not None, 'original_continuity_calibration_available': False,
              'full_system_reproduction': False, 'author_code_verified': False,
              'artifacts': [{'path': str(p.relative_to(output)), 'sha256': sha256(p)}
                            for p in sorted(output.rglob('*')) if p.is_file()]}
    write_json(output/'report.json', report)
    return report
