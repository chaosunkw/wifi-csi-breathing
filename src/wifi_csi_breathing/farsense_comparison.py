"""Receipt-verified comparison of the FarSense core and existing FFT/ACF baselines."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import html
import json
import math
import re

import numpy as np

from .analysis import write_csv, write_json
from .baselines.farsense import FarSenseConfig, analyze_window, complex_ratios
from .data import FeatureConfig, sha256
from .fixed_configuration import segment_summary
from .joint import JointCapture, LIMITATION
from .rate_comparison import (PROCESSING, draw_filtered, estimate_minute, manual_reference,
                              mixed_rate_observations, observation_gaps)
from .subcarriers import new_figure, implementation_hashes


METHODS = ('LogRatio-FFT', 'LogRatio-ACF', 'FarSense-core')
COLORS = ('#0072B2', '#AD6700', '#8055B3')


def save_plot(fig, output, name):
    fig.savefig(output/(name+'.png'), dpi=160)
    fig.savefig(output/(name+'.svg'))
    fig.clear()


def validate_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1:
        raise ValueError('Expected comparison manifest schema_version 1')
    records = manifest.get('recordings')
    if not isinstance(records, list) or not 1 <= len(records) <= 100:
        raise ValueError('Expected 1..100 recordings')
    keys, jobs = set(), set()
    for record in records:
        if not isinstance(record, dict) or not re.fullmatch(r'[a-z0-9][a-z0-9-]{0,79}', record.get('key', '')):
            raise ValueError('Use a short anonymous recording key')
        if record['key'] in keys:
            raise ValueError('Duplicate recording key')
        keys.add(record['key'])
        if not isinstance(record.get('label'), str) or type(record.get('empty_room')) is not bool:
            raise ValueError('Each recording needs a label and explicit empty_room boolean')
        if not isinstance(record.get('input'), str) or not Path(record['input']).is_absolute():
            raise ValueError('Use an absolute input directory')
        starts = record.get('starts_s')
        if not isinstance(starts, list) or not 1 <= len(starts) <= 100:
            raise ValueError('Each recording needs predefined minute starts')
        for start in starts:
            if type(start) not in (int, float) or not math.isfinite(start) or start < 0:
                raise ValueError('Invalid comparison start')
            key = (str(Path(record['input']).resolve()), round(start*1e9))
            if key in jobs:
                raise ValueError('Duplicate source/time window')
            jobs.add(key)
    return records


def export_block(output, result, arrays, cfg):
    output.mkdir()
    write_json(output/'result.json', result)
    if arrays is None:
        return
    # Full numerical inputs and intermediates, without executable pickle objects.
    np.savez_compressed(output/'arrays.npz', **arrays)
    rows = []
    for group in range(30):
        angle = int(arrays['angle_indices'][group])
        rows.append({'group': group+1, 'theta_rad': angle*2*np.pi/cfg.angles if angle >= 0 else None,
                     'bnr': float(arrays['bnr'][group]), 'included': bool(arrays['included'][group])})
    write_csv(output/'subcarriers.csv', rows, tuple(rows[0]))
    write_csv(output/'projection-search.csv',
              ({'group': g+1, 'theta_rad': a*2*np.pi/cfg.angles,
                'bnr': float(arrays['angle_scores'][g, a])} for g in range(30) for a in range(cfg.angles)),
              ('group', 'theta_rad', 'bnr'))
    write_csv(output/'fused-acf.csv',
              ({'lag_s': k/cfg.sample_hz, 'weighted_acf': float(value),
                'selected_peak': k == result['lag_samples']}
               for k, value in enumerate(arrays['fused_acf'])), ('lag_s', 'weighted_acf', 'selected_peak'))


def analyze_minute(capture, start, cfg):
    """CSI first, reference second. Five nonoverlapping paper-length windows."""
    if capture.t0+round((start+60)*1e9) > capture.t1:
        raise ValueError('Comparison minute extends beyond the recording')
    observations, _ = mixed_rate_observations(capture, FeatureConfig('log_ratio'), 'first-stream')
    baseline, points = estimate_minute(observations, start, 'preserve-short')
    times, ratios, audit = complex_ratios(capture)
    blocks, intermediates = [], []
    for offset in range(5):
        result, arrays = analyze_window(times, ratios, start+12*offset, cfg)
        blocks.append(result)
        intermediates.append(arrays)
    rates = [b['candidate_bpm'] for b in blocks if b['candidate_bpm'] is not None]
    # Never silently average only convenient subwindows into a complete-minute rate.
    far_rate = float(np.mean(rates)) if len(rates) == 5 else None
    comparison = {'start_s': start, 'end_s': start+60,
                  'fft_bpm': baseline['spectral_peak_bpm'], 'acf_bpm': baseline['autocorrelation_bpm'],
                  'baseline_accepted_bpm': baseline['candidate_bpm'],
                  'baseline_status': baseline['status'], 'baseline_reason': baseline['reason'],
                  'farsense_bpm': far_rate, 'farsense_valid_blocks': len(rates), 'farsense_total_blocks': 5,
                  'farsense_available_blocks_mean_bpm': float(np.mean(rates)) if rates else None,
                  'farsense_status': 'periodic_candidate' if far_rate is not None else 'incomplete_minute',
                  'farsense_aggregation': 'arithmetic mean of five nonoverlapping 12 s rates; all five required',
                  'manual_events_used_in_estimation': False}
    comparison.update(manual_reference(capture, start))
    gaps = observation_gaps(observations, start+60, 'preserve-short')
    return comparison, baseline, points, gaps, blocks, intermediates, audit


def draw_minute(label, comparison, points, gaps, blocks, intermediates, markers, cfg):
    fig = new_figure((13, 12))
    ax1, ax2, ax3, ax4 = fig.subplots(4, 1)
    fig.subplots_adjust(left=.10, right=.97, top=.90, bottom=.09, hspace=.65)
    fig.suptitle(label, y=.98, fontsize=17)
    fig.text(.10, .94, 'FarSense core adaptation: 12 s windows; LogRatio baselines: one 60 s window. All rates are candidates.', fontsize=10)
    start, end = comparison['start_s'], comparison['end_s']
    draw_filtered(ax1, points, segment_summary(points, PROCESSING), start, end, gaps, True)
    ax1.set_title('Existing feature: causal bandpass of ln(|H_RX1| / |H_RX2|), group 15', loc='left', fontsize=11)
    ax1.set_ylabel('Filtered log ratio')
    ax2.set_title('FarSense: best-BNR projection of complex CSI ratio in each 12 s block', loc='left', fontsize=11)
    ax2.set_ylabel('Centered projection\n(unitless)')
    ax3.set_title('Rate candidates over the same minute; gray line is the received manual count', loc='left', fontsize=11)
    ax3.set_ylabel('Cycles/min')
    for value, name, color in zip((comparison['fft_bpm'], comparison['acf_bpm']), METHODS[:2], COLORS[:2]):
        if value is not None:
            ax3.axhline(value, color=color, linestyle='--', label=f'{name}: {value:.2f}')
    if comparison['manual_reference_bpm'] is not None:
        ax3.axhline(comparison['manual_reference_bpm'], color='#444444', linewidth=1.8,
                    label=f"Manual: {comparison['manual_reference_bpm']:g}")
    ax4.set_title('BNR-weighted autocorrelation, normalized for display only', loc='left', fontsize=11)
    ax4.set_xlabel('Lag (s)')
    ax4.set_ylabel('Weighted ACF / ACF(0)')
    ax4.set_xlim(0, 8)
    for i, (block, arrays) in enumerate(zip(blocks, intermediates)):
        left, right = block['start_s'], block['end_s']
        for ax in (ax1, ax2, ax3):
            if i:
                ax.axvline(left, color='#cccccc', linestyle=':', linewidth=.8)
        if arrays is not None and block.get('best_group') is not None:
            group = block['best_group']-1
            waveform = arrays['patterns'][:, group].copy()
            waveform[arrays['in_gap']] = np.nan
            ax2.plot(arrays['times'], waveform, color=COLORS[2], linewidth=1)
            ax2.text((left+right)/2, .96, f'G{group+1}', transform=ax2.get_xaxis_transform(),
                     ha='center', va='top', fontsize=8, color='#555555')
            fused = arrays['fused_acf']
            ax4.plot(np.arange(len(fused))/cfg.sample_hz, fused/max(float(fused[0]), 1e-30),
                     linewidth=1, label=f'{left:g}-{right:g} s')
            if block['candidate_bpm'] is not None:
                ax3.plot([left, right], [block['candidate_bpm']]*2, color=COLORS[2], linewidth=2)
        else:
            ax2.axvspan(left, right, color='#eeeeee')
            ax2.text((left+right)/2, .5, block['reason'].replace('_', '\n'),
                     transform=ax2.get_xaxis_transform(), ha='center', fontsize=8)
    ax3.plot([], [], color=COLORS[2], label='FarSense core: each 12 s')
    ax3.legend(fontsize=8, ncol=2, loc='upper right')
    ax4.legend(fontsize=8, ncol=3, loc='upper right')
    for ax in (ax1, ax2):
        for t in markers:
            ax.axvline(t, color='#008B78', alpha=.22, linewidth=.8)
    for ax in (ax1, ax2, ax3):
        ax.set_xlim(start, end)
        ax.set_xlabel('Seconds from common Ubuntu T0')
    for ax in fig.axes:
        ax.grid(alpha=.16)
    fig.text(.10, .025, 'Green lines: manual events (comparison only). Projection direction/group can change at each block; curves are not stitched.\n'
             'No motion gate or validated presence detector. Empty-room candidates do not establish breathing. Actual gaps remain blank.', fontsize=9)
    return fig


def summary_metrics(rows):
    occupied = [r for r in rows if not r['empty_room'] and r['manual_reference_bpm'] is not None]
    empty = [r for r in rows if r['empty_room']]
    keys = ('fft_bpm', 'acf_bpm', 'farsense_bpm')
    common = [r for r in occupied if all(r[k] is not None for k in keys)]
    metrics = []
    for method, key in zip(METHODS, keys):
        available = [r for r in occupied if r[key] is not None]
        errors = [abs(r[key]-r['manual_reference_bpm']) for r in available]
        paired_errors = [abs(r[key]-r['manual_reference_bpm']) for r in common]
        metrics.append({'method': method, 'eligible_manual_minutes': len(occupied), 'available_minutes': len(available),
                        'mae_vs_manual_available_bpm': float(np.mean(errors)) if errors else None,
                        'common_minutes': len(common),
                        'mae_vs_manual_common_bpm': float(np.mean(paired_errors)) if paired_errors else None,
                        'empty_minutes': len(empty), 'empty_candidate_minutes': sum(r[key] is not None for r in empty)})
    return metrics


def draw_summary(rows):
    occupied = [r for r in rows if not r['empty_room']]
    fig = new_figure((14, 8))
    rates, errors = fig.subplots(2, 1, sharex=True)
    fig.subplots_adjust(left=.08, right=.98, top=.86, bottom=.23, hspace=.25)
    fig.suptitle('First FarSense core comparison on existing recordings', fontsize=17, y=.98)
    fig.text(.08, .91, 'Same predefined 60 s intervals. FarSense = mean of five 12 s estimates; FFT/ACF = existing 60 s candidates.', fontsize=10)
    x = np.arange(len(occupied))
    refs = [r['manual_reference_bpm'] if r['manual_reference_bpm'] is not None else np.nan for r in occupied]
    rates.scatter(x, refs, marker='_', s=360, color='#222222', label='Manual count / min', zorder=5)
    for offset, key, name, color in zip((-.20, 0, .20), ('fft_bpm', 'acf_bpm', 'farsense_bpm'), METHODS, COLORS):
        values = np.array([r[key] if r[key] is not None else np.nan for r in occupied])
        rates.scatter(x+offset, values, color=color, label=name, s=45)
        errors.bar(x+offset, np.abs(values-np.asarray(refs)), width=.18, color=color)
    rates.set_ylabel('Candidate rate (/min)')
    errors.set_ylabel('Absolute difference\nfrom manual (/min)')
    errors.set_xticks(x, [r['label']+'\n'+f"{r['start_s']:g}-{r['end_s']:g} s" for r in occupied], rotation=25, ha='right', fontsize=9)
    rates.legend(ncol=4, fontsize=9, loc='upper right')
    for ax in (rates, errors):
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    errors.set_ylim(bottom=0)
    fig.text(.08, .035, 'FFT/ACF candidates are shown before their combined acceptance gate. Different window lengths and search bands are documented.\n'
             'Manual counts are self-reports; repeated recordings are not independent subjects. Empty-room results are listed in the report.', fontsize=9)
    return fig


def table_html(rows, fields):
    def fmt(value):
        if value is None:
            return 'Unavailable'
        return f'{value:.3f}' if isinstance(value, float) else str(value)
    return '<table><thead><tr>'+''.join('<th>'+html.escape(label)+'</th>' for _, label in fields)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(fmt(row[key]))+'</td>' for key, _ in fields)+'</tr>' for row in rows)+'</tbody></table>'


def compare_farsense(manifest_path, output, cfg=FarSenseConfig()):
    if cfg.window_seconds != 12:
        raise ValueError('This comparison requires five 12 s FarSense windows per minute')
    records = validate_manifest(manifest_path)
    output = Path(output).resolve()
    if any(output.is_relative_to(Path(r['input']).resolve()) for r in records):
        raise ValueError('Output must not be inside a source recording')
    implementation = implementation_hashes()
    output.mkdir(parents=True, exist_ok=False)
    # Keep the local run definition with ignored outputs for reproducibility.
    manifest_hash = sha256(manifest_path)
    manifest_content = json.loads(Path(manifest_path).read_text(encoding='utf-8'))
    write_json(output/'input-manifest.json', manifest_content)
    rows, sources, details = [], [], []
    for record in records:
        capture = JointCapture(record['input'])
        sources.append({'key': record['key'], 'directory': str(capture.directory),
                        'receipt_sha256': capture.receipt_hash, 'files': capture.files})
        for start in record['starts_s']:
            c, baseline, points, gaps, blocks, intermediates, audit = analyze_minute(capture, start, cfg)
            key = f"{record['key']}-{start:g}-{start+60:g}s"
            dest = output/key
            dest.mkdir()
            c.update(key=key, label=record['label'], empty_room=record['empty_room'])
            if record['empty_room']:
                c['manual_reference_bpm'] = None
                c['reference_status'] = 'empty_control_not_zero_respiration_truth'
            for method_key in ('fft_bpm', 'acf_bpm', 'farsense_bpm'):
                c[method_key+'_absolute_difference'] = (abs(c[method_key]-c['manual_reference_bpm'])
                    if c[method_key] is not None and c['manual_reference_bpm'] is not None else None)
            rows.append(c)
            for i, (block, arrays) in enumerate(zip(blocks, intermediates)):
                export_block(dest/f'block-{i+1}', block, arrays, cfg)
            write_json(dest/'baseline.json', baseline)
            write_json(dest/'comparison.json', c)
            write_json(dest/'farsense-blocks.json', blocks)
            write_csv(dest/'packet-audit.csv', audit, ('record_index', 'time_s', 'status', 'ntx', 'nrx', 'rate_flags', 'valid_groups'))
            markers = [(e['host_monotonic_ns']-capture.t0)/1e9 for e in capture.events
                       if start <= (e['host_monotonic_ns']-capture.t0)/1e9 < start+60]
            write_json(dest/'manual-events.json', {'relative_seconds': markers, 'used_in_estimation': False})
            save_plot(draw_minute(record['label']+f' | {start:g}-{start+60:g} s', c, points, gaps,
                                 blocks, intermediates, markers, cfg), dest, 'comparison')
            block_table = table_html(blocks, [('start_s', 'Start (s)'), ('end_s', 'End (s)'),
                ('candidate_bpm', 'FarSense candidate (/min)'), ('valid_groups', 'Complete groups'),
                ('selected_groups', 'BNR-selected groups'), ('bridged_grid_fraction', 'Gap fraction'), ('reason', 'Decision')])
            links = ' | '.join(f'<a href="{key}/block-{i+1}/result.json">Block {i+1} audit</a>' for i in range(5))
            details.append(f'<details><summary>{html.escape(c["label"])} | {start:g}-{start+60:g} s</summary>'
                           f'<img src="{key}/comparison.png" alt="Waveforms, rates and fused autocorrelation">'
                           +block_table+f'<p><a href="{key}/comparison.svg">Vector figure</a> | {links}</p></details>')
            print(f"{key}: FFT={c['fft_bpm']}, ACF={c['acf_bpm']}, FarSense={c['farsense_bpm']}, manual={c['manual_reference_bpm']}", flush=True)
        capture.verify()
    metrics = summary_metrics(rows)
    save_plot(draw_summary(rows), output, 'comparison-overview')
    write_csv(output/'summary.csv', rows, tuple(rows[0]))
    write_csv(output/'metrics.csv', metrics, tuple(metrics[0]))
    fields = [('label', 'Recording'), ('start_s', 'Start'), ('end_s', 'End'), ('fft_bpm', 'FFT /min'),
              ('acf_bpm', 'ACF /min'), ('farsense_bpm', 'FarSense core /min'),
              ('farsense_valid_blocks', 'Valid blocks / 5'), ('received_button_count', 'Buttons'),
              ('baseline_status', 'Existing combined gate')]
    page = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<title>FarSense core: first comparison</title><style>'
        'body{font:16px system-ui;color:#183249;background:#f7f9fb;max-width:1400px;margin:32px auto;padding:0 24px;line-height:1.6}'
        'h1{line-height:1.2}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}'
        'td,th{padding:9px;text-align:left;border-bottom:1px solid #d8e1e8}.scroll{overflow:auto}'
        'section,details{background:white;border:1px solid #d8e1e8;border-radius:8px;padding:18px;margin:20px 0}'
        'summary{font-size:20px;cursor:pointer;font-weight:650}a{color:#006c9a}.note{border-left:4px solid #b47727;padding-left:14px}'
        '</style><h1>FarSense core vs. LogRatio FFT and ACF</h1>'
        '<p>Independent implementation of the complex-ratio, projection and BNR-weighted autocorrelation core from '
        '<a href="https://doi.org/10.1145/3351279">Zeng et al., FarSense (2019)</a>. '
        'This is an adapted core reproduction, not the authors\' full system or a replication of their reported accuracy.</p>'
        '<p class="note">Parameters were fixed before running these comparisons. The large-motion exclusion module is not implemented. '
        'Candidates are not validated respiration or presence decisions. No button events select projections, groups, peaks or parameters.</p>'
        '<section><h2>Occupied recordings</h2><img src="comparison-overview.png" alt="Candidate rates and absolute differences">'
        '<p>FFT and ACF are the existing 60 s LogRatio candidates before the combined four-check gate. FarSense uses five '
        'nonoverlapping 12 s windows and reports their mean only if all five produce candidates. A missing block remains missing. '
        'The observation interval is shared; internal window lengths and search bands differ (6-30 vs. 10-37 /min).</p><div class="scroll">'
        +table_html([r for r in rows if not r['empty_room']], fields)+'</div></section>'
        '<section><h2>Availability and differences from manual counts</h2><div class="scroll">'
        +table_html(metrics, [('method', 'Method'), ('available_minutes', 'Available minutes'),
            ('eligible_manual_minutes', 'Eligible minutes'), ('mae_vs_manual_available_bpm', 'MAE on available'),
            ('common_minutes', 'Common minutes'), ('mae_vs_manual_common_bpm', 'MAE on common minutes'),
            ('empty_candidate_minutes', 'Empty candidates'), ('empty_minutes', 'Empty minutes')])
        +'</div><p>Units: cycles/min. These are descriptive errors on a small reused pilot dataset, not a held-out performance claim.</p></section>'
        '<section><h2>Empty-room controls</h2><div class="scroll">'+table_html([r for r in rows if r['empty_room']], fields)
        +'</div><p>Zero manual presses are not a zero-rate physiological reference. A numerical peak here exposes why candidate '
        'generation alone is not a presence detector. The original combined gate is shown separately.</p></section>'
        '<h2>Inspect every predefined minute</h2>'+''.join(details)
        +'<section><h2>What was reproduced and adapted</h2><p>Complex division RX1/RX2 for stream 1 and all 30 groups; '
        'Savitzky-Golay smoothing; 100 projections at pi/50 spacing; strongest respiratory FFT-bin BNR; '
        'subcarriers above 0.7 of the maximum BNR; normalized ACF per group; BNR-weighted fusion and first positive peak.</p>'
        '<p>Project adaptation: 20 Hz resampling, 2048-point padded FFT, 11-sample/order-3 smoothing, mean subtraction before BNR, '
        'and peak search limited to the paper\'s 10-37 /min band. The paper used 100 Hz and 8192 FFT points. '
        'Its smoothing parameters and several numeric conventions were not fully specified. Stream-1 pooling and bounded '
        'short-gap interpolation follow this project; they do not establish identical physical transmit mapping.</p>'
        '<p>Smoothing/projection use only a completed 12 s block, with availability at its end or its bounded right interpolation '
        'endpoint. Projected sample curves are retrospective within that block. No future block enters an earlier rate.</p>'
        f'<p>{LIMITATION}</p><p><a href="summary.csv">Comparison CSV</a> | <a href="metrics.csv">Metrics CSV</a> | '
        '<a href="report.json">Full audit</a> | <a href="input-manifest.json">Run manifest</a> | '
        '<a href="comparison-overview.svg">Vector overview</a></p></section></html>')
    (output/'index.html').write_text(page, encoding='utf-8')
    if implementation_hashes() != implementation:
        raise ValueError('Implementation changed during comparison')
    if sha256(manifest_path) != manifest_hash:
        raise ValueError('Input manifest changed during comparison')
    # Verify all sessions again, not only the last session processed.
    for source in sources:
        if sha256(Path(source['directory'])/'transfer.json') != source['receipt_sha256']:
            raise ValueError('Source receipt changed during batch')
        for item in source['files']:
            if sha256(Path(source['directory'])/item['name']) != item['sha256']:
                raise ValueError('Source recording changed during batch')
    report = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'method': 'FarSense-core-adapted-v1', 'paper_doi': '10.1145/3351279', 'configuration': asdict(cfg),
              'source_manifest_sha256': manifest_hash, 'sources': sources, 'results': rows,
              'metrics': metrics, 'implementation_hashes': implementation,
              'raw_data_modified': False, 'manual_events_used_in_estimation': False,
              'motion_gate_implemented': False, 'author_code_verified': False,
              'artifacts': [{'path': str(p.relative_to(output)), 'sha256': sha256(p)}
                            for p in sorted(output.rglob('*')) if p.is_file()]}
    write_json(output/'report.json', report)
    return report
