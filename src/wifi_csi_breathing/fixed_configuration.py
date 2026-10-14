"""Opt-in, fixed-configuration projection with bounded missing-sample gaps.

This does not configure a radio or prove that its spatial mapping is stable.
The original strict joint/replay processing policy is unchanged.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import html
import math

from .analysis import process_observations, write_csv, write_json
from .data import FeatureConfig, MODES, Observation, feature, sha256
from .joint import EVENT_FIELDS, JointCapture, LIMITATION
from .processing import ProcessingConfig
from .static_analysis import NAMES, UNITS
from .subcarriers import implementation_hashes, new_figure, save_figure


def fixed_observations(capture, config, rate_flags):
    """Keep only explicitly selected (3, 3, rate_flags) packet features.

    Other configurations are auditable missing samples. The unchanged Processor
    measures elapsed time between retained samples, so short gaps may be
    interpolated only when a right endpoint arrives. Long gaps still reset.
    Invalid matching packets, duplicate times and unknown records remain hard
    barriers. Neither future packets nor marker timing choose a configuration.
    """
    if type(rate_flags) is not int or not 0 <= rate_flags <= 65535:
        raise ValueError('Rate flags must be an explicit integer in 0..65535')
    observations, audit, previous_ns = [], [], None
    for row, packet in zip(capture.index, capture.packets):
        ns = row['host_monotonic_ns']
        if not capture.inside(ns):
            continue
        t = (ns - capture.t0) / 1e9
        value, status = None, 'kept'
        if previous_ns is not None and ns < previous_ns:
            raise ValueError('Backward host time in fixed-configuration input')
        if ns == previous_ns:
            status = 'duplicate_timestamp'
        elif packet is None:
            status = 'non_csi_record'
        elif (packet['ntx'], packet['nrx']) != (3, 3):
            status = 'other_shape'
        elif packet['rate'] != rate_flags:
            status = 'other_rate_flags'
        else:
            value = feature(packet, config)
            if value is None or not math.isfinite(value):
                value, status = None, 'invalid_feature'
        previous_ns = ns
        audit.append({'record_index': row['record_index'], 'time': t,
                      'ntx': packet['ntx'] if packet else None,
                      'nrx': packet['nrx'] if packet else None,
                      'rate_flags': packet['rate'] if packet else None,
                      'status': status, 'value': value})
        if status in ('other_shape', 'other_rate_flags'):
            continue
        observations.append(Observation(t, value, status != 'kept',
                                        '' if status == 'kept' else status, row['record_index']))
    return observations, audit


def segment_summary(points, processing):
    groups = defaultdict(list)
    for point in points:
        groups[point.segment].append(point)
    return [{'segment': key, 'start_s': group[0].time, 'end_s': group[-1].time,
             'duration_s': group[-1].time - group[0].time,
             'warmup_until_s': group[0].time + processing.warmup_seconds}
            for key, group in groups.items()]


def draw_fixed(capture, config, rate_flags, observations, points, processing):
    fig = new_figure((12, 8.6))
    raw, filtered, marker = fig.subplots(3, 1, sharex=True,
        gridspec_kw={'height_ratios': [2, 2, .7]})
    fig.subplots_adjust(left=.10, right=.97, top=.85, bottom=.15, hspace=.32)
    valid = [o for o in observations if o.value is not None]
    raw.scatter([o.time for o in valid], [o.value for o in valid], s=3, color='#0072B2')
    raw.set_title('Retained packet feature (unfiltered; excluded configurations are absent)', loc='left', fontsize=10)
    raw.set_ylabel(UNITS[config.mode])
    for span in segment_summary(points, processing):
        group = [p for p in points if p.segment == span['segment']]
        settled = [p for p in group if p.time >= span['warmup_until_s'] - 1e-9]
        if settled:
            filtered.plot([p.time for p in settled], [p.filtered for p in settled], color='#0072B2', linewidth=1)
        filtered.axvspan(span['start_s'], min(span['warmup_until_s'], span['end_s']),
                         color='#dddddd', alpha=.65)
    filtered.set_title('Existing causal 0.1-0.5 Hz bandpass; gray = warm-up (output hidden)', loc='left', fontsize=10)
    filtered.set_ylabel('Filtered feature\n' + UNITS[config.mode])
    times = [(e['host_monotonic_ns'] - capture.t0) / 1e9 for e in capture.events
             if capture.inside(e['host_monotonic_ns'])]
    marker.vlines(times, 0, 1, color='#008678', linewidth=1)
    marker.set_ylim(-.1, 1.1)
    marker.set_yticks([])
    marker.set_title('Manual button events (host receipt; not shifted to match the waveform)', loc='left', fontsize=10)
    marker.set_xlabel('Seconds from common Ubuntu T0')
    for ax in (raw, filtered, marker):
        ax.set_xlim(0, capture.duration)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle(f'{NAMES[config.mode]} | fixed 3 x 3, rate flags {rate_flags}', fontsize=15, y=.98)
    fig.text(.10, .914, f'Stream {config.tx}, RX {config.rx}, reference RX {config.reference_rx}, group {config.subcarrier}. '
             f'Only gaps <= {processing.max_gap_seconds:g} s may be interpolated.\n'
             'Long gaps / invalid matching packets still reset. No channel search or manual-marker fitting.', fontsize=10)
    fig.text(.10, .045, 'Exploratory fixed-configuration assumption: identical reported flags do not prove an unchanged physical channel.\n'
             'The causal filter has delay; filtered peaks are not inhalation-onset timestamps. Manual events are not physiological ground truth.', fontsize=9)
    return fig


def analyze_fixed(directory, output, rate_flags, modes=MODES, tx=1, rx=1, reference_rx=2, subcarrier=15):
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Select at least one non-duplicate mode')
    configs = [FeatureConfig(m, tx, rx, reference_rx, subcarrier) for m in modes]
    capture = JointCapture(directory)
    implementation = implementation_hashes()
    prepared = [(config, *fixed_observations(capture, config, rate_flags)) for config in configs]
    if not any(any(o.value is not None for o in observations) for _, observations, _ in prepared):
        raise ValueError('No valid packets match the explicit 3x3 / rate-flags selection')
    output = Path(output).resolve()
    if output.is_relative_to(capture.directory):
        raise ValueError('Output must not be inside the source recording')
    output.mkdir(parents=True, exist_ok=False)
    processing = ProcessingConfig()
    event_rows = [{**e, 'relative_seconds': (e['host_monotonic_ns']-capture.t0)/1e9,
                   'in_window': capture.inside(e['host_monotonic_ns'])} for e in capture.events]
    write_csv(output / 'markers.csv', event_rows, (*EVENT_FIELDS, 'relative_seconds', 'in_window'))
    results = []
    for config, observations, audit in prepared:
        print(f'Analyzing fixed configuration: {config.mode}', flush=True)
        processor, points, estimates = process_observations(observations, config.mode, processing)
        write_csv(output / f'{config.mode}-selection.csv', audit,
                  ('record_index', 'time', 'ntx', 'nrx', 'rate_flags', 'status', 'value'))
        write_csv(output / f'{config.mode}-observations.csv', (asdict(o) for o in observations),
                  ('time', 'value', 'boundary', 'reason', 'index'))
        write_csv(output / f'{config.mode}-waveform.csv', (asdict(p) for p in points),
                  ('time', 'available_at', 'raw', 'filtered', 'segment'))
        write_csv(output / f'{config.mode}-estimates.csv', (asdict(e) for e in estimates),
                  ('time', 'available_at', 'segment', 'status', 'candidate_bpm', 'spectral_peak_bpm',
                   'autocorrelation_bpm', 'autocorrelation', 'concentration', 'band_fraction'))
        save_figure(draw_fixed(capture, config, rate_flags, observations, points, processing), output, config.mode + '-fixed')
        spans = segment_summary(points, processing)
        candidates = [e.candidate_bpm for e in estimates if e.candidate_bpm is not None]
        # Preserve the original strict policy as a comparison, never replace it.
        strict_processor, strict_points, strict_estimates = process_observations(capture.observations(config), config.mode, processing)
        results.append({'feature': asdict(config), 'selection_counts': dict(Counter(a['status'] for a in audit)),
            'filter_resets': processor.resets, 'segments': spans,
            'longest_segment_s': max((s['duration_s'] for s in spans), default=0.),
            'eligible_estimate_windows': len(estimates), 'periodic_candidate_windows': len(candidates),
            'candidate_bpm_range': [min(candidates), max(candidates)] if candidates else None,
            'strict_baseline': {'filter_resets': strict_processor.resets, 'eligible_estimate_windows': len(strict_estimates),
                'longest_segment_s': max((s['duration_s'] for s in segment_summary(strict_points, processing)), default=0.)}})
    rows = ''.join('<tr>' + ''.join(f'<td>{html.escape(str(v))}</td>' for v in
        (r['feature']['mode'], r['selection_counts'].get('kept', 0), f"{r['longest_segment_s']:.2f}",
         r['eligible_estimate_windows'], r['periodic_candidate_windows'], r['candidate_bpm_range'])) + '</tr>' for r in results)
    plots = ''.join(f'<h2>{NAMES[m]}</h2><img src="{m}-fixed.png" alt="{m} fixed configuration">'
        f'<p><a href="{m}-fixed.pdf">PDF</a> | <a href="{m}-fixed.svg">SVG</a> | '
        f'<a href="{m}-selection.csv">Packet selection audit</a> | <a href="{m}-estimates.csv">All estimate windows</a></p>' for m in modes)
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>Fixed-configuration CSI exploration</title>'
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:0 20px}img{width:100%}'
        'p{line-height:1.6}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}</style>'
        f'<h1>Fixed-configuration CSI exploration</h1><p>Explicit selection: 3x3, rate flags {rate_flags}. '
        'No automatic rate or subcarrier ranking. Only selected packet values enter the unchanged causal processor. '
        'All other configurations remain in the originals and are listed as missing observations in the selection CSV.</p>'
        '<p>Short gaps of at most 0.2 s between retained samples may be interpolated only after the next sample arrives. '
        'Long gaps, duplicate timestamps, non-CSI records and invalid selected features reset processing. '
        'This assumes that returning to identical reported flags recovers a comparable channel; hidden spatial mapping, '
        'antenna selection or hardware changes remain possible. This is an opt-in research analysis, not a radio fix.</p>'
        '<p>The 10 s warm-up and 30 s estimation window are unchanged. The original strict joint command and replay GUI '
        'are unchanged. There is no marker-based phase alignment. A filtered waveform or periodic candidate is not proof of respiration.</p>'
        f'<p>{LIMITATION}</p><table><tr><th>Feature</th><th>Retained packets</th><th>Longest segment (s)</th>'
        '<th>Eligible windows</th><th>Periodic candidates</th><th>Candidate range (/min)</th></tr>' + rows + '</table>' + plots + '</html>', encoding='utf-8')
    capture.verify()
    if implementation_hashes() != implementation:
        raise ValueError('Implementation changed during export; use a new output directory')
    report = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'session_id': capture.session_id, 'source_directory': str(capture.directory), 'source_files': capture.files,
        'receipt_sha256': capture.receipt_hash, 'timing': capture.timing,
        'policy': {'name': 'explicit-fixed-configuration', 'shape': [3,3], 'rate_flags': rate_flags,
                   'short_gap_interpolation': True, 'automatically_selected': False,
                   'physical_mapping_stability_verified': False},
        'processing': asdict(processing), 'results': results, 'implementation_hashes': implementation,
        'raw_data_modified': False, 'radio_configuration_modified': False, 'physiological_validation': False,
        'limitation': LIMITATION,
        'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / 'fixed-configuration-report.json', report)
    return report
