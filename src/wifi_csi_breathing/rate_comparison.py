"""One predefined minute of log-ratio periodicity versus manual button counts.

Exploratory pooling of all reported rates within 3x3 by default, with an opt-in
first-stream projection across 1x3/2x3/3x3. No radio configuration, marker-driven
fitting, automatic channel selection or physiological validation.
"""
from collections import Counter
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
import html
import math

from .analysis import write_csv, write_json
from .data import FeatureConfig, Observation, feature, sha256
from .fixed_configuration import segment_summary
from .joint import EVENT_FIELDS, JointCapture, LIMITATION
from .processing import ProcessingConfig, Processor, estimate_window
from .rate_diagnostics import export_diagnostics
from .subcarriers import implementation_hashes, new_figure, save_figure


PROCESSING = ProcessingConfig(window_seconds=60.0)
SHAPE_POLICIES = ('3x3', 'first-stream')
SHAPE_COLORS = {1: '#D55E00', 2: '#8E6BBE', 3: '#0072B2'}
GAP_POLICIES = ('reset', 'preserve-short')
SHORT_GAP_LIMIT = 0.5
MAX_BRIDGED_FRACTION = 0.05


def gap_processing(gap_policy):
    if gap_policy not in GAP_POLICIES:
        raise ValueError('Unknown gap policy')
    return (PROCESSING if gap_policy == 'reset' else
            replace(PROCESSING, max_gap_seconds=SHORT_GAP_LIMIT))


def observation_gaps(observations, end_seconds, gap_policy):
    """Observed feature intervals, not proof of over-air packet loss.

    A bridge is known only when both valid endpoints have arrived. Its values
    are numerical interpolation, never observations or displayed measurements.
    Invalid records remain barriers, even when their timestamps are close.
    """
    processing = gap_processing(gap_policy)
    gaps, previous = [], None
    for observation in observations:
        if observation.time > end_seconds + processing.max_gap_seconds:
            break
        if previous is not None and observation.time - previous.time > PROCESSING.max_gap_seconds:
            valid = all(o.value is not None and math.isfinite(o.value) and not o.boundary
                        for o in (previous, observation))
            duration = observation.time - previous.time
            gaps.append({'start_s': previous.time, 'end_s': observation.time,
                         'duration_s': duration, 'available_at_s': observation.time,
                         'action': ('interpolated_state_preserved' if valid and
                                    gap_policy == 'preserve-short' and duration <= SHORT_GAP_LIMIT
                                    else 'reset')})
        previous = observation
        if round(observation.time * 1e9) >= round(end_seconds * 1e9):
            break
    return gaps


def inside_gap(time, gaps):
    return any(g['start_s'] + 1e-9 < time < g['end_s'] - 1e-9 for g in gaps)


def draw_filtered(ax, points, spans, start, end, gaps=(), show_transients=False,
                  color='#0072B2'):
    """Leave true gap intervals blank; optional dashed, visible filter settling."""
    for span in spans:
        group = [p for p in points if p.segment == span['segment'] and start <= p.time < end]
        if not group:
            continue
        times = [p.time for p in group]
        values = [p.filtered if not inside_gap(p.time, gaps) else math.nan for p in group]
        stable = [p.time >= span['warmup_until_s'] - 1e-9 for p in group]
        ax.plot(times, [v if ok else math.nan for v, ok in zip(values, stable)],
                color=color, linewidth=1)
        if show_transients:
            ax.plot(times, [v if not ok else math.nan for v, ok in zip(values, stable)],
                    color='#B36B00', linestyle='--', linewidth=1)
        else:
            left, right = max(start, span['start_s']), min(end, span['warmup_until_s'], span['end_s'])
            if left < right:
                ax.axvspan(left, right, color='#dddddd', alpha=.6)


def validate_shape_policy(shape_policy, config):
    if shape_policy not in SHAPE_POLICIES:
        raise ValueError('Unknown shape policy')
    if shape_policy == 'first-stream' and config.tx != 1:
        raise ValueError('The first-stream policy requires --tx 1')


def first_stream_ratio(packet, config):
    """Common array indices, not a claim of identical physical spatial mapping."""
    a = packet['csi'][0][config.rx - 1][config.subcarrier - 1]
    b = packet['csi'][0][config.reference_rx - 1][config.subcarrier - 1]
    magnitudes = abs(a), abs(b)
    if not all(math.isfinite(v) and v > 0 for v in magnitudes):
        return None
    return math.log(magnitudes[0]) - math.log(magnitudes[1])


def window_bounds(start_seconds):
    if not math.isfinite(start_seconds) or start_seconds < 0 or not math.isfinite(start_seconds * 1e9):
        raise ValueError('Window start must be finite and nonnegative')
    start_ns = round(start_seconds * 1e9)
    return start_ns, start_ns + 60_000_000_000


def mixed_rate_observations(capture, config, shape_policy='3x3'):
    """Keep valid selected log ratios, with every indexed packet audited.

Other shapes are missing samples. Duplicate times (even on another shape),
unknown records and invalid selected features remain hard filter barriers.
The opt-in policy retains stream 1 of 1x3/2x3/3x3 without shape-only resets,
offset correction, stream ranking or imputation of absent streams.
"""
    if config.mode != 'log_ratio':
        raise ValueError('The minute comparison supports log_ratio only')
    validate_shape_policy(shape_policy, config)
    observations, audit, previous_ns = [], [], None
    previous_shape = None
    for row, packet in zip(capture.index, capture.packets):
        ns = row['host_monotonic_ns']
        if not capture.inside(ns):
            continue
        time = (ns - capture.t0) / 1e9
        shape = (packet['ntx'], packet['nrx']) if packet else None
        changed = shape is not None and previous_shape is not None and shape != previous_shape
        previous_shape = shape
        value, status = None, 'kept'
        if previous_ns is not None and ns < previous_ns:
            raise ValueError('Backward host time in rate-comparison input')
        if ns == previous_ns:
            status = 'duplicate_timestamp'
        elif packet is None:
            status = 'non_csi_record'
        elif shape not in (((3, 3),) if shape_policy == '3x3' else ((1, 3), (2, 3), (3, 3))):
            status = 'other_shape'
        else:
            value = (feature(packet, config) if shape_policy == '3x3'
                     else first_stream_ratio(packet, config))
            if value is None or not math.isfinite(value):
                value, status = None, 'invalid_feature'
        previous_ns = ns
        audit.append({'record_index': row['record_index'], 'time': time,
                      'ntx': packet['ntx'] if packet else None,
                      'nrx': packet['nrx'] if packet else None,
                      'rate_flags': packet['rate'] if packet else None,
                      'status': status, 'value': value, 'shape_changed': changed})
        if status != 'other_shape':
            observations.append(Observation(time, value, status != 'kept',
                                            '' if status == 'kept' else status, row['record_index']))
    return observations, audit


def estimate_minute(observations, start_seconds=10.0, gap_policy='reset'):
    """Estimate the exact [start, start+60) interval, never a shifted good window.

Filter all available prehistory. A right interpolation endpoint may arrive at
most the selected interpolation limit after the end; available_at records this
latency. Later samples cannot change the result. The original reset policy uses
0.2 s. Opt-in preserve-short uses 0.5 s and audits/caps the added interpolation.
"""
    processing = gap_processing(gap_policy)
    start_ns, end_ns = window_bounds(start_seconds)
    start, end = start_ns / 1e9, end_ns / 1e9
    gaps = observation_gaps(observations, end, gap_policy)
    processor = Processor('log_ratio', processing)
    points = []
    for observation in observations:
        if observation.time > end + processing.max_gap_seconds:
            break
        emitted, _ = processor.feed(observation)
        points.extend(emitted)
        if round(observation.time * 1e9) >= end_ns:
            break
    selected = [p for p in points if start_ns <= round(p.time * 1e9) < end_ns]
    spans = segment_summary(points, PROCESSING)
    settled_after = {s['segment']: s['warmup_until_s'] for s in spans}
    settled = [p for p in selected if p.time >= settled_after[p.segment] - 1e-9]
    expected = round(60 * PROCESSING.sample_hz)
    # Count occupied grid cells, not raw packets. Interpolated samples are included.
    cells = {min(expected - 1, max(0, int((p.time - start) * PROCESSING.sample_hz + 1e-7)))
             for p in settled}
    result = {'start_s': start, 'end_s': end, 'duration_s': 60.0,
              'status': 'insufficient_data', 'reason': '', 'candidate_bpm': None,
              'spectral_peak_bpm': None, 'autocorrelation_bpm': None,
              'autocorrelation': None, 'concentration': None, 'band_fraction': None,
              'available_at_s': None, 'expected_grid_samples': expected,
              'settled_grid_samples': len(settled),
              'supported_grid_fraction': len(cells) / expected,
              'filter_resets': processor.resets}
    bridges = [g for g in gaps if g['action'] == 'interpolated_state_preserved']
    bridged_count = sum(inside_gap(p.time, bridges) for p in selected)
    if gap_policy == 'preserve-short':
        result.update(gap_policy=gap_policy, bridged_grid_samples=bridged_count,
                      bridged_grid_fraction=bridged_count / expected,
                      settled_unbridged_grid_samples=sum(not inside_gap(p.time, bridges) for p in settled),
                      bridged_intervals=sum(g['start_s'] < end and g['end_s'] > start for g in bridges))
    if len({p.segment for p in selected}) > 1:
        result['reason'] = 'filter_reset_inside_window'
    elif len(selected) != expected:
        result['reason'] = 'incomplete_60s_grid'
    elif len(settled) != expected:
        result['reason'] = 'filter_warmup_inside_window'
    elif bridged_count / expected > MAX_BRIDGED_FRACTION:
        result['reason'] = 'excessive_gap_interpolation'
    elif any(not math.isclose(b.time - a.time, 1 / PROCESSING.sample_hz, abs_tol=1e-7)
             for a, b in zip(selected, selected[1:])):
        result['reason'] = 'nonuniform_grid'
    else:
        estimate = estimate_window(selected, PROCESSING)
        for key in ('status', 'candidate_bpm', 'spectral_peak_bpm', 'autocorrelation_bpm',
                    'autocorrelation', 'concentration', 'band_fraction'):
            result[key] = getattr(estimate, key)
        result['available_at_s'] = max(end, estimate.available_at)
        result['reason'] = ('frequency_and_autocorrelation_agree' if estimate.candidate_bpm is not None
                            else estimate.status)
    return result, points


def manual_reference(capture, start_seconds=10.0):
    """Count received events in the same half-open window, without timing fits.

Known sequence gaps overlapping the window and sender changes require review.
Zero received events do not establish a zero physiological respiration rate.
"""
    start, end = window_bounds(start_seconds)
    start += capture.t0
    end += capture.t0
    events = [e for e in capture.events if start <= e['host_monotonic_ns'] < end]
    issues = set()
    previous_ns = capture.t0
    for event in capture.events:
        ns = event['host_monotonic_ns']
        if event['missing_before'] and ns >= start and previous_ns < end:
            issues.add('sequence_gap_overlaps_window')
        previous_ns = ns
    if not events:
        issues.add('no_manual_events')
    if len({e['sender_address'] for e in events}) > 1:
        issues.add('multiple_senders')
    if len({(e['sender_address'], e['boot_id']) for e in events}) > 1:
        issues.add('sender_boot_changed')
    if any(e['quality'] not in ('ok', 'first_marker') for e in events):
        issues.add('marker_quality_requires_review')
    return {'received_button_count': len(events),
            'received_count_bpm': float(len(events)),
            'manual_reference_bpm': float(len(events)) if not issues else None,
            'reference_status': 'review_required' if issues else 'manual_reference',
            'reference_issues': ';'.join(sorted(issues))}


def compare_reference(csi, manual):
    result = {**csi, **manual}
    candidate, reference = result['candidate_bpm'], result['manual_reference_bpm']
    result['absolute_error_bpm'] = (abs(candidate - reference)
                                    if candidate is not None and reference is not None else None)
    return result


def draw_comparison(observations, points, capture, result, config, shape_policy='3x3', audit=(),
                    gap_policy='reset', gaps=()):
    exploratory = shape_policy == 'first-stream'
    fig = new_figure((12, 10) if exploratory else (12, 8))
    if exploratory:
        raw, filtered, shapes, marker = fig.subplots(4, 1, sharex=True,
            gridspec_kw={'height_ratios': [2, 2, .65, .7]})
    else:
        raw, filtered, marker = fig.subplots(3, 1, sharex=True,
            gridspec_kw={'height_ratios': [2, 2, .7]})
    fig.subplots_adjust(left=.10, right=.97, top=.84, bottom=.13, hspace=.42)
    start, end = result['start_s'], result['end_s']
    valid = [o for o in observations if o.value is not None and start <= o.time < end]
    if exploratory:
        for ntx, color in SHAPE_COLORS.items():
            selected = [a for a in audit if a['status'] == 'kept' and a['ntx'] == ntx
                        and start <= a['time'] < end]
            raw.scatter([a['time'] for a in selected], [a['value'] for a in selected],
                        s=4, color=color, label=f'{ntx} x 3')
            reported = [a for a in audit if a['ntx'] == ntx and start <= a['time'] < end]
            shapes.scatter([a['time'] for a in reported], [ntx] * len(reported), s=3, color=color)
        raw.legend(loc='upper right', ncol=3, fontsize=8)
        shapes.set_yticks([1, 2, 3])
        shapes.set_ylim(.7, 3.3)
        shapes.set_ylabel('Ntx')
        shapes.set_title('Reported stream count | transitions retained, not corrected', loc='left', fontsize=10)
        raw.set_title('Packet log amplitude ratio | first stream across 1 x 3, 2 x 3, 3 x 3', loc='left', fontsize=11)
    else:
        raw.scatter([o.time for o in valid], [o.value for o in valid], s=3, color='#0072B2')
        raw.set_title('Packet log amplitude ratio | all reported rates within 3 x 3', loc='left', fontsize=11)
    raw.set_ylabel('Log ratio (unitless)')
    draw_filtered(filtered, points, segment_summary(points, PROCESSING), start, end,
                  gaps=gaps, show_transients=gap_policy == 'preserve-short')
    filtered.set_title('Causal 0.1-0.5 Hz bandpass | ' +
                       ('blank = gap; dashed amber = settling (not rate eligible)' if gap_policy == 'preserve-short'
                        else 'gray = warm-up; gaps remain visible'), loc='left', fontsize=10)
    filtered.set_ylabel('Filtered log ratio')
    a, b = window_bounds(start)
    times = [(e['host_monotonic_ns'] - capture.t0) / 1e9 for e in capture.events
             if capture.t0 + a <= e['host_monotonic_ns'] < capture.t0 + b]
    marker.vlines(times, 0, 1, color='#008678', linewidth=1)
    marker.set_ylim(-.1, 1.1)
    marker.set_yticks([])
    marker.set_title(f"Manual button events | {len(times)} received in [{start:g}, {end:g}) s", loc='left', fontsize=11)
    marker.set_xlabel('Seconds from common Ubuntu T0 (host receipt)')
    for ax in fig.axes:
        ax.set_xlim(start, end)
        ax.grid(axis='x', alpha=.2)
    fig.suptitle('Exploratory 60 s respiratory-rate comparison', fontsize=16, y=.98)
    def value(number):
        return 'unavailable' if number is None else f'{number:.2f} /min'
    fig.text(.10, .92, f"CSI candidate: {value(result['candidate_bpm'])} | Manual reference: "
             f"{value(result['manual_reference_bpm'])} | Absolute difference: {value(result['absolute_error_bpm'])}", fontsize=11)
    fig.text(.10, .88, f"Stream {config.tx}, RX {config.rx}, reference RX {config.reference_rx}, group {config.subcarrier}. "
             f"Status: {result['status']}; {result['reason']}.", fontsize=9)
    fig.text(.10, .04, 'Manual events are used only for the final comparison. No marker fitting or time shift.\n'
             'Rate/shape pooling is exploratory; configuration jumps can mimic motion. A periodic candidate is not validated respiration.', fontsize=9)
    return fig


def analyze_rate(directory, output, start_seconds=10.0, tx=1, rx=1, reference_rx=2, subcarrier=15,
                 shape_policy='3x3', gap_policy='reset'):
    processing = gap_processing(gap_policy)
    start_ns, end_ns = window_bounds(start_seconds)
    config = FeatureConfig('log_ratio', tx, rx, reference_rx, subcarrier)
    validate_shape_policy(shape_policy, config)
    capture = JointCapture(directory)
    if capture.t0 + end_ns > capture.t1:
        raise ValueError('The complete 60 s comparison window must fit inside the recording')
    implementation = implementation_hashes()
    observations, audit = mixed_rate_observations(capture, config, shape_policy)
    csi, points = estimate_minute(observations, start_seconds, gap_policy)
    gaps = observation_gaps(observations, end_ns / 1e9, gap_policy)
    result = compare_reference(csi, manual_reference(capture, start_seconds))
    window_audit = [a for a in audit if start_ns <= round(a['time'] * 1e9) < end_ns]
    kept = [a for a in window_audit if a['status'] == 'kept']
    result.update(indexed_records=len(window_audit), retained_packets=len(kept),
                  retained_packet_fraction=len(kept) / len(window_audit) if window_audit else None)
    output = Path(output).resolve()
    if output.is_relative_to(capture.directory):
        raise ValueError('Output must not be inside the source recording')
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'comparison.csv', [result], tuple(result))
    write_csv(output / 'selection.csv', audit,
              ('record_index', 'time', 'ntx', 'nrx', 'rate_flags', 'status', 'value', 'shape_changed'))
    settling = {s['segment']: s['warmup_until_s'] for s in segment_summary(points, processing)}
    write_csv(output / 'waveform.csv',
              ({**asdict(p), 'in_gap': inside_gap(p.time, gaps),
                'settled': p.time >= settling[p.segment] - 1e-9,
                'display_filtered': None if inside_gap(p.time, gaps) else p.filtered} for p in points),
              ('time', 'available_at', 'raw', 'filtered', 'segment', 'in_gap', 'settled', 'display_filtered'))
    write_csv(output / 'gaps.csv', gaps, ('start_s', 'end_s', 'duration_s', 'available_at_s', 'action'))
    write_csv(output / 'markers.csv',
              ({**e, 'relative_seconds': (e['host_monotonic_ns'] - capture.t0) / 1e9,
                'in_comparison_window': capture.t0 + start_ns <= e['host_monotonic_ns'] < capture.t0 + end_ns}
               for e in capture.events), (*EVENT_FIELDS, 'relative_seconds', 'in_comparison_window'))
    save_figure(draw_comparison(observations, points, capture, result, config, shape_policy, audit,
                               gap_policy, gaps), output, 'comparison')
    diagnostics, diagnostic_html = export_diagnostics(output, points, result, processing)
    columns = [('Shape policy', shape_policy), ('Gap policy', gap_policy),
               ('Window (s)', f"[{result['start_s']:g}, {result['end_s']:g})"),
               ('Accepted CSI rate candidate (/min)', result['candidate_bpm']),
               ('FFT frequency candidate before quality checks (/min)', result['spectral_peak_bpm']),
               ('ACF frequency candidate before quality checks (/min)', result['autocorrelation_bpm']),
               ('Received button count', result['received_button_count']),
               ('Manual reference (/min)', result['manual_reference_bpm']),
               ('Absolute difference (/min)', result['absolute_error_bpm']),
               ('CSI status', result['status']), ('Reason', result['reason']),
               ('Manual reference status', result['reference_status']),
               ('Reference issues', result['reference_issues'] or 'none'),
               ('Retained / indexed packets', f"{len(kept)} / {len(window_audit)}"),
               ('Supported grid incl. interpolation', f"{result['supported_grid_fraction']:.1%}")]
    if gap_policy == 'preserve-short':
        columns += [('Grid inside bridged gaps (not displayed)', f"{result['bridged_grid_fraction']:.2%}"),
                    ('Bridged intervals overlapping the minute', result['bridged_intervals'])]
    rows = ''.join(f'<tr><th>{html.escape(label)}</th><td>{html.escape("Unavailable" if value is None else str(value))}</td></tr>'
                   for label, value in columns)
    selection_text = ('all reported rates within 3x3' if shape_policy == '3x3'
                      else 'all reported rates; stream 1 across 1x3, 2x3 and 3x3')
    gap_text = ('Gaps over 0.2 s reset processing and hide 10 s of filter warm-up.' if gap_policy == 'reset' else
                'Intervals over 0.2 s and at most 0.5 s preserve filter state. The existing causal linear '
                'resampler supplies internal grid values only after the right endpoint arrives. These values '
                'are used for filtering and the FFT/ACF estimate but are hidden in the plot; the actual gap '
                'stays blank. No time is removed. A candidate is withheld if more than 5% of the minute grid '
                'falls inside these bridges. This bound is exploratory, not a validated error guarantee. '
                'Intervals over 0.5 s and invalid records still reset. Initial/reset settling is shown as '
                'dashed amber, not hidden, and remains ineligible for rate estimation.')
    (output / 'index.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width"><title>60 s rate comparison</title>'
        '<style>body{font:16px system-ui;max-width:1200px;margin:30px auto;padding:0 20px}'
        'p{line-height:1.6}img{width:100%}td,th{padding:7px 14px;border-bottom:1px solid #ddd;text-align:left}</style>'
        f'<h1>60 s Log Ratio rate comparison</h1><p>One predefined window; {selection_text}. '
        'CSI periodicity is estimated independently, then compared with the received manual button count. '
        'This is an average rate estimate, not individual breath detection or an accuracy percentage.</p>'
        '<table>' + rows + '</table><h2>1. Time-domain Log Ratio waveform</h2>'
        '<p>Each packet contributes ln(|H_RX1|/|H_RX2|) for the selected stream, RX indices and subcarrier group. '
        'The filtered curve is this dimensionless feature after median processing, smoothing, resampling and '
        'a causal 0.1-0.5 Hz bandpass. It is not chest displacement or measured air volume. Its sign does not '
        'label inhalation versus exhalation; the causal filter introduces delay.</p>'
        '<img src="comparison.png" alt="Log ratio and manual events on one host timeline">'
        '<p><a href="comparison.pdf">PDF</a> | <a href="comparison.svg">SVG</a> | '
        '<a href="comparison.csv">Comparison CSV</a> | <a href="selection.csv">Packet audit</a> | '
        '<a href="waveform.csv">Waveform</a> | <a href="markers.csv">Manual events</a> | '
        '<a href="gaps.csv">Gap intervals</a> | '
        '<a href="rate-report.json">Provenance and diagnostics</a></p>'+diagnostic_html+
        '<p>Shapes outside the selected policy are missing observations. In first-stream mode, stream-count '
        'changes alone do not reset the filter. No mapping correction or per-configuration normalization is applied. '
        'Changing the physical mapping can introduce jumps or spurious periodicity. '
        f'{gap_text} Filter resets, incomplete coverage or unfinished warm-up prevent a full-minute estimate. '
        'Grid coverage includes interpolation and is not packet reception rate.</p>'
        '<p>The unchanged causal filter uses a 10 s warm-up and 0.1-0.5 Hz band. A 60 s FFT '
        'has 1 /min bin spacing; this is not an accuracy guarantee. Autocorrelation and existing quality gates '
        'can withhold the candidate. No automatic channel/rate search or marker-based alignment is used.</p>'
        f'<p>{LIMITATION} Mixed-rate pooling does not establish physical channel equivalence. '
        'A single-window difference is not a population MAE or a physiological validation.</p></html>', encoding='utf-8')
    capture.verify()
    if implementation_hashes() != implementation:
        raise ValueError('Implementation changed during export; use a new output directory')
    report = {'schema_version': 1, 'complete': True, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'session_id': capture.session_id, 'source_directory': str(capture.directory),
              'source_files': capture.files, 'receipt_sha256': capture.receipt_hash, 'timing': capture.timing,
              'feature': asdict(config), 'processing': asdict(processing), 'comparison': result,
              'estimator_diagnostics': diagnostics,
              'policy': {'name': ('exploratory-3x3-all-rates-minute' if shape_policy == '3x3'
                                   else 'exploratory-first-stream-all-rates-minute'),
                         'shape_policy': shape_policy,
                         'gap_policy': gap_policy,
                         'gap_display_threshold_s': PROCESSING.max_gap_seconds,
                         'max_bridged_grid_fraction': MAX_BRIDGED_FRACTION if gap_policy == 'preserve-short' else 0,
                         'bridge_used_for_estimation': gap_policy == 'preserve-short',
                         'shape': [3, 3] if shape_policy == '3x3' else None,
                         'accepted_shapes': [[3, 3]] if shape_policy == '3x3' else [[1, 3], [2, 3], [3, 3]],
                         'mapping_correction': False, 'automatic_stream_selection': False,
                         'rate_flags': 'all_reported', 'window': '[start_s, end_s)',
                         'marker_fitting': False, 'physical_mapping_stability_verified': False},
              'window_selection_counts': dict(Counter(a['status'] for a in window_audit)),
              'window_retained_rate_counts': dict(Counter(a['rate_flags'] for a in kept)),
              'window_retained_shape_counts': dict(Counter(f"{a['ntx']}x{a['nrx']}" for a in kept)),
              'window_shape_changes': [a['time'] for a in window_audit if a['shape_changed']],
              'segments': segment_summary(points, processing), 'gaps': gaps, 'marker_status': capture.status,
              'raw_data_modified': False, 'radio_configuration_modified': False, 'physiological_validation': False,
              'limitation': LIMITATION, 'implementation_hashes': implementation,
              'artifacts': [{'path': p.name, 'sha256': sha256(p)} for p in sorted(output.iterdir()) if p.is_file()]}
    write_json(output / 'rate-report.json', report)
    return report
