"""Independent VitalCSI adaptation (10.3390/s26010225), equations (3)-(14).

Completed trailing windows only; no manual events or activity labels are inputs.
Unspecified conventions, BSSF settings, and Kalman choices are documented in
docs/VITALCSI_REPRODUCTION.md. This is not the authors' calibrated implementation.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class VitalCSIConfig:
    sample_hz: float = 20.
    window_seconds: float = 30.
    hop_seconds: float = 1.
    rx: int = 1
    top_k: int = 5
    variance_increment: float = .01
    low_bpm: float = 5.
    high_bpm: float = 35.
    filter_order_each: int = 4
    bssf_box_seconds: float = .4
    bssf_adaptation: float = .1
    agreement_bpm: float = 4.
    process_variance_per_second: float = 1.
    measurement_variance: float = 4.
    initial_variance: float = 4.
    residual_floor_bpm: float = .5
    loading_similarity: float = .8
    max_gap_seconds: float = .5
    audited_gap_seconds: float = .2
    max_bridged_fraction: float = .05

    def __post_init__(self):
        if any(not math.isfinite(v) for v in vars(self).values()):
            raise ValueError('VitalCSI parameters must be finite')
        for name in ('rx', 'top_k', 'filter_order_each'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name+' must be an integer')
        if (self.sample_hz <= 0 or self.window_seconds <= 0 or self.hop_seconds <= 0
                or not 1 <= self.rx <= 3 or not 1 <= self.top_k <= 30
                or not 0 < self.variance_increment < 1
                or not 0 < self.low_bpm < self.high_bpm < 30*self.sample_hz
                or not 1 <= self.filter_order_each <= 8
                or not 0 < self.bssf_box_seconds < 60/self.high_bpm
                or not 0 < self.bssf_adaptation < 1 or self.agreement_bpm < 0
                or self.process_variance_per_second < 0 or self.measurement_variance <= 0
                or self.initial_variance <= 0 or self.residual_floor_bpm <= 0
                or not 0 <= self.loading_similarity <= 1
                or not 0 < self.audited_gap_seconds <= self.max_gap_seconds
                or not 0 <= self.max_bridged_fraction <= 1):
            raise ValueError('Invalid VitalCSI settings')
        for seconds in (self.window_seconds, self.hop_seconds, self.bssf_box_seconds):
            n = seconds*self.sample_hz
            if n < 1 or not math.isclose(n, round(n)):
                raise ValueError('Window, hop and box must use integer sample counts')
        if self.window_seconds*self.sample_hz <= 12*self.filter_order_each+3:
            raise ValueError('Window too short for the filter padding')


def amplitudes(capture, cfg=VitalCSIConfig()):
    """Stream 1 / one fixed RX / all 30 groups, raw magnitudes as in the paper.

    No per-packet normalization or ratio. Stream/physical-antenna equivalence is
    not assumed; shape and rate flags remain in the audit. Missing is never zero.
    """
    times, values, audit = [], [], []
    previous = None
    for row, packet in zip(capture.index, capture.packets):
        ns = row['host_monotonic_ns']
        if not capture.inside(ns):
            continue
        if previous is not None and ns < previous:
            raise ValueError('Backward host timestamp')
        a, status = np.full(30, np.nan), 'kept'
        if ns == previous:
            status = 'duplicate_timestamp'
        elif packet is None:
            status = 'non_csi_record'
        elif packet['ntx'] not in (1, 2, 3) or packet['nrx'] < cfg.rx:
            status = 'unsupported_shape'
        else:
            x = np.abs(np.asarray(packet['csi'][0][cfg.rx-1], complex))
            valid = np.isfinite(x) & (x > 0)
            a[valid] = x[valid]
            if not valid.all():
                status = 'invalid_groups'
        t = (ns-capture.t0)/1e9
        times.append(t)
        values.append(a)
        audit.append({'record_index': row['record_index'], 'time_s': t, 'status': status,
                      'ntx': packet['ntx'] if packet else None, 'nrx': packet['nrx'] if packet else None,
                      'rate_flags': packet['rate'] if packet else None, 'valid_groups': int(np.isfinite(a).sum())})
        previous = ns
    return np.asarray(times), np.asarray(values).reshape(-1, 30), audit


def resample_window(times, values, start, cfg):
    """Bounded interpolation on real time; results wait for needed endpoints."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    n = round(cfg.sample_hz*cfg.window_seconds)
    grid = start+np.arange(n)/cfg.sample_hz
    missing = np.full((n, 30), np.nan)
    meta = {'start_s': float(start), 'end_s': float(start+cfg.window_seconds),
            'available_at_s': None, 'ready': False, 'reason': '',
            'valid_groups': 0, 'bridged_grid_fraction': 0., 'max_observed_gap_s': None}
    if (not math.isfinite(start) or start < 0 or values.shape != (len(times), 30)
            or not np.isfinite(times).all() or np.any(np.diff(times) < 0)):
        raise ValueError('Invalid amplitude samples or timestamps')
    right = np.searchsorted(times, grid-1e-9)
    if not len(times) or np.any(right >= len(times)):
        meta['reason'] = 'missing_right_endpoint'
        return meta, grid, missing
    exact = np.abs(times[right]-grid) <= 1e-9
    left = np.where(exact, right, right-1)
    if np.any(left < 0):
        meta['reason'] = 'missing_left_endpoint'
        return meta, grid, missing
    support = slice(int(left.min()), int(right.max())+1)
    intervals = np.diff(times[support])
    dt = times[right]-times[left]
    meta['available_at_s'] = max(meta['end_s'], float(times[right].max()))
    meta['max_observed_gap_s'] = float(intervals.max()) if len(intervals) else 0.
    if np.any(intervals <= 0):
        meta['reason'] = 'duplicate_timestamp'
    elif np.any(dt > cfg.max_gap_seconds+1e-9):
        meta['reason'] = 'long_gap'
    else:
        meta['bridged_grid_fraction'] = float(np.mean(dt > cfg.audited_gap_seconds+1e-9))
        if meta['bridged_grid_fraction'] > cfg.max_bridged_fraction:
            meta['reason'] = 'excessive_gap_interpolation'
    if meta['reason']:
        return meta, grid, missing
    valid = np.isfinite(values[support]).all(axis=0)
    meta['valid_groups'] = int(valid.sum())
    if not valid.any():
        meta['reason'] = 'no_complete_group'
        return meta, grid, missing
    alpha = np.divide(grid-times[left], dt, out=np.zeros(n), where=dt > 0)
    missing[:, valid] = values[left][:, valid]+alpha[:, None]*(values[right][:, valid]-values[left][:, valid])
    meta.update(ready=True, reason='bounded_timestamp_resampling')
    return meta, grid, missing


def power_spectrum(x, fs):
    """Mean-centered Hann periodogram, one-sided energy, no zero padding."""
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[:, None]
    p = abs(np.fft.rfft((x-x.mean(axis=0))*np.hanning(len(x))[:, None], axis=0))**2
    weights = np.full(len(p), 2.)
    weights[0] = 1.
    if len(x) % 2 == 0:
        weights[-1] = 1.
    return np.fft.rfftfreq(len(x), 1/fs), p*weights[:, None]


def spectral_purity(hz, power):
    """Eqs. (11)-(12); frequency units and spectral scale cancel in the ratio."""
    p = np.asarray(power, float)
    m0 = np.sum(p, axis=0)
    m2 = np.sum(hz[:, None]**2*p, axis=0)
    m4 = np.sum(hz[:, None]**4*p, axis=0)
    den = m0*m4
    return np.clip(np.divide(m2*m2, den, out=np.zeros_like(m0), where=den > 1e-30), 0., 1.)


def select_components(values, cfg):
    """Centered covariance PCA, prose-based 1% elbow, and Eqs. (4)-(6).

    The paper's Eq. (3) contradicts its cumulative-variance example. Retain the
    variance-ordered prefix before the first component adding <1%, at least one.
    Keep up to five positive-CQI PCs, never pad with a zero-quality component.
    """
    valid = np.flatnonzero(np.isfinite(values).all(axis=0))
    x = values[:, valid]
    centered = x-x.mean(axis=0)
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2/(len(x)-1)
    total = float(variance.sum())
    fractions = variance/total if total > 1e-20 else np.zeros_like(variance)
    # Stable sign by largest loading, not by reference events or waveform phase.
    signs = np.sign(vt[np.arange(len(vt)), np.argmax(abs(vt), axis=1)])
    signs[signs == 0] = 1
    vt *= signs[:, None]
    pcs = u*singular*signs
    small = np.flatnonzero(fractions < cfg.variance_increment)
    retained = max(1, int(small[0])) if len(small) else len(fractions)
    variance_gate = (np.arange(len(fractions)) < retained) & (fractions > 1e-12)
    hz, power = power_spectrum(pcs, cfg.sample_hz)
    band = (hz*60 >= cfg.low_bpm-1e-9) & (hz*60 <= cfg.high_bpm+1e-9)
    total_power = power[1:].sum(axis=0)
    snr = np.divide(power[band].sum(axis=0), total_power, out=np.zeros_like(total_power), where=total_power > 1e-20)
    peak_hz = hz[1+np.argmax(power[1:], axis=0)]
    peak_gate = peak_hz*60 >= cfg.low_bpm-1e-9
    cqi = variance_gate*snr*peak_gate
    selected = np.argsort(-cqi, kind='stable')[:cfg.top_k]
    selected = selected[cqi[selected] > 0]
    loadings = np.zeros((len(vt), 30))
    loadings[:, valid] = vt
    return {'group_indices': valid, 'centered_amplitudes': centered, 'pcs': pcs,
            'loadings': loadings, 'variance': variance, 'variance_fraction': fractions,
            'variance_gate': variance_gate, 'variance_prefix_count': retained,
            'snr': snr, 'peak_hz': peak_hz, 'peak_gate': peak_gate, 'cqi': cqi,
            'selected_indices': selected, 'frequencies_hz': hz, 'raw_power': power}


def filter_components(pcs, cfg):
    """Eight poles in one pass: order-4 high-pass then order-4 low-pass.

    Forward/backward evaluation uses only this completed 30 s window. It is
    retrospective within the window, not a sample-level causal waveform.
    """
    high = signal.butter(cfg.filter_order_each, cfg.low_bpm/60, 'highpass', fs=cfg.sample_hz, output='sos')
    low = signal.butter(cfg.filter_order_each, cfg.high_bpm/60, 'lowpass', fs=cfg.sample_hz, output='sos')
    return signal.sosfiltfilt(low, signal.sosfiltfilt(high, pcs, axis=0), axis=0)


def breath_onsets(x, cfg):
    """BSSF adaptation: positive-slope box sums, pulse boxes, adaptive threshold.

    Box construction and 10% update follow Villarroel et al. (2020), the same
    author's published BSSF description. Respiratory box/refractory durations
    and per-completed-window initialization are explicit adaptation choices.
    """
    x = np.asarray(x, float)
    m = round(cfg.bssf_box_seconds*cfg.sample_hz)
    positive = np.maximum(np.diff(x, prepend=x[0]), 0.)
    ssf = np.convolve(positive, np.ones(m), mode='full')[:len(x)]
    tol = max(float(ssf.max())*1e-10, 1e-14)
    active = ssf > tol
    edges = np.diff(np.r_[False, active, False].astype(int))
    starts, ends = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    boxes = [(int(a), int(b), float(ssf[a:b].max())) for a, b in zip(starts, ends)
             if a > 1 and b < len(x)]
    bssf = np.zeros_like(x)
    threshold_trace = np.full(len(x), np.nan)
    accepted, decisions = [], []
    threshold = float(np.median([v for _, _, v in boxes])) if boxes else 0.
    refractory = math.ceil(cfg.sample_hz*60/cfg.high_bpm)
    for left, right, height in boxes:
        bssf[left:right] = height
        threshold_trace[left:right] = threshold
        above = height >= threshold-tol
        separated = not accepted or left-accepted[-1] >= refractory
        keep = above and separated
        decisions.append({'sample': left, 'height': height, 'threshold': threshold,
                          'above_threshold': bool(above), 'refractory_ok': bool(separated), 'accepted': bool(keep)})
        if keep:
            accepted.append(left)
        threshold = (threshold+cfg.bssf_adaptation*(height-threshold) if above
                     else threshold*(1-cfg.bssf_adaptation))
    onsets = np.asarray(accepted, int)
    rate = float(60*cfg.sample_hz*(len(onsets)-1)/(onsets[-1]-onsets[0])) if len(onsets) >= 2 else None
    if rate is not None and not cfg.low_bpm <= rate <= cfg.high_bpm:
        rate = None
    return rate, onsets, ssf, bssf, threshold_trace, decisions


def analyze_frame(times, values, start, cfg=VitalCSIConfig()):
    meta, grid, sampled = resample_window(times, values, start, cfg)
    meta.update(selected_components=0, usable_components=0, raw_component_mean_bpm=None,
                fused_bpm=None, components=[])
    if not meta['ready']:
        return meta, None
    arrays = select_components(sampled, cfg)
    arrays.update(times=grid, amplitudes=sampled)
    selected = arrays['selected_indices']
    meta['selected_components'] = len(selected)
    if not len(selected):
        meta['reason'] = 'no_positive_cqi_component'
        return meta, arrays
    filtered = filter_components(arrays['pcs'][:, selected], cfg)
    hz, power = power_spectrum(filtered, cfg.sample_hz)
    purity = spectral_purity(hz, power)
    arrays.update(filtered=filtered, filtered_power=power,
                  ssf=np.empty_like(filtered), bssf=np.empty_like(filtered), thresholds=np.empty_like(filtered))
    rates = []
    for column, index in enumerate(selected):
        peak = int(1+np.argmax(power[1:, column]))
        fft = float(hz[peak]*60) if power[peak, column] > 1e-20 else None
        if fft is not None and not cfg.low_bpm <= fft <= cfg.high_bpm:
            fft = None
        breath, onsets, ssf, bssf, thresholds, decisions = breath_onsets(filtered[:, column], cfg)
        arrays['ssf'][:, column], arrays['bssf'][:, column], arrays['thresholds'][:, column] = ssf, bssf, thresholds
        combined = (fft+breath)/2 if fft is not None and breath is not None else None
        agree = combined is not None and abs(fft-breath) <= cfg.agreement_bpm
        sqi = float(arrays['snr'][index]*purity[column]) if agree else 0.
        if combined is not None:
            rates.append(combined)
        meta['components'].append({'pc_index': int(index), 'variance_fraction': float(arrays['variance_fraction'][index]),
                                  'cqi': float(arrays['cqi'][index]), 'snr': float(arrays['snr'][index]),
                                  'spi': float(purity[column]), 'agreement': bool(agree), 'sqi': sqi,
                                  'fft_bpm': fft, 'breath_bpm': breath, 'combined_bpm': combined,
                                  'onsets_s': grid[onsets].tolist(), 'bssf_decisions': decisions})
    meta['raw_component_mean_bpm'] = float(np.mean(rates)) if rates else None
    meta['usable_components'] = sum(c['sqi'] > 0 for c in meta['components'])
    meta['reason'] = 'components_ready' if meta['usable_components'] else 'no_positive_sqi_component'
    return meta, arrays


class KalmanFusion:
    """Scalar random walks followed by Eqs. (13)-(14) inverse-variance fusion.

    Source identity is the variance-ordered PC, with a loading similarity guard.
    No output is carried forward when current measurements are unavailable.
    """
    def __init__(self, cfg=VitalCSIConfig()):
        self.cfg, self.states = cfg, {}

    def update(self, frame, arrays):
        cfg = self.cfg
        live, details, weights, estimates = {}, [], [], []
        for component in frame['components']:
            i, z, sqi = component['pc_index'], component['combined_bpm'], component['sqi']
            if z is None or sqi <= 0:
                continue
            loading = arrays['loadings'][i]
            old = self.states.get(i)
            similarity = abs(float(loading @ old['loading'])) if old is not None else None
            same_groups = old is not None and np.array_equal(arrays['group_indices'], old['groups'])
            reuse = old is not None and same_groups and similarity >= cfg.loading_similarity
            if reuse:
                dt = frame['end_s']-old['time']
                predicted, variance = old['value'], old['variance']+dt*cfg.process_variance_per_second
                residual = z-predicted
                gain = variance/(variance+cfg.measurement_variance)
                estimate, posterior = predicted+gain*residual, (1-gain)*variance
            else:
                predicted, residual, gain = z, 0., None
                estimate, posterior = z, cfg.initial_variance
            effective = max(abs(residual), cfg.residual_floor_bpm)
            variance_for_fusion = (effective/sqi)**2
            weights.append(1/variance_for_fusion)
            estimates.append(estimate)
            details.append({'pc_index': i, 'measurement_bpm': z, 'sqi': sqi,
                            'prediction_bpm': predicted, 'innovation_bpm': residual, 'gain': gain,
                            'posterior_variance': posterior, 'estimate_bpm': estimate,
                            'loading_similarity': similarity, 'reinitialized': not reuse,
                            'fusion_variance': variance_for_fusion})
            live[i] = {'value': estimate, 'variance': posterior, 'time': frame['end_s'],
                       'loading': loading.copy(), 'groups': arrays['group_indices'].copy()}
        self.states = live
        if not weights:
            return None, details
        weights = np.asarray(weights)/sum(weights)
        for d, w in zip(details, weights):
            d['fusion_weight'] = float(w)
        return float(weights @ estimates), details


def analyze_interval(times, values, start, end, cfg=VitalCSIConfig()):
    if not all(math.isfinite(v) for v in (start, end)) or start < 0 or end-start < cfg.window_seconds:
        raise ValueError('Invalid VitalCSI interval')
    count = int(math.floor((end-start-cfg.window_seconds+1e-9)/cfg.hop_seconds))+1
    frames, data, fusion = [], [], KalmanFusion(cfg)
    for i in range(count):
        frame, arrays = analyze_frame(times, values, start+i*cfg.hop_seconds, cfg)
        frame['fused_bpm'], frame['fusion'] = fusion.update(frame, arrays)
        if frame['fused_bpm'] is not None:
            frame['reason'] = 'quality_weighted_kalman_candidate'
        frames.append(frame)
        data.append(arrays)
    rates = [f['fused_bpm'] for f in frames if f['fused_bpm'] is not None]
    raw = [f['raw_component_mean_bpm'] for f in frames if f['raw_component_mean_bpm'] is not None]
    summary = {'total_frames': count, 'available_frames': len(rates),
               'fused_mean_bpm': float(np.mean(rates)) if len(rates) == count else None,
               'available_only_mean_bpm': float(np.mean(rates)) if rates else None,
               'raw_component_frames': len(raw),
               'raw_component_mean_bpm': float(np.mean(raw)) if len(raw) == count else None,
               'aggregation': 'mean of all trailing frame estimates; every frame required',
               'manual_events_used_in_estimation': False}
    return summary, frames, data
