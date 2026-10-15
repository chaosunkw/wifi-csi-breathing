"""Independent WiResP adaptation, DOI 10.1109/JSEN.2024.3399110.

Equations (5)-(10), sections V-A..E. Unspecified numeric conventions and the
experimental pulse checker are documented in docs/WIRESP_REPRODUCTION.md.
Only completed trailing windows enter a result. No labels or markers are inputs.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy import ndimage, signal


@dataclass(frozen=True)
class WiResPConfig:
    sample_hz: float = 20.
    window_seconds: float = 15.
    enhancement_seconds: float = 30.
    hop_seconds: float = 1.
    nfft: int = 2048
    median_samples: int = 5
    loess_samples: int = 11
    top_k: int = 10
    low_bpm: float = 10.
    high_bpm: float = 30.
    max_gap_seconds: float = .5
    audited_gap_seconds: float = .2
    max_bridged_fraction: float = .05
    min_full_pulses: int = 2
    max_period_cv: float = .25
    min_duty: float = .15
    max_duty: float = .85
    max_template_mismatch: float = .25
    motion_threshold: float | None = None

    def __post_init__(self):
        if any(not math.isfinite(v) for v in vars(self).values() if v is not None):
            raise ValueError('WiResP parameters must be finite')
        for name in ('nfft', 'median_samples', 'loess_samples', 'top_k', 'min_full_pulses'):
            if type(getattr(self, name)) is not int:
                raise ValueError(name+' must be an integer')
        if (self.sample_hz <= 0 or self.window_seconds <= 0 or self.hop_seconds <= 0
                or self.enhancement_seconds < self.hop_seconds
                or not 0 < self.low_bpm < self.high_bpm < 30*self.sample_hz
                or not 0 < self.audited_gap_seconds <= self.max_gap_seconds
                or not 0 <= self.max_bridged_fraction <= 1
                or not 1 <= self.top_k <= 90 or self.min_full_pulses < 2
                or not 0 <= self.max_period_cv <= 1
                or not 0 < self.min_duty < self.max_duty < 1
                or not 0 <= self.max_template_mismatch < .5
                or self.motion_threshold is not None and not -1 <= self.motion_threshold <= 1):
            raise ValueError('Invalid WiResP settings')
        n = self.window_seconds*self.sample_hz
        if (not math.isclose(n, round(n)) or self.nfft < n
                or not math.isclose(self.enhancement_seconds/self.hop_seconds,
                                    round(self.enhancement_seconds/self.hop_seconds))
                or not math.isclose(self.hop_seconds*self.sample_hz, round(self.hop_seconds*self.sample_hz))
                or any(v % 2 != 1 or not 3 <= v <= n for v in (self.median_samples, self.loess_samples))
                or self.window_seconds < 2*60/self.low_bpm):
            raise ValueError('Invalid WiResP sample/window sizes')


def normalized_amplitudes(capture):
    """First effective stream, RX1..3, all 30 groups; per-packet/RX RMS norm.

    Nine/ six / three physical-path interpretations are deliberately not assumed.
    A zero/invalid group invalidates its RX normalization rather than becoming 0.
    """
    times, values, audit = [], [], []
    previous = None
    for row, packet in zip(capture.index, capture.packets):
        ns = row['host_monotonic_ns']
        if not capture.inside(ns):
            continue
        if previous is not None and ns < previous:
            raise ValueError('Backward host timestamp')
        t = (ns-capture.t0)/1e9
        amp = np.full((3, 30), np.nan)
        status = 'kept'
        if ns == previous:
            status = 'duplicate_timestamp'
        elif packet is None:
            status = 'non_csi_record'
        elif packet['nrx'] != 3 or packet['ntx'] not in (1, 2, 3):
            status = 'unsupported_shape'
        else:
            a = np.abs(np.asarray(packet['csi'][0], complex))
            for rx in range(3):
                if np.isfinite(a[rx]).all() and np.all(a[rx] > 0):
                    # Scale first to avoid overflow; common packet/RX gain cancels.
                    scaled = a[rx]/a[rx].max()
                    amp[rx] = scaled/np.sqrt(np.mean(scaled**2))
            if not np.isfinite(amp).all():
                status = 'invalid_channels'
        times.append(t)
        values.append(amp.reshape(90))
        audit.append({'record_index': row['record_index'], 'time_s': t, 'status': status,
                      'ntx': packet['ntx'] if packet else None, 'nrx': packet['nrx'] if packet else None,
                      'rate_flags': packet['rate'] if packet else None,
                      'valid_channels': int(np.isfinite(amp).sum())})
        previous = ns
    return np.asarray(times), np.asarray(values).reshape(-1, 90), audit


def resample_window(times, values, start, cfg):
    """Keep real time, bound interpolation, and expose right-endpoint latency."""
    times, values = np.asarray(times, float), np.asarray(values, float)
    n = round(cfg.sample_hz*cfg.window_seconds)
    grid = start+np.arange(n)/cfg.sample_hz
    missing = np.full((n, 90), np.nan)
    meta = {'start_s': float(start), 'end_s': float(start+cfg.window_seconds),
            'available_at_s': None, 'reason': '', 'ready': False,
            'valid_channels': 0, 'bridged_grid_fraction': 0., 'max_observed_gap_s': None}
    if (not math.isfinite(start) or start < 0 or values.shape != (len(times), 90)
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
    dt = times[right]-times[left]
    meta['available_at_s'] = max(meta['end_s'], float(times[right].max()))
    meta['max_observed_gap_s'] = float(np.diff(times[support]).max())
    if np.any(np.diff(times[support]) <= 0):
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
    meta['valid_channels'] = int(valid.sum())
    if not valid.any():
        meta['reason'] = 'no_complete_channel'
        return meta, grid, missing
    alpha = np.divide(grid-times[left], dt, out=np.zeros(n), where=dt > 0)
    missing[:, valid] = values[left][:, valid]+alpha[:, None]*(values[right][:, valid]-values[left][:, valid])
    meta.update(ready=True, reason='bounded_timestamp_resampling')
    return meta, grid, missing


def covariance_acf(values):
    """Eq. (5) convention: global mean, overlap-count covariance / variance.

    Unlike the existing FarSense finite-sum convention, divide lag k by N-k.
    High-lag values can exceed one. These are not correlation probabilities.
    """
    x = np.asarray(values, float)
    if x.ndim == 1:
        x = x[:, None]
    x = x-x.mean(axis=0)
    n = len(x)
    fft = np.fft.rfft(x, 2*n, axis=0)
    sums = np.fft.irfft(fft*fft.conjugate(), 2*n, axis=0)[:n].real
    denominator = sums[0][None, :]*(1-np.arange(n)[:, None]/n)
    return np.divide(sums, denominator, out=np.zeros_like(sums), where=denominator > 1e-20)


def breathing_noise_ratio(values, cfg):
    """Section V-C: in-band / out-of-band positive-frequency energy.

    Explicit conventions: global mean removal, Hann taper, nfft padding,
    DC excluded, 10-30 /min. Denominator floor is numeric, not a quality gate.
    """
    x = np.asarray(values, float)
    power = np.abs(np.fft.rfft((x-x.mean(axis=0))*np.hanning(len(x))[:, None], cfg.nfft, axis=0))**2
    hz = np.fft.rfftfreq(cfg.nfft, 1/cfg.sample_hz)
    weights = np.full(len(hz), 2.)
    weights[0] = 0
    if cfg.nfft % 2 == 0:
        weights[-1] = 1
    power *= weights[:, None]
    band = (hz*60 >= cfg.low_bpm) & (hz*60 <= cfg.high_bpm)
    inside, outside = power[band].sum(axis=0), power[~band].sum(axis=0)
    floor = np.maximum((inside+outside)*1e-12, 1e-20)
    return inside/np.maximum(outside, floor), power, hz


@lru_cache(maxsize=16)
def loess_matrix(n, span):
    """Local quadratic tricube weighted least squares, no robust reweighting."""
    matrix = np.zeros((n, n))
    for i in range(n):
        first = min(max(i-span//2, 0), n-span)
        indices = np.arange(first, first+span)
        x = indices-i
        weights = (1-(np.abs(x)/max(abs(x)))**3)**3
        design = np.column_stack((np.ones(span), x, x*x))
        root = np.sqrt(weights)
        matrix[i, indices] = np.linalg.pinv(design*root[:, None])[0]*root
    matrix.setflags(write=False)
    return matrix


def analyze_frame(values, cfg):
    """Median amplitudes, BNR-ranked channels, weighted ACF, then LOESS."""
    valid = np.isfinite(values).all(axis=0)
    n = len(values)
    if not valid.any():
        return {'reason': 'no_complete_channel', 'motion_statistic': None}, None
    indices = np.flatnonzero(valid)
    amp = ndimage.median_filter(values[:, valid], size=(cfg.median_samples, 1), mode='nearest')
    acfs = covariance_acf(amp)
    bnr, power, hz = breathing_noise_ratio(amp, cfg)
    order = np.argsort(-bnr, kind='stable')[:cfg.top_k]
    selected = order[bnr[order] > 1e-12]
    if not len(selected):
        return {'reason': 'insufficient_variation', 'motion_statistic': 0.}, None
    combined = acfs[:, selected] @ bnr[selected]
    smooth = loess_matrix(n, cfg.loess_samples) @ combined
    # Eq. (6) explicitly mentions CSI power. Measure before median smoothing.
    motion = float(covariance_acf(values[:, valid]**2)[1].mean())
    motion_pass = None if cfg.motion_threshold is None else motion < cfg.motion_threshold
    return {'reason': 'acf_ready', 'motion_statistic': motion, 'motion_pass': motion_pass,
            'selected_channels': (indices[selected]+1).tolist(),
            'max_bnr': float(bnr.max())}, {
                'normalized_amplitudes': values, 'median_amplitudes': amp,
                'channel_indices': indices, 'selected_indices': selected,
                'acf': acfs, 'bnr': bnr, 'power': power, 'frequency_hz': hz,
                'combined_acf': combined, 'smoothed_acf': smooth}


def equalize_ecdf(x):
    """Eq. (7): empirical CDF with <=, including exact tie handling."""
    x = np.asarray(x, float)
    if not x.size or not np.isfinite(x).all():
        raise ValueError('ECDF requires a finite, nonempty image')
    return np.searchsorted(np.sort(x, axis=None), x, side='right')/x.size


def otsu_threshold(y):
    """Eq. (8): exact empirical between-class variance, no histogram bins."""
    values, counts = np.unique(y, return_counts=True)
    if len(values) < 2:
        return None
    p = counts/counts.sum()
    weight = np.cumsum(p)[:-1]
    moment = np.cumsum(p*values)[:-1]
    variance = (moment-weight*np.dot(p, values))**2/(weight*(1-weight))
    i = int(np.argmax(variance))
    return float((values[i]+values[i+1])/2)


def pulse_check(trend, cfg):
    """Adapted pulse/template rule; numeric thresholds are NOT from the paper."""
    trend = np.asarray(trend, bool)
    edges = np.diff(np.r_[False, trend, False].astype(int))
    left, right = np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
    full = (left > 0) & (right < len(trend))
    centers = (left[full]+right[full]-1)/2/cfg.sample_hz
    widths = (right[full]-left[full])/cfg.sample_hz
    result = {'passed': False, 'full_pulses': len(centers), 'period_s': None,
              'period_cv': None, 'mean_duty': None, 'template_mismatch': None,
              'reason': 'too_few_complete_pulses'}
    template = np.zeros(len(trend), bool)
    if len(centers) < cfg.min_full_pulses:
        return result, template
    intervals = np.diff(centers)
    period = float(np.median(intervals))
    cv = float(np.std(intervals)/np.mean(intervals))
    duty = float(np.mean(widths)/period)
    template = np.cos(2*np.pi*(np.arange(len(trend))/cfg.sample_hz-centers[0])/period) >= 0
    # Compare over the interior supported by complete pulses, not clipped edges.
    support = slice(left[full][0], right[full][-1])
    mismatch = float(np.mean(template[support] != trend[support]))
    passed = (60/cfg.high_bpm <= period <= 60/cfg.low_bpm
              and cv <= cfg.max_period_cv and cfg.min_duty <= duty <= cfg.max_duty
              and mismatch <= cfg.max_template_mismatch)
    result.update(passed=bool(passed), period_s=period, period_cv=cv, mean_duty=duty,
                  template_mismatch=mismatch, reason='adapted_pulse_check_passed' if passed else 'adapted_pulse_check_failed')
    return result, template


def first_peak(y, cfg):
    peaks, _ = signal.find_peaks(y)
    peaks = peaks[(peaks >= math.ceil(60*cfg.sample_hz/cfg.high_bpm))
                  & (peaks <= math.floor(60*cfg.sample_hz/cfg.low_bpm))]
    return int(peaks[0]) if len(peaks) else None


def enhance_history(image, cfg):
    """Trailing image only. Rate is from latest enhanced column, not a template."""
    enhanced = equalize_ecdf(image)
    threshold = otsu_threshold(enhanced)
    binary = enhanced >= threshold if threshold is not None else np.zeros_like(enhanced, bool)
    average = binary.mean(axis=1)
    trend = average >= .5
    check, template = pulse_check(trend, cfg)
    lag = first_peak(enhanced[:, -1], cfg)
    rate = 60*cfg.sample_hz/lag if lag is not None else None
    return {**check, 'threshold': threshold, 'lag_samples': lag, 'candidate_bpm': rate}, {
        'enhanced': enhanced, 'binary': binary, 'binary_average': average,
        'binary_trend': trend, 'template': template}


def analyze_interval(times, amplitudes, start, end, cfg=WiResPConfig()):
    """Sliding frames confined to this interval, causal at frame completion.

    Raw candidates remain visible before continuity history is ready. No
    retrospective gate is propagated backwards onto earlier frame decisions.
    """
    if not math.isfinite(end) or end-start < cfg.window_seconds:
        raise ValueError('Interval must hold at least one WiResP window')
    ends = start+cfg.window_seconds+np.arange(
        math.floor((end-start-cfg.window_seconds)/cfg.hop_seconds+1e-9)+1)*cfg.hop_seconds
    history_count = round(cfg.enhancement_seconds/cfg.hop_seconds)
    frames, arrays = [], []
    for i, stop in enumerate(ends):
        meta, grid, uniform = resample_window(times, amplitudes, stop-cfg.window_seconds, cfg)
        result = {**meta, 'candidate_bpm': None, 'enhanced_candidate_bpm': None,
                  'accepted_bpm': None, 'continuity_pass': None, 'continuity_reason': 'history_not_ready',
                  'motion_pass': None, 'motion_statistic': None, 'selected_channels': []}
        data = None
        if meta['ready']:
            detail, data = analyze_frame(uniform, cfg)
            result.update(detail)
            if data is not None:
                data['times'] = grid
                lag = first_peak(data['smoothed_acf'], cfg)
                result['candidate_bpm'] = float(60*cfg.sample_hz/lag) if lag is not None else None
                result['lag_samples'] = lag
        frames.append(result)
        arrays.append(data)
        if i+1 < history_count:
            continue
        history = arrays[i+1-history_count:i+1]
        if any(a is None for a in history):
            result['continuity_reason'] = 'missing_acf_history'
            continue
        if any(r['motion_pass'] is False for r in frames[i+1-history_count:i+1]):
            result['continuity_reason'] = 'motion_in_history'
            continue
        image = np.column_stack([a['smoothed_acf'] for a in history])
        check, extra = enhance_history(image, cfg)
        result.update(continuity_pass=check['passed'], continuity_reason=check['reason'],
                      enhanced_candidate_bpm=check['candidate_bpm'], continuity=check)
        result['available_at_s'] = max(r['available_at_s'] for r in frames[i+1-history_count:i+1])
        if check['passed'] and result['motion_pass'] is not False:
            result['accepted_bpm'] = check['candidate_bpm']
        data.update(extra)
    candidates = [r['candidate_bpm'] for r in frames if r['candidate_bpm'] is not None]
    accepted = [r['accepted_bpm'] for r in frames if r['accepted_bpm'] is not None]
    summary = {'start_s': start, 'end_s': end, 'total_frames': len(frames),
               'candidate_frames': len(candidates), 'continuity_evaluated_frames': sum(r['continuity_pass'] is not None for r in frames),
               'accepted_frames': len(accepted),
               'candidate_mean_bpm': float(np.mean(candidates)) if len(candidates) == len(frames) else None,
               'available_candidates_mean_bpm': float(np.mean(candidates)) if candidates else None,
               'accepted_frames_mean_bpm': float(np.mean(accepted)) if accepted else None,
               'motion_gate_enabled': cfg.motion_threshold is not None,
               'continuity_gate': 'adapted_uncalibrated_pulse_and_template_rule',
               'full_system_reproduction': False}
    return summary, frames, arrays
