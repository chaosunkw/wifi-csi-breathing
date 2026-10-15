"""FarSense core, independently implemented from DOI 10.1145/3351279.

Equations (9)-(11), Sections 5.2 and 6.2-6.4. This is an adapted core,
not the authors' code or a reproduction of their full motion-gated system.
See docs/FARSENSE_REPRODUCTION.md for every convention and adaptation.
No labels, button events, or samples beyond a completed window enter estimation.
"""
from dataclasses import dataclass
import math

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class FarSenseConfig:
    sample_hz: float = 20.0
    window_seconds: float = 12.0
    nfft: int = 2048
    sg_samples: int = 11
    sg_order: int = 3
    angles: int = 100
    low_bpm: float = 10.0
    high_bpm: float = 37.0
    relative_bnr: float = .7
    max_gap_seconds: float = .5
    audited_gap_seconds: float = .2
    max_bridged_fraction: float = .05

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError('FarSense parameters must be finite')
        n = self.sample_hz * self.window_seconds
        if (self.sample_hz <= 0 or self.window_seconds <= 0 or not math.isclose(n, round(n))
                or not 0 < self.low_bpm < self.high_bpm < self.sample_hz * 30
                or not 0 < self.relative_bnr < 1
                or not 0 < self.audited_gap_seconds <= self.max_gap_seconds
                or not 0 <= self.max_bridged_fraction <= 1):
            raise ValueError('Invalid FarSense timing, band or selection settings')
        if any(type(v) is not int for v in (self.nfft, self.sg_samples, self.sg_order, self.angles)):
            raise ValueError('FFT, smoothing and angle counts must be integers')
        if (self.nfft < n or self.sg_samples % 2 != 1 or not 0 <= self.sg_order < self.sg_samples <= n
                or self.angles < 2):
            raise ValueError('Invalid FFT length, Savitzky-Golay window or projection count')


def complex_ratios(capture, rx=1, reference_rx=2, shape_policy='first-stream'):
    """Stream 1, all 30 groups; invalid ratios stay missing, never zero-filled."""
    if rx == reference_rx or not 1 <= rx <= 3 or not 1 <= reference_rx <= 3:
        raise ValueError('Choose distinct receive chains in 1..3')
    if shape_policy not in ('first-stream', '3x3'):
        raise ValueError('Unknown shape policy')
    times, values, audit = [], [], []
    previous = None
    for row, packet in zip(capture.index, capture.packets):
        ns = row['host_monotonic_ns']
        if not capture.inside(ns):
            continue
        t = (ns - capture.t0) / 1e9
        if previous is not None and ns < previous:
            raise ValueError('Backward host timestamp')
        z = np.full(30, complex(np.nan, np.nan))
        status = 'kept'
        if ns == previous:
            status = 'duplicate_timestamp'
        elif packet is None:
            status = 'non_csi_record'
        elif packet['nrx'] != 3 or packet['ntx'] not in (1, 2, 3):
            status = 'unsupported_shape'
        elif shape_policy == '3x3' and packet['ntx'] != 3:
            status = 'other_shape'
        else:
            a = np.asarray(packet['csi'][0][rx-1], complex)
            b = np.asarray(packet['csi'][0][reference_rx-1], complex)
            valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) > 0)
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                z[valid] = a[valid] / b[valid]
            z[~np.isfinite(z)] = complex(np.nan, np.nan)
            if not np.isfinite(z).all():
                status = 'invalid_groups'
        previous = ns
        audit.append({'record_index': row['record_index'], 'time_s': t, 'status': status,
                      'ntx': packet['ntx'] if packet else None,
                      'nrx': packet['nrx'] if packet else None,
                      'rate_flags': packet['rate'] if packet else None,
                      'valid_groups': int(np.isfinite(z).sum())})
        # A deliberately unselected shape is missing; malformed records remain barriers.
        if status != 'other_shape':
            times.append(t)
            values.append(z)
    return np.asarray(times), np.asarray(values, complex).reshape(-1, 30), audit


def uniform_window(times, ratios, start, cfg=FarSenseConfig()):
    """Bounded interpolation on real timestamps, with right-endpoint latency.

    A group with a zero denominator/invalid record anywhere in its support is
    excluded for this window. Long gaps invalidate the window; no compression.
    """
    times, ratios = np.asarray(times, float), np.asarray(ratios, complex)
    if not math.isfinite(start) or start < 0 or ratios.shape != (len(times), 30):
        raise ValueError('Invalid window start or ratio array shape')
    if not np.isfinite(times).all() or np.any(np.diff(times) < 0):
        raise ValueError('Ratio timestamps must be finite and ordered')
    n = round(cfg.sample_hz * cfg.window_seconds)
    grid = start + np.arange(n) / cfg.sample_hz
    result = {'start_s': start, 'end_s': start+cfg.window_seconds, 'status': 'unavailable',
              'reason': '', 'available_at_s': None, 'bridged_grid_samples': 0,
              'bridged_grid_fraction': 0., 'valid_groups': 0, 'max_observed_gap_s': None}
    missing = np.full((n, 30), complex(np.nan, np.nan))
    if not len(times):
        result['reason'] = 'no_packets'
        return result, grid, missing, np.zeros(n, bool)
    right = np.searchsorted(times, grid - 1e-9, side='left')
    if np.any(right == len(times)):
        result['reason'] = 'missing_right_endpoint'
        return result, grid, missing, np.zeros(n, bool)
    exact = np.abs(times[right] - grid) <= 1e-9
    left = np.where(exact, right, right-1)
    if np.any(left < 0):
        result['reason'] = 'missing_left_endpoint'
        return result, grid, missing, np.zeros(n, bool)
    durations = times[right] - times[left]
    support = slice(int(left.min()), int(right.max())+1)
    support_times = times[support]
    result['available_at_s'] = max(result['end_s'], float(times[right].max()))
    result['max_observed_gap_s'] = float(np.max(np.diff(support_times))) if len(support_times) > 1 else 0.
    if np.any(np.diff(support_times) <= 0):
        result['reason'] = 'duplicate_timestamp'
        return result, grid, missing, np.zeros(n, bool)
    bridges = durations > cfg.audited_gap_seconds + 1e-9
    result['bridged_grid_samples'] = int(bridges.sum())
    result['bridged_grid_fraction'] = float(bridges.mean())
    if np.any(durations > cfg.max_gap_seconds + 1e-9):
        result['reason'] = 'long_gap'
        return result, grid, missing, bridges
    if result['bridged_grid_fraction'] > cfg.max_bridged_fraction:
        result['reason'] = 'excessive_gap_interpolation'
        return result, grid, missing, bridges
    valid = np.isfinite(ratios[support]).all(axis=0)
    result['valid_groups'] = int(valid.sum())
    if not valid.any():
        result['reason'] = 'no_complete_subcarrier'
        return result, grid, missing, bridges
    alpha = np.divide(grid - times[left], durations, out=np.zeros(n), where=durations > 0)
    missing[:, valid] = ratios[left][:, valid] + alpha[:, None] * (ratios[right][:, valid] - ratios[left][:, valid])
    result.update(status='ready', reason='bounded_timestamp_resampling')
    return result, grid, missing, bridges


def normalized_acf(y):
    """Paper Eq. (10): global mean, biased finite-length normalized ACF."""
    y = np.asarray(y, float)
    centered = y - np.mean(y)
    energy = np.dot(centered, centered)
    if not np.isfinite(energy) or energy <= 1e-24:
        return np.zeros(len(y))
    return signal.correlate(centered, centered, mode='full', method='fft')[len(y)-1:] / energy


def projection_search(z, cfg):
    """Eq. (9) and Section 5.2.2. No Hann taper or linear detrending.

    Mean removal before BNR is an explicit implementation convention. The
    numerator is one positive FFT bin; denominator is the full two-sided energy.
    Two FFTs suffice by linearity, but all 100 projection scores are evaluated.
    """
    z = np.asarray(z, complex)
    z = z - z.mean()
    angles = np.arange(cfg.angles) * 2 * np.pi / cfg.angles
    c, s = np.cos(angles), np.sin(angles)
    fi, fq = np.fft.rfft(z.real, cfg.nfft), np.fft.rfft(z.imag, cfg.nfft)
    spectra = fi[:, None] * c + fq[:, None] * s
    power = np.abs(spectra) ** 2
    weights = np.full(len(power), 2.)
    weights[0] = 1
    if cfg.nfft % 2 == 0:
        weights[-1] = 1
    total = weights @ power
    frequencies = np.fft.rfftfreq(cfg.nfft, 1/cfg.sample_hz)
    band = (frequencies*60 >= cfg.low_bpm) & (frequencies*60 <= cfg.high_bpm)
    scores = np.divide(power[band].max(axis=0), total, out=np.zeros(cfg.angles), where=total > 1e-24)
    # Opposite axes tie mathematically; take the first numerically equivalent maximum.
    best = int(np.flatnonzero(scores >= scores.max()-1e-14)[0])
    y = z.real*c[best] + z.imag*s[best]
    return y, best, scores, power[:, best], frequencies


def estimate_core(ratios, cfg=FarSenseConfig()):
    """A completed 12 s window only. No claims of motion/presence detection."""
    ratios = np.asarray(ratios, complex)
    if ratios.shape != (round(cfg.sample_hz*cfg.window_seconds), 30):
        raise ValueError('Expected one complete FarSense window with 30 groups')
    n = len(ratios)
    valid = np.isfinite(ratios).all(axis=0)
    patterns = np.full((n, 30), np.nan)
    acfs = np.zeros((n, 30))
    bnrs, angle_indices = np.zeros(30), np.full(30, -1)
    scores = np.zeros((30, cfg.angles))
    spectra = np.zeros((cfg.nfft//2+1, 30))
    for group in np.flatnonzero(valid):
        z = ratios[:, group]
        smooth = (signal.savgol_filter(z.real, cfg.sg_samples, cfg.sg_order, mode='interp')
                  + 1j*signal.savgol_filter(z.imag, cfg.sg_samples, cfg.sg_order, mode='interp'))
        y, best, score, spectrum, _ = projection_search(smooth, cfg)
        patterns[:, group], acfs[:, group] = y, normalized_acf(y)
        angle_indices[group], scores[group], spectra[:, group] = best, score, spectrum
        bnrs[group] = score[best]
    included = valid & (bnrs > cfg.relative_bnr * bnrs.max()) & (bnrs > 0)
    fused = acfs[:, included] @ bnrs[included]  # Eq. (11), deliberately not an unweighted mean.
    peaks, _ = signal.find_peaks(fused)
    # Paper does not specify numeric peak bounds. Reuse its 10-37 /min BNR band.
    peaks = peaks[(peaks >= math.ceil(60*cfg.sample_hz/cfg.high_bpm))
                  & (peaks <= math.floor(60*cfg.sample_hz/cfg.low_bpm)) & (fused[peaks] > 0)]
    lag = int(peaks[0]) if len(peaks) else None
    best_group = int(np.argmax(bnrs)) if included.any() else None
    result = {'status': 'periodic_candidate' if lag is not None else 'unavailable',
              'reason': 'first_positive_local_acf_peak_in_band' if lag is not None else 'no_acf_peak',
              'candidate_bpm': float(60*cfg.sample_hz/lag) if lag is not None else None,
              'lag_samples': lag, 'max_bnr': float(bnrs.max()),
              'selected_groups': (np.flatnonzero(included)+1).tolist(),
              'best_group': best_group+1 if best_group is not None else None,
              'motion_gate_implemented': False, 'presence_detection': False}
    return result, {'patterns': patterns, 'acfs': acfs, 'fused_acf': fused,
                    'bnr': bnrs, 'angle_indices': angle_indices, 'angle_scores': scores,
                    'spectra': spectra, 'included': included,
                    'frequencies': np.fft.rfftfreq(cfg.nfft, 1/cfg.sample_hz)}


def analyze_window(times, ratios, start, cfg=FarSenseConfig()):
    support, grid, uniform, bridges = uniform_window(times, ratios, start, cfg)
    if support['status'] != 'ready':
        return {**support, 'candidate_bpm': None, 'selected_groups': [], 'best_group': None}, None
    result, arrays = estimate_core(uniform, cfg)
    return {**support, **result}, {**arrays, 'times': grid, 'ratios': uniform, 'in_gap': bridges}
