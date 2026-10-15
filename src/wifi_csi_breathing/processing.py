"""Independent, causal exploratory processing; not a clinical rate estimator.

No TI firmware implementation or filter coefficients are used here. Labels and
future samples are deliberately absent from this module's inputs.
"""
from collections import deque
from dataclasses import dataclass
import math

import numpy as np
from scipy import signal

from .data import Observation


@dataclass(frozen=True)
class ProcessingConfig:
    sample_hz: float = 20.0
    low_hz: float = 0.1
    high_hz: float = 0.5
    warmup_seconds: float = 10.0
    window_seconds: float = 30.0
    estimate_every: float = 2.0
    max_gap_seconds: float = 0.2

    def __post_init__(self):
        values = tuple(vars(self).values())
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError('Processing settings must be positive and finite')
        if not self.low_hz < self.high_hz < self.sample_hz / 2:
            raise ValueError('Band must lie strictly inside Nyquist frequency')
        if self.window_seconds * self.low_hz < 3:
            raise ValueError('Window must contain at least three low-band cycles')


@dataclass(frozen=True)
class Point:
    time: float
    available_at: float
    raw: float
    filtered: float
    segment: int


@dataclass(frozen=True)
class Estimate:
    time: float
    available_at: float
    segment: int
    status: str
    candidate_bpm: float | None
    spectral_peak_bpm: float | None
    autocorrelation_bpm: float | None
    autocorrelation: float
    concentration: float
    band_fraction: float


def quality_checks(peak_bpm, ac_bpm, ac_value, concentration, fraction):
    """The existing four heuristic gates, shared by estimation and reporting."""
    difference = abs(peak_bpm - ac_bpm) if ac_bpm is not None else None
    return [
        {'name': 'Frequency agreement', 'value': difference, 'operator': '<=',
         'threshold': 3.0, 'unit': '/min', 'passed': difference is not None and difference <= 3.0},
        {'name': 'Autocorrelation peak', 'value': ac_value, 'operator': '>=',
         'threshold': 0.5, 'unit': 'unitless', 'passed': ac_value >= 0.5},
        {'name': 'Spectral concentration', 'value': concentration, 'operator': '>=',
         'threshold': 0.6, 'unit': 'fraction', 'passed': concentration >= 0.6},
        {'name': 'Pre-bandpass energy fraction', 'value': fraction, 'operator': '>=',
         'threshold': 0.4, 'unit': 'fraction', 'passed': fraction >= 0.4},
    ]


def inspect_window(points, cfg):
    """Return the estimate and the exact arrays behind its FFT/ACF diagnostics.

    This is the single numerical path for estimation and exported plots. No
    manual events, extra padding, spectral smoothing or alternative peaks enter.
    """
    last = points[-1]
    raw = signal.detrend(np.array([p.raw for p in points], dtype=float))
    filtered = signal.detrend(np.array([p.filtered for p in points], dtype=float))
    energy = float(np.dot(filtered, filtered))
    raw_energy = float(np.dot(raw, raw))
    if energy < 1e-16 or raw_energy < 1e-16:
        return (Estimate(last.time, last.available_at, last.segment, 'insufficient_variation',
                         None, None, None, 0.0, 0.0, 0.0), None)
    taper = np.hanning(len(filtered))
    frequencies = np.fft.rfftfreq(len(filtered), 1 / cfg.sample_hz)
    power = np.abs(np.fft.rfft(filtered * taper)) ** 2
    raw_power = np.abs(np.fft.rfft(raw * taper)) ** 2
    band = (frequencies >= cfg.low_hz - 1e-9) & (frequencies <= cfg.high_hz + 1e-9)
    band_indices = np.flatnonzero(band)
    peak = int(band_indices[np.argmax(power[band])])
    peak_hz = float(frequencies[peak])
    # Three-bin concentration is intentionally described as an uncalibrated
    # spectral statistic, not detector confidence or presence probability.
    near = band & (np.abs(frequencies - peak_hz) <= cfg.sample_hz / len(filtered) * 1.01)
    concentration = float(power[near].sum() / max(power[band].sum(), 1e-30))
    fraction = float(raw_power[band].sum() / max(raw_power[1:].sum(), 1e-30))
    ac = signal.correlate(filtered, filtered, mode='full', method='fft')[len(filtered) - 1:]
    ac /= max(float(ac[0]), 1e-30)
    min_lag = int(math.ceil(cfg.sample_hz / cfg.high_hz))
    max_lag = min(len(filtered) - 2, int(math.floor(cfg.sample_hz / cfg.low_hz)))
    peaks, _ = signal.find_peaks(ac)
    peaks = peaks[(peaks >= min_lag) & (peaks <= max_lag)]
    lag = int(peaks[np.argmax(ac[peaks])]) if len(peaks) else None
    ac_value = float(ac[lag]) if lag is not None else 0.0
    ac_bpm = cfg.sample_hz / lag * 60 if lag is not None else None
    checks = quality_checks(peak_hz * 60, ac_bpm, ac_value, concentration, fraction)
    passes = all(check['passed'] for check in checks)
    estimate = Estimate(last.time, last.available_at, last.segment,
                        'periodic_candidate' if passes else 'inconclusive',
                        peak_hz * 60 if passes else None, peak_hz * 60, ac_bpm,
                        ac_value, concentration, fraction)
    return estimate, {'raw_detrended': raw, 'filtered_detrended': filtered,
                      'taper': taper, 'frequencies': frequencies, 'power': power,
                      'raw_power': raw_power, 'band': band, 'near': near, 'peak': peak,
                      'ac': ac, 'min_lag': min_lag, 'max_lag': max_lag,
                      'acf_peaks': peaks, 'lag': lag, 'checks': checks}


def estimate_window(points, cfg):
    """Trailing-window diagnostics; heuristic agreement is not probability."""
    return inspect_window(points, cfg)[0]


class Processor:
    """Feed packet observations in order. Speed and activity labels are not inputs.

    Linear interpolation is emitted only once its right endpoint has arrived;
    available_at records that delay. A missing feature, changed dimension, or
    long gap resets the entire filter/window. No zero-phase filtering is used.
    """
    def __init__(self, mode='amplitude', config=None):
        self.config = config or ProcessingConfig()
        if mode not in ('amplitude', 'log_ratio', 'phase_difference'):
            raise ValueError('Unknown feature mode')
        self.mode = mode
        self.sos = signal.butter(2, (self.config.low_hz, self.config.high_hz),
                                 btype='bandpass', fs=self.config.sample_hz, output='sos')
        self.segment = 0
        self.resets = 0
        self.uniform_count = 0
        self.last_input_time = None
        self.history = deque(maxlen=int(60 * self.config.sample_hz))
        self.latest = None
        self.status = 'waiting'
        self._clear()

    def _clear(self):
        self.previous = None
        self.next_grid = self.segment_start = self.anchor = None
        self.phase_raw = self.phase_unwrapped = None
        self.recent = deque(maxlen=3)
        self.window = deque(maxlen=round(self.config.window_seconds * self.config.sample_hz))
        self.zi = np.zeros((len(self.sos), 2))
        self.history.clear()
        self.last_estimate_time = -math.inf
        self.latest = None

    def _reset(self, reason):
        self.segment += 1
        self.resets += 1
        self._clear()
        self.status = reason

    def feed(self, observation: Observation):
        t, value = observation.time, observation.value
        if not math.isfinite(t) or t < 0:
            raise ValueError('Time must be finite and nonnegative')
        if self.last_input_time is not None and t < self.last_input_time:
            raise ValueError('Observations must be time ordered')
        reason = observation.reason if observation.boundary else ''
        if self.last_input_time is not None:
            if t == self.last_input_time:
                reason, value = 'duplicate_timestamp', None
            elif t - self.last_input_time > self.config.max_gap_seconds:
                reason = 'long_gap'
        self.last_input_time = t
        if value is None or not math.isfinite(value):
            self._reset(reason or 'invalid_feature')
            return [], []
        if reason:
            self._reset(reason)
        if self.mode == 'phase_difference':
            if self.phase_raw is None:
                self.phase_unwrapped = value
            else:
                self.phase_unwrapped += (value - self.phase_raw + math.pi) % (2 * math.pi) - math.pi
            self.phase_raw = value
            value = self.phase_unwrapped
        self.recent.append(value)
        value = float(np.median(self.recent))  # Past-only three-packet median.
        if self.previous is None:
            self.previous = (t, value)
            self.next_grid = self.segment_start = t
            self.anchor = value
            self.status = 'warming_up'
        else:
            # First-order, elapsed-time-aware smoothing before resampling. This
            # reduces high-frequency content, but is not a precision anti-alias
            # filter for arbitrary packet rates (see METHODS.md).
            dt = t - self.previous[0]
            alpha = -math.expm1(-2 * math.pi * 3.0 * dt)
            value = self.previous[1] + alpha * (value - self.previous[1])
        left_t, left_value = self.previous
        points, estimates = [], []
        while self.next_grid <= t + 1e-9:
            grid = self.next_grid
            interpolated = value if t == left_t else left_value + (value - left_value) * (grid - left_t) / (t - left_t)
            filtered, self.zi = signal.sosfilt(self.sos, [interpolated - self.anchor], zi=self.zi)
            point = Point(grid, t, float(interpolated), float(filtered[0]), self.segment)
            points.append(point)
            self.history.append(point)
            self.uniform_count += 1
            if grid - self.segment_start >= self.config.warmup_seconds - 1e-9:
                self.window.append(point)
            if len(self.window) == self.window.maxlen and grid - self.last_estimate_time >= self.config.estimate_every - 1e-9:
                self.latest = estimate_window(list(self.window), self.config)
                self.status = self.latest.status
                self.last_estimate_time = grid
                estimates.append(self.latest)
            self.next_grid += 1 / self.config.sample_hz
        self.previous = (t, value)
        return points, estimates
