# Exploratory processing contract

## Input and timing

The catalog references immutable local copies of `csi.dat`, `metadata.json`,
`ping.log`, and `receiver.log`, with byte counts and SHA-256 digests from the
existing experiment verification manifest. All four files are checked before
reading and after successful analysis. The original first empty recording still
has `status: failed`; its diagnostic use does not rewrite that historical status.

Intel 5300 CSI timestamps are relative microsecond hardware counts, not UTC.
The reader handles a forward 32-bit rollover and rejects a backward discontinuity
instead of inventing a uniform clock. The half-range interpretation assumes no
silent gap longer than about 35.8 minutes; the pilot recordings are about two
minutes. Duplicate timestamps reset processing. Nominal 30 Hz is metadata only;
actual packet timing drives both analysis and replay.

The current feature baseline requires 3 TX x 3 RX. Unsupported shapes are skipped;
every shape transition resets state even if a particular channel exists on both
sides. This is conservative and may discard potentially usable windows, but it
does not silently splice different spatial mappings.

## Features

Channels and subcarrier group are explicit, 1-based settings. Initial defaults
are TX 1, RX 1, reference RX 2, and group 15 of 30. They are fixed across all six
recordings, not selected using the activity label or future data.

| Mode | Feature | Units / limitation |
| --- | --- | --- |
| `amplitude` | Absolute CSI multiplied by the decoder's scaling factor | sqrt(SNR)-style units, not dBm or millimeters |
| `log_ratio` | log(abs(RX)) minus log(abs(reference RX)), same TX/group | Unitless; zeros are invalid; can reduce some shared amplitude effects, not guaranteed |
| `phase_difference` | angle(CSI_RX × conjugate(CSI_reference_RX)) | Radians with causal within-segment unwrap; not calibrated displacement or a complete phase-error correction |

Raw absolute phase is not interpreted as chest motion. No radar wavelength,
range-bin logic, range FFT, or TI phase-to-displacement conversion is used.

## Streaming stages

Each `Observation` is processed only after it arrives:

1. Validate its timestamp, feature and segment. Reset on a missing/nonfinite
   feature, changed dimension, duplicate timestamp, or interval over 0.2 s.
2. For phase difference only, unwrap using the previous valid value, within the
   current segment. No full-record unwrapping uses a future sample.
3. Apply a past-only median of up to three packets and elapsed-time-aware
   first-order 3 Hz smoothing.
4. Resample to a 20 Hz grid using the two already-arrived endpoints. Emit a grid
   sample only after its right endpoint arrives; record both `time` and
   `available_at`. Never interpolate across a reset/gap.
5. Subtract the segment's first-value anchor and run a newly designed SciPy
   Butterworth order-2 bandpass (0.1–0.5 Hz) with persistent SOS state. Coefficients
   are not copied from TI. No `filtfilt` or other future-dependent filtering.
6. Exclude 10 seconds of segment settling, then retain a trailing 30-second
   analysis window. This is a provisional settling rule, not a proven bound
   on all possible filter transients. Compute diagnostics every 2 seconds.

The 3 Hz smoother is a modest high-frequency reduction, not a rigorously designed
anti-alias solution for arbitrary low packet rates or strong interference.
Long-gap rejection alone does not guarantee sufficient local sampling density.
Sampling-quality gates and stronger rate-aware preprocessing are future work.
Replay is causally ordered, but recordings are verified/decoded in a worker before
playback begins; this is not a live transport implementation.

## Candidate diagnostics

Detrend the **trailing window** and calculate a Hann-windowed real spectrum. The
largest 0.1–0.5 Hz peak gives a spectral rate candidate. A 30-second window has
approximately 0.033 Hz (2 cycles/minute) bin spacing; decimal display is not a
claim of fine rate accuracy. Compute a normalized biased autocorrelation and its
strongest local maximum at lags corresponding to that frequency band.

Current provisional gates require:

- Spectral and autocorrelation rates agree within 3 cycles/minute.
- Autocorrelation peak at least 0.5.
- At least 0.6 of in-band spectral power lies within the peak's adjacent three bins.
- At least 0.4 of detrended input-feature power lies in the analysis band.

These thresholds are engineering starting values, not calibrated probabilities,
validated motion rejection, or established respiration criteria. No independent
ground truth was recorded for these six sessions. The code does not claim that
passing these gates means a person is present, or that failing means no breathing.
The output status is `periodic_candidate`, `inconclusive`, or
`insufficient_variation`; a weak or missing window has no current numerical rate.

The signal band represents 6–30 cycles/minute. Rates outside that range are not
covered by this exploratory configuration. No heart-rate estimation is attempted.

## Replay / offline equivalence

Offline export and GUI replay both call `Processor.feed` in packet order. Speed
does not enter the processor. Replay uses absolute monotonic deadlines based on
recorded timestamps, subtracting pause time. Processing delays do not accumulate
into a new synthetic sample clock; a slow machine can lag behind its target pace.

The display queue is bounded. If the UI is slow, older **display snapshots** can
be replaced; every raw observation still passes through the processor. Paused,
stopped, stale, failed, and EOF states clear the numerical current candidate.
The remaining curve is explicitly historical. No timing gap is represented by
a fabricated continuation of a previous rate.

## Evaluation boundaries

F0 and R90 are operator-defined orientations, not an algorithmic guarantee that
one is better. Session labels remain in audit/GUI output only. Six recordings
cannot establish cross-person, cross-room, or clinical performance. The two empty
recordings are valuable negative controls. Compare false periodic candidates as
well as apparent respiratory peaks before choosing or tuning a feature.

Future evaluation should use synchronized ground truth, explicit motion/absence
states, independent recording-level splits, and channel-selection rules that
do not inspect the held-out record's future or labels. Avoid claiming accuracy
from window counts or from seeing a plausible-looking waveform.
