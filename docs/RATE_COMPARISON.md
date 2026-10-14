# One-minute Log Ratio rate comparison

`joint-rate` adds one quantitative step to the joint packet plots: a predefined
60 s average periodic-rate estimate, followed by comparison with manual XBee
button counts. It does not detect individual breaths or calculate an accuracy
percentage. The original `joint`, `joint-fixed` and replay GUI remain unchanged.

```bash
.venv/bin/python -m wifi_csi_breathing joint-rate \
  --input /absolute/path/to/verified-transferred-session \
  --start 10 \
  --output results/rate-comparison-v1
```

The output directory must be new and outside the source recording. Source and
implementation hashes are verified; `rate-report.json` is written last as the
completion marker. Raw data, participant information and generated results must
not be committed. This command never starts acquisition or connects to hardware.

## Fixed choices for this first step

- The interval is exactly `[start, start + 60)` on the stored Ubuntu host clock;
  the default is `[10, 70)` s. Choose the start before inspecting agreement. It
  never searches for, shifts to, or shortens a window to obtain a result.
- The feature is `ln(abs(H1)) - ln(abs(H2))`, with stream 1, RX 1, reference RX 2
  and reported subcarrier group 15 by default. The usual explicit `--tx`, `--rx`,
  `--reference-rx` and `--subcarrier` options are available; there is no ranking.
- All rate flags within reported 3x3 are retained. Other shapes are audited
  missing observations, never zeroes or time-compressed values. Pooling rates is
  exploratory and does not establish equivalence of physical channel mappings.
- The existing past-only median, elapsed-time smoothing, 20 Hz resampling and
  causal 0.1-0.5 Hz bandpass are reused. Each segment needs 10 s of filter warm-up.
  Gaps up to 0.2 s can be interpolated after the right endpoint arrives. Longer
  gaps, duplicate times, unknown records and invalid features reset processing.

All available prehistory is filtered. The same uninterrupted, settled segment
must supply all 1,200 grid samples in the requested minute. The first packet
time anchors the grid; a late recording start or a reset can leave the requested
window without sufficient warm-up. Such a window is unavailable rather than
silently moved. A final interpolation endpoint can arrive at most 0.2 s after
the window end; `available_at` records this latency. No extrapolation is used.

## Estimate and reference

The existing detrend, Hann-window FFT and autocorrelation estimator is reused
with a 60 s window. Its 0.1-0.5 Hz band corresponds to 6-30 /min. The FFT peak
frequency times 60 gives a candidate rate only if the existing autocorrelation,
spectral concentration and band-energy gates pass. A 60 s FFT has 1 /min bin
spacing, which is frequency resolution, not a guaranteed error bound. Flat,
insufficient or ambiguous signals receive no candidate or numerical error.

Only after CSI estimation, received manual events in exactly the same half-open
window are counted: `manual rate = 60 * count / 60`. An event at the left boundary
is included; one at the right boundary is excluded. No reaction-time adjustment,
marker-guided frequency selection or waveform shifting is performed. The causal
filter has delay, so its peak timestamps are not inhalation-onset detections.

Sequence gaps that may overlap the window, changed/multiple sender identities
and non-normal marker quality require review: received counts remain visible,
but the reference rate and error are withheld. No received events is not evidence
of zero breathing. Clean counters do not prove that every breath was pressed.
The reference remains a manual self-report, not independent physiological truth.

`absolute_error_bpm = abs(candidate_bpm - manual_reference_bpm)` is reported only
when both are available. This is one window's discrepancy against manual counts,
not population MAE or independently validated accuracy. Finite-window button
counts also depend on where the interval boundaries fall within a breath cycle.

## Outputs and development checks

`index.html` presents a one-row comparison and three panels: packet Log Ratio,
settled filtered signal, and manual events. PNG/PDF/SVG, `comparison.csv`,
`selection.csv`, `waveform.csv`, `markers.csv` and the JSON completion report
support inspection. Rate/shape codes and excluded records remain in the audit.
The report also includes FFT/ACF diagnostics, quality statistics and timestamps.

Retained packet fraction uses indexed records as its denominator; it is not a
network delivery rate. Supported grid fraction includes short-gap interpolation
and excludes warm-up; it is not a raw sample fraction. Both are reported so a
smooth waveform is not mistaken for complete packet reception.

Synthetic tests cover a known rate, mixed rate codes, short shape interruptions,
long gaps, warm-up, flat signals, causal prefix invariance, exact event-window
boundaries, missing markers, export integrity and non-overwriting behavior.
These development checks do not validate an experimental recording or establish
respiratory sensing performance.
