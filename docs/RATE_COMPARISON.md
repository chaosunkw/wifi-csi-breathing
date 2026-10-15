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

## Default baseline

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

## Opt-in first-stream experiment

The default remains `--shape-policy 3x3`. To compare a common first-stream
feature across packet configurations, write a separate output directory:

```bash
.venv/bin/python -m wifi_csi_breathing joint-rate \
  --input /absolute/path/to/verified-transferred-session \
  --start 10 --shape-policy first-stream \
  --output results/rate-first-stream-v1
```

This mode requires `--tx 1` and accepts reported 1x3, 2x3 and 3x3 packets.
Each valid packet contributes exactly one log ratio, using stream 1, the same
two RX indices and the same reported subcarrier group. The absent streams are
not zero-filled, ranked or averaged; the remaining streams do not contribute.
All reported rate flags are retained. A change in stream count alone does not
reset the filter. Real long gaps, unknown records, duplicate timestamps and
invalid selected values still reset; the 0.2 s gap limit, 10 s warm-up, 60 s
window and quality gates are unchanged. Other RX dimensions remain excluded.

Common indices do not guarantee a common physical channel across configurations.
A common nonzero gain cancels in the ratio, but spatial mapping or RX-specific
changes may produce offsets and spurious periodicity. This experiment applies
no mapping inversion, per-configuration centering, transition correction or
marker-guided fitting. A synthetic offset test verifies that jumps remain in
the selected raw values. Increased output availability is not proof of greater
accuracy or person presence.

`selection.csv` retains every packet's shape, rate and `shape_changed` indicator.
The report records the policy, retained shape counts and transition times. In
first-stream plots, raw point colors identify 1x3/2x3/3x3 and a separate Ntx panel
shows the reported stream count. Gray still means filter warm-up. Compare both
modes on identical preselected windows, including empty-room and unavailable
results, before making claims about the new mode. The baseline and shared
feature extractor, replay processor and fixed-configuration mode are unchanged.

## Opt-in short-gap state preservation

Add `--gap-policy preserve-short` to `joint-rate` and use a new output directory.
This option is independent of packet selection; use `--shape-policy first-stream`
to include 1x3/2x3/3x3 while comparing the old and new gap rules on the same input.
The default `--gap-policy reset` preserves the original numerical behavior.

- Intervals up to 0.2 s retain the existing resampling behavior.
- Intervals over 0.2 s and at most 0.5 s no longer clear the filter state or
  restart settling. The existing linear resampler advances the filter on its
  original 20 Hz time grid, only after the right endpoint arrives. These internal
  values are interpolation assumptions, not recovered CSI packets. They feed
  both the filter and FFT/ACF calculation, and are explicitly audited.
- The plot masks every grid point strictly between the endpoints of such a
  gap. It does not connect a line across it. The gap retains its actual duration;
  time is never compressed. The 0.2 s display threshold distinguishes interruptions
  from the normal inter-packet intervals already resampled by the baseline.
- A minute remains unavailable if more than 5% of its grid falls inside the
  added bridges. The 0.5 s bound and 5% cap are provisional engineering limits,
  selected before comparing manual agreement, not validated error guarantees.
  Bridges can smooth or distort real motion; no data-driven optimality is claimed.
- Intervals over 0.5 s, invalid values, duplicate timestamps and unknown records
  still reset processing. No extrapolation through a trailing outage is used.
  Initial/reset settling is now displayed as dashed amber, not hidden gray.
  These transient values remain excluded from rate estimation for 10 s.
- FFT/ACF quality thresholds, feature selection, 60 s window, manual count and
  initial settling duration are unchanged. A short gap does not itself make a
  candidate reliable, and visible settling does not make a minute rate eligible.

A final interpolation endpoint can arrive up to 0.5 s after the requested end
under this policy. `available_at` records the actual delay; later input never
alters previously emitted points. Unknown endpoint data is never fabricated.

`gaps.csv` and the report list endpoint times, durations and the action taken.
The intervals describe available selected features, not proven wireless packet
loss or a duration of hardware stream switching. `waveform.csv` keeps internal
filter values for reproducibility, with `in_gap`, `settled` and `display_filtered`:
the latter is empty inside gaps. `selection.csv` remains the raw packet audit.
`bridged_grid_fraction` reports added interpolation, and
`settled_unbridged_grid_samples` excludes it. Existing `supported_grid_fraction`
includes interpolation, so 100% there does not mean 100% packet reception.

Tests cover state continuity with a real timestamp gap, unchanged elapsed time,
blank plots, visible transient output, prefix invariance, delayed endpoints,
long/invalid interruptions, excessive bridging, source integrity and export masks.

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

The time-domain curve is the **filtered, dimensionless Log Ratio feature**.
It is not chest displacement, airflow, or a calibrated respiratory waveform.
Its positive/negative values do not identify inhalation/exhalation. It has the
delay of the causal filter. The page labels this explicitly and shows packet
Log Ratio before the processing chain, followed by the filtered feature.

Each eligible minute now includes `estimator-diagnostics.png`/PDF/SVG:

- FFT: the squared rFFT magnitude after linear detrending and a Hann window.
  The plot compares the pre-bandpass smoothed/resampled feature with its filtered
  version. Each spectrum is divided by its own sum across positive-frequency
  bins; this is relative power, not a probability or absolute amplitude. The
  displayed 0-60 cycles/min range includes the 6-30 /min search band. Selected
  peak and neighboring bins show the spectral-concentration calculation.
- Autocorrelation: the linearly detrended filtered feature correlated with
  itself, divided by its zero-lag value, without Hann tapering. The horizontal
  axis is time lag, not recording time. Only local maxima at 2-10 s are searched;
  the selected peak is marked along with its lag, `60 / lag` rate, and the
  existing 0.5 threshold. The view shows 0-15 s.
- A table lists all four existing quality checks, their values, unchanged
  thresholds, and pass/fail results. FFT/ACF frequency candidates remain visible
  for inconclusive windows even when the accepted rate is unavailable.

`spectrum.csv` and `autocorrelation.csv` retain the full arrays, including bins
outside the plotted ranges. `diagnostic-input.csv` gives the detrended input
values and Hann weights at the original uniform grid times; bounded internal
gap interpolation is included exactly as in estimation, not removed and time
compressed. Its count is recorded in `estimator-diagnostics.json`. That JSON
also describes normalizations and gate failures. No manual markers enter the
FFT, correlation, peak selection, or quality checks.

The plots and estimator share `processing.inspect_window`; exports assert exact
agreement with the previously computed result. This is a diagnostic display
change, not a new rate algorithm or a threshold adjustment. Windows without
eligible data or signal variation show an explicit explanation and header-only
diagnostic CSVs. Separate segments are never concatenated to fabricate a full
minute spectrum.

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

Additional synthetic tests cover long runs of 1x3/2x3 packets, common-gain
cancellation, ignored extra streams, uncorrected configuration offsets, invalid
values in smaller shapes, genuine outages, prefix invariance and CLI exports.
