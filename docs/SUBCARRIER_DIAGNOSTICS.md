# All-subcarrier packet diagnostics

This offline command extends the shape-separated batch figures to all 30 Intel
5300 measured subcarrier groups. It reads an existing verified six-session joint
archive. It never starts capture, connects to Ubuntu, changes source recordings,
or changes the existing fixed-3x3 causal replay processor.

```bash
.venv/bin/python -m wifi_csi_breathing subcarriers \
  --archive data/archives/B004 \
  --output results/b004-subcarriers-v1
```

The default modes are `amplitude` and `phase_difference`. Add repeated `--mode`
options to explicitly select any of `amplitude`, `log_ratio`, or
`phase_difference`. For example, `--mode log_ratio` exports just log amplitude
ratio. Default observations use stream 1, RX 1, and reference RX 2. These are
configurable with `--tx`, `--rx`, and `--reference-rx`.

## Reading the results

Open `index.html`. Each mode has a centered six-session overview, followed by
expandable session figures. Every figure has PNG, PDF, and SVG versions:

* Heatmaps show original packet features beside centered features. Rows separate
  1x3, 2x3, and 3x3. The vertical axis indexes the 30 measured subcarrier groups.
* 3D waterfalls show all groups using centered features. Each shape has its own
  Z-axis range, explicitly labeled. No surface fills missing measurements.
* Example waveforms use fixed groups 1, 15, and 30, with manual marker ticks on
  the shared host-relative axis. They are not selected by apparent periodicity.
  Use `--groups 1 10 20 30` to choose up to four fixed examples.

Each point uses `(host_monotonic_ns - T0_ns) / 1e9` after integer subtraction.
Only the half-open `[T0, T1)` interval is included. Sessions share relative axis
limits, not absolute clock origins or respiratory phase. Button events retain
transport and human-response timing uncertainty and are not physiological ground
truth. Heatmaps and their marker rows use identical time-axis widths; colorbars
occupy separate layout columns so equal timestamps align vertically.

Gray heatmap areas mean no valid displayed sample, not absent breathing. Each
packet is drawn as a time cell capped at 0.05 s and clipped by the midpoints to
its neighbors, including neighbors of other configurations. This is a display
glyph, not interpolation or proof of continuous coverage. Duplicate timestamps
have zero-width heatmap glyphs but remain in saved arrays and point plots. Change
the cap with `--cell-seconds`; it must not exceed `--max-gap`.

Waterfall and waveform lines break at shape changes, reported rate changes,
nonconsecutive record indices, non-increasing timestamps, invalid values, and
gaps longer than `--max-gap` (default 0.2 s). Wrapped phase jumps exceeding pi
also break lines. Original packet values remain present, including the first
point after a break. No point decimation, smoothing, respiratory bandpass,
interpolation, phase unwrapping, or automatic channel search is applied.

## Feature definitions and retrospective centering

For group k at packet n, let a be the CSI entry for the selected stream/RX and b
the same-stream reference RX. The definitions match the existing packet views:

* Scaled amplitude: `s[n] * abs(a[n,k])`, where s is the existing Intel 5300
  RSSI/AGC/noise scaling factor. Units are relative linear amplitude, not dBm.
* Log amplitude ratio: `ln(abs(a[n,k])) - ln(abs(b[n,k]))`.
* Wrapped phase difference: `arg(a[n,k] * conjugate(b[n,k]))`, in radians.

An undefined scale or a zero entry needed for a ratio/phase difference yields an
invalid value. An unavailable selected stream/RX similarly yields NaN, not zero.

For amplitude and log ratio, subtract each group's median over valid packets of
the same shape in that session. For phase, compute the circular mean angle
`theta = arg(mean(exp(j*phase)))` and plot `arg(exp(j*(phase-theta)))`. If the
mean resultant magnitude is below 1e-8, the circular center is undefined and
centered phase is unavailable. Original wrapped phase remains available.

These baselines use the whole session and are explicitly **retrospective**.
They are visualization aids, not a new causal filter or an implemented learning
algorithm. No division by standard deviation amplifies low-variance channels.
Configuration-specific centering does not make physical channels interchangeable.
Rate-dependent mappings and hardware disturbances can remain in each panel.

Original heatmaps use full-range color limits shared across all sessions and
shapes for a mode. Centered amplitude/ratio heatmaps use a symmetric limit at the
99th percentile of absolute residuals, separately for each shape and shared
across all six sessions. Centered scales can differ between shape rows. Use
`--color-percentile 100` for full-range centered colors. Color saturation never
alters saved values or waveform/waterfall data. Phase colors always span
[-pi, pi]. The report records limits and saturated-value counts. Colored bands
or periodic agreement alone do not establish respiratory origin.

## Data integrity and reproducibility

Original B004 transfer files and existing results remain unchanged. Output must
be a new directory outside the raw archive. It is refused if it already exists.
The archive, correction manifest, and implementation hashes are verified before
and after export. A source-code change during export invalidates the run.
`subcarrier-report.json` is written last, only after successful verification and
export. A partial directory without that completion report is not a valid run.

The report stores configuration, input hashes, implementation hashes, output
hashes, timing, shape counts, and color-saturation statistics. Each session/mode
also has `*-samples.npz`, loadable with `numpy.load(path, allow_pickle=False)`:

* `times`, `record_indices`, `shapes`, `rates`, `segments`: packet provenance.
* `raw`, `centered`: N by 30 feature arrays. Missing values are NaN.
* `centers`, `phase_resultant`: 3 by 30 baseline and circular-quality arrays.
* `markers`: all accepted in-window manual event times.
* `edges`, `owner`: display cell boundaries and source packet indices (-1 is a gap).

Data, participant identifiers, and output artifacts remain ignored by Git.
Source code, synthetic tests, and documentation can be committed separately after
review. Results are exploratory and do not quantify respiratory detection accuracy.

## Tests

Synthetic tests cover agreement with single-group features, wrapped phase and
circular centering, constant/invalid inputs, shape and rate boundaries, long
gaps, duplicate timestamps, integer host timing, marker positions and rendered
time-axis alignment, full 30-group exports, fixed examples, non-overwrite
protection, source integrity, and incomplete output behavior. Run the full suite using the existing README command.
