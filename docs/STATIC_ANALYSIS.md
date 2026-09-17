# Shared-scale six-session static analysis

## Scope

This extension adds presentation and batch diagnostics, not a new breathing
detector. The decoder, feature extraction, causal processor, candidate thresholds,
and replay GUI remain unchanged. The saved preview baseline is commit `de9223e`.
No PCA, VMD, automatic channel selection, augmentation, or live input is added.

Run `python -m wifi_csi_breathing static --help` inside the project environment.
The command requires six unique catalog records: two distinct trial numbers for
each of Empty, normal breathing at F0, and normal breathing at R90. It validates
metadata and sorts conditions/trials rather than trusting catalog order. Empty
Trial 3 remains Trial 3; the comparison does not rename it Trial 2. Labels control
panel placement only, not feature extraction, filtering, or estimation.

## Fair comparison and exclusions

- Time zero is each file's first CSI packet, not UTC or a synchronized breath.
  The default comparison interval is 0–120 s. A sample is included only if both
  its timestamp and its interpolation availability time lie within the interval.
- Full-file observations, waveforms, estimates, and provenance remain under
  `numerical/`; the first empty recording retains its original failed status and
  approximately 125 s span. No source file is modified or silently trimmed.
- All panels within a figure share limits. Different feature types have different
  units and are deliberately not forced to share one vertical scale.
- Smoothed-feature panels display the pre-bandpass feature, not original packet
  samples. Phase is unwrapped within each segment; cross-segment phase offsets
  are not physically comparable or calibrated displacement.
- Gray spans show unavailable processed intervals. Hatched spans show the first
  10 s of each processing segment. Smoothed inputs are visible there, but filtered
  comparison traces and spectrum windows exclude those settling samples.
- The 10 s settling rule is inherited and provisional, not a guarantee of zero
  filter transient. Gaps, invalid features, duplicate times, and shape transitions
  reset processing. No plot line or spectrum window crosses a segment boundary.
- Settled coverage is the number of eligible 20 Hz grid samples divided by
  sample rate and comparison duration, capped at 100%. It is an approximate grid
  occupancy measure, not physiological signal quality or a probability.
- Normalized plots use one population mean and standard deviation per recording
  and feature, computed from settled filtered values in the comparison interval.
  They compare shape only. Flat/empty signals have no normalized trace; use
  non-normalized plots and the table for magnitude comparisons.

## Spectra and periodicity

Spectra use the **smoothed input before bandpass**, not the filtered waveform.
This avoids making a band-limited-looking result a foregone conclusion of the
bandpass itself. After the segment settling exclusion, use 30 s windows (600
samples at 20 Hz), 50% overlap, linear detrending, a Hann taper, and one-sided
power spectral density scaling. Average periodograms with equal window weights.
All windows have identical lengths; no segment concatenation, short-window
fallback, or zero-padding changes the frequency grid. Fewer eligible windows
mean a less stable spectral average, not an automatically worse physiological
record. The count is displayed explicitly.

The plot displays 0–1 Hz on a common linear scale; the CSV retains the entire
0–10 Hz spectrum. The shaded 0.1–0.5 Hz band is an exploratory analysis band.
Frequency-bin spacing is 1/30 Hz, equivalent to 2 cycles/min; Hann main-lobe
width is wider than one bin. Display decimals do not imply finer accuracy.
Band power fraction is the sum of PSD bins in the band divided by the sum of
all positive-frequency bins, not the fraction of breathing in the recording.
Input-band peak is reported even for empty data when nonzero power exists; it
must not be called a measured respiration rate.

Separate periodicity figures show the unchanged baseline's trailing 30 s
diagnostics every 2 s: spectral peak, autocorrelation peak, and dots for heuristic
agreement. These windows strongly overlap and are not independent trials. A
periodic candidate is not a breathing or presence decision. No peak-to-peak breath
count is imposed on irregular traces and no target such as 15 breaths/min is used.

## Outputs and reproducibility

- `index.html`: local, network-free figure gallery and processed-coverage table.
- `SUMMARY.md`: quality, feature statistics, interpretation, and figure links.
- `figures/`: 15 figures (five kinds for three features), each PNG/PDF/SVG.
  White background, shared axes, English labels, colorblind-friendly condition
  colors, six labeled panels, and embedded TrueType fonts in PDF. PNG is 300 dpi.
- `quality.csv`: 18 recording/feature rows. Full-file and common-interval metrics
  are explicitly named; packet rate is not packet loss and candidate count is
  not accuracy.
- `spectra.csv`: complete pre-bandpass PSD values, without invented spectra for
  records lacking a continuous settled 30 s window.
- `numerical/`: full-record exports produced by the existing `analyze` command.
- `static-report.json`: final completion marker, settings, dependency versions,
  source hashes, implementation hashes, and output artifact hashes. An interrupted
  run may have partial files but no static completion marker. Use a new directory
  for a retry; existing output directories are never overwritten.

Results, source metadata, and machine-specific paths remain in ignored local
folders. Keep separate backups; do not publish these artifacts without reviewing
privacy. No reference respiration was captured, so these plots cannot establish
rate error, clinical normality, apnea, or cross-person/cross-room generalization.

## Validation

New synthetic checks cover metadata-based panel order, duplicate/missing records,
common-interval clipping including availability time, per-segment settling,
segment-safe spectra and known sinusoid frequency/power, shape normalization,
shared axes and disconnected plot lines, source preservation, output refusal,
and the missing completion marker after a simulated rendering failure. Existing
decoder, processing, and GUI tests continue to apply.
