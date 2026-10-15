# WiResP: second-batch adapted reproduction

Source: Wei-Hsiang Wang et al., *WiResP: A Robust Wi-Fi-Based Respiration
Monitoring via Spectrum Enhancement*, IEEE Sensors Journal 24(13),
20999-21011 (2024), DOI <https://doi.org/10.1109/JSEN.2024.3399110>.
The final 13-page PDF was read from the local Zotero library; equations and
figures on pages 4-6 were visually checked. No author implementation was
verified. This code is independently written, not copied from another system.

This is **WiResP-adapted-v1**, not a complete replication of the authors'
calibrated system or published accuracy. The article does not supply numeric
values for Tw, K, motion threshold eta, the median/LOESS spans, and the final
pulse/template thresholds. It says Te usually spans five to seven cycles and
notes that the continuity rule requires application-dependent tuning. This
implementation makes these omissions explicit instead of inferring parameters
from manual labels or favorable real-data results.

## Mathematical implementation

All functions below are in `src/wifi_csi_breathing/baselines/wiresp.py`.

| Paper step | Implementation | Convention |
| --- | --- | --- |
| V-A CSI normalization | `normalized_amplitudes` | First effective stream, RX1-3, 30 groups each: 90 amplitude channels. For each packet/RX, divide magnitudes by their RMS across 30 groups. This removes common gain but can also attenuate common respiration modulation. |
| V-A median preprocessing | `analyze_frame` | Five-point median on normalized magnitudes, nearest endpoint padding. The paper mentions complex normalization and median filtering before amplitude extraction without defining a complex median; magnitude median is an explicit adaptation. |
| Eq. (5) amplitude covariance ACF | `covariance_acf` | Global mean subtraction; lag-k covariance uses N-k overlaps, divided by lag-zero variance. High-lag estimates can exceed one. This differs from FarSense's finite-sum normalization. |
| Eq. (6) motion statistic | `analyze_frame` | Mean first-lag ACF of normalized CSI **power**, before median filtering, as the prose specifies. The equation's reused rho notation is ambiguous. Statistic exported; threshold is unset by default. |
| V-C1 BNR/top-K | `breathing_noise_ratio`, `analyze_frame` | Positive-frequency energy in 10-30/min divided by out-of-band energy, DC excluded. Mean removal, Hann taper and 2048-point FFT are conventions. K=10, stable ranking; zero-energy channels excluded. |
| V-C1 combining | `analyze_frame` | Sum of selected ACFs times their BNR, not a waveform average. No weight normalization before enhancement. |
| V-C1 LOESS | `loess_matrix` | Local quadratic regression, tricube weights, 11 samples, no robust iterations. |
| Eq. (7) image enhancement | `equalize_ecdf` | Exact empirical CDF of the entire trailing ACF image; ties use <=. No per-column normalization, quantization or learned transformation. |
| Eqs. (8)-(9) Otsu | `otsu_threshold`, `enhance_history` | Exact maximum between-class variance over possible splits; midpoint threshold; high class includes equality. |
| Eq. (10) binary trend | `enhance_history` | Average binary image over its time columns; threshold at >=0.5. |
| V-D pulse/template examination | `pulse_check` | Explicit experimental rule below; not the authors' unavailable calibration. |
| V-E rate | `first_peak`, `enhance_history` | First local maximum at lag 2-6 s, 60/lag. Enhanced latest column used after continuity. No fitting to buttons or harmonic correction. |

An ACF spectrum here means an **image of lag versus observation time**, not an
FFT frequency spectrum. Global histogram equalization is monotonic: it improves
contrast but does not create new information or change the ideal ordering of
peaks. The original and enhanced images are both exported.

## Parameters fixed before the real-data comparison

| Setting | This adaptation | Paper |
| --- | --- | --- |
| Sampling grid | 20 Hz, timestamp interpolation | 30 Hz acquisition on Qualcomm hardware |
| Spatial input | Stream 1 across 1x3/2x3/3x3; 3 RX chains, 30 groups | 1 TX, 2 RX; 128 subcarriers with 24 reserved |
| ACF window Tw | 15 s | Numeric setting not given |
| ACF snapshot hop | 1 s | Image dimension notation suggests Fs snapshots/s; reduced update cadence is an adaptation |
| Enhancement Te | 30 snapshots at 1 s hop | Typically 5-7 breathing cycles, no fixed duration |
| Candidate band | 10-30/min | Footnote 2: 10-30/min |
| Subcarrier count | Top 10 of up to 90 channels | Empirical K, much smaller than total |
| Gap handling | <=0.5 s interpolation; grid points in gaps >0.2 s audited; <=5% per frame | Not specified |
| Motion gate | Statistic available; eta=None, **not calibrated/enabled** | Empirical eta; no numeric value |
| Continuity gate | At least 2 complete interior pulses; period 2-6 s; period CV <=0.25; mean duty 0.15-0.85; cosine-sign template mismatch <=0.25 | Pulse count, widths and reference pattern described; thresholds not supplied |

For the adapted continuity rule, clipped pulses at the lag-axis endpoints are
excluded. Full-pulse center intervals give a median period; pulse widths give
duty fraction. The reference is the sign of a cosine with that period, aligned
to the first full pulse. Mismatch is evaluated only between the first and last
full pulses. This is a transparent implementation choice, not a learned or
physiologically validated breathing detector. In particular, passing this rule
while the motion threshold is unset is not the full paper's acceptance decision.

## Timing and comparison

Use the exact predefined 60 s intervals from the first-batch manifest. Each
interval contains 46 trailing 15 s frames at one-second spacing, ending from
start+15 through start+60. No support is silently compressed or gap-filled beyond
the stated limits. A malformed record or duplicate timestamp remains a barrier.
Invalid normalization excludes that RX's channels for affected frames.

The raw WiResP minute candidate is the arithmetic mean of **all 46** frame
candidates, only if every frame yields one. Available-only mean is exported
separately and never substituted. These overlapping frames are not independent
trials. The continuity checker first has 30 snapshots at start+44 s; its
available history is explicit, not a filter-restart blackout. Earlier raw
candidates and ACFs remain visible. Each decision uses only its trailing history,
and is available at frame completion or its required interpolation endpoint.
Plots inside completed frames are retrospective, not sample-level live output.

The after-check mean includes only passing frames and is **partial coverage**.
It is not used as a complete-minute value in the four-method MAE comparison.
Report candidate/eligible/passing frame counts alongside it. A future fair
detector benchmark needs calibrated motion and continuity gates and held-out
data; raw candidate generation alone cannot support a false-alarm-rate claim.

LogRatio-FFT and LogRatio-ACF retain the previous 60 s signal and estimators.
FarSense retains five nonoverlapping 12 s windows and its complete-five mean.
WiResP retains its amplitude-specific preprocessing rather than being fed the
existing log ratio. Inputs, window lengths and search bands therefore differ
by method: this is an end-to-end exploratory system comparison, not an isolated
estimator ablation. Errors use identical received manual counts, which are
self-reports rather than a physiological reference sensor. Empty controls remain
separate and are not assigned zero respiratory-rate truth.

## Run and inspect

```bash
.venv/bin/python -m wifi_csi_breathing compare-wiresp \
  --manifest /absolute/path/to/comparison-manifest.json \
  --output results/wiresp-comparison-v1
```

The manifest uses the existing `compare-farsense` schema. The command reruns
the three previous methods unchanged under `reference-methods/`, then adds
WiResP. It never starts capture, connects to Ubuntu or edits transferred data.
Existing output directories are refused. Inputs/receipts, implementation and
artifacts are hashed; a completion report is written only after verification.

Outputs include a four-method overview, per-minute raw/enhanced ACF images,
binary history/trend and gate diagnostics, frame-level decisions, normalized
inputs and intermediate arrays, packet configuration audit, and per-method
availability/MAE on available and common windows. Real recordings, local
manifests, paper PDFs, participant identifiers and generated results are not
intended for Git. Numerical parameters must be revised as a named version,
not quietly fitted to these reused pilot windows.
