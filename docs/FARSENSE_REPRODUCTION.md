# FarSense core reproduction, first batch

Source: Youwei Zeng et al., *FarSense: Pushing the Range Limit of WiFi-based
Respiration Sensing with CSI Ratio of Two Antennas*, IMWUT 3(3), Article 121,
2019. DOI: https://doi.org/10.1145/3351279; author preprint:
https://arxiv.org/abs/1907.03994. The 26-page final PDF in the local Zotero library
was checked, including equations on PDF pages 13-17. The PDF is not redistributed.

This is an independently implemented **adapted core reproduction**, not the
authors' code, a complete FarSense system, or a replication of the published
accuracy/range. No official executable implementation was verified in this batch.
Source recordings and manual labels never determine parameters or selections.

## Scope and mathematical mapping

The numerical implementation is in `src/wifi_csi_breathing/baselines/farsense.py`.
The report/evaluation adapter is `src/wifi_csi_breathing/farsense_comparison.py`.

| Paper operation | Formula / interpretation | Implementation |
| --- | --- | --- |
| Section 6.2, two-receiver complex ratio | `z_g[n] = H_(stream1,RX1,g)[n] / H_(stream1,RX2,g)[n]` | `complex_ratios` |
| Section 6.2, smoothing | Savitzky-Golay separately on real/imaginary components | `estimate_core` |
| Eq. (9), projection | `y_(g,theta) = Re(z_g) cos(theta) + Im(z_g) sin(theta)` | `projection_search` |
| Section 6.3, 100 directions | `theta = j*pi/50`, `j = 0..99` | `projection_search` |
| Section 5.2.2, BNR | Largest positive FFT-bin energy in 10-37 /min divided by total full FFT energy | `projection_search` |
| Section 6.3, select projection | Maximum BNR for each group; first equivalent maximum breaks ties | `projection_search`, `estimate_core` |
| Eq. (10), ACF | `r_g[k] = sum((y_g[n]-mean(y_g))*(y_g[n-k]-mean(y_g))) / sum((y_g[n]-mean(y_g))^2)` | `normalized_acf` |
| Section 6.4.2, select groups | `BNR_g > 0.7 * max_g(BNR_g)` (strict inequality) | `estimate_core` |
| Eq. (11), fuse | `r_fused[k] = sum_g BNR_g * r_g[k]` over selected groups | `estimate_core` |
| Section 6.4.2, rate | First eligible nonzero local ACF maximum, `rate = 60*fs/k` | `estimate_core` |

The ratio is **complex division**, not the logarithm of an amplitude ratio. All
30 reported Intel groups participate. No median, project bandpass, PCA, harmonic
correction, marker alignment or automatic receive-pair search is added to FarSense.
FFT linearity reduces the computation to two FFTs per group while still evaluating
all 100 direction scores. Numerical tests compare every score to a direct full FFT.

## Explicit adaptations and unresolved details

These choices were fixed before evaluating the existing real recordings.

| Item | Paper | This batch |
| --- | --- | --- |
| Acquisition | 100 Hz, 5.24 GHz, 20 MHz; one transmit/two receive antennas | Existing recordings, approximately 30 requested packets/s, variable reported streams; no radio changes |
| Stream handling | Controlled transmit setup | Stream 1 across 1x3/2x3/3x3, RX1/RX2; all rate flags; no physical-mapping correction |
| Resampling | Uniform collected sequence | Timestamp-based linear resampling at 20 Hz, matching the project's analysis grid; no claim of 100 Hz measurements |
| Time window | 12 s, 1200 samples | 12 s, 240 samples |
| FFT length | 8192 | 2048 at 20 Hz; zero padding is interpolation, not added information |
| SG settings | Window/order and edge convention not specified in the paper | 11 samples (0.55 s nominal window), order 3, `mode='interp'`; within completed blocks only |
| DC and BNR conventions | Exact mean-removal/one-sided implementation not specified | Remove block mean, no Hann or linear detrend; numerator one positive bin, denominator all two-sided bins, including DC (zero after centering) |
| ACF normalization | Eq. (10) | Exact biased/global-mean normalization; no `T/(T-k)` correction |
| Peak selection | First nonzero peak; numerical constraints not specified | First **positive** local maximum in the 10-37 /min lag interval, no prominence gate or peak interpolation |
| Large-motion exclusion | Detector from reference [27], with CSI ratio input | **Not implemented**; intended pilot intervals are stationary; this assumption is not verified by this module |
| Missing samples | Policy not specified | Interpolate only intervals <=0.5 s; audit grid cells inside intervals >0.2 s, cap at 5% per 12 s block; longer gaps make that block unavailable |
| Zero denominator/invalid data | Not fully specified | Do not fabricate a ratio; exclude affected group for the supported block; unknown records/duplicate timestamps block estimation |
| Full-minute comparison | Not specified | Mean of five nonoverlapping 12 s rate candidates; **all five required** for a reported 60 s FarSense result |

The core also accepts `FarSenseConfig(sample_hz=100, nfft=8192, sg_samples=51)`
for genuinely uniform 100 Hz input; a synthetic test covers this configuration.
The 51-point SG choice is still an implementation convention, not a paper value.
Upsampling these recordings would not recreate 100 Hz acquisition.

The paper's first-peak rule and BNR search can produce a rate for periodic noise.
An analytic constant-magnitude complex arc at 18 /min also demonstrates a
possible 36 /min harmonic selection: a very small real-component oscillation
can have a larger normalized BNR than the fundamental imaginary component.
The synthetic regression test retains that behavior; no reference-informed
harmonic repair is added. At 24 /min the second harmonic is outside the search
band, and the corresponding quadrature test recovers the fundamental.
We do not import the project's four acceptance gates into FarSense or claim the
omitted motion gate is an absence detector. Empty controls remain visible with
candidate counts, without treating zero button presses as a 0 /min ground truth.

## Timing and causality

`uniform_window` uses real host-relative timestamps. Each interpolated sample
requires a received right endpoint; `available_at_s` records availability at the
block end or that endpoint, whichever is later. No trailing extrapolation or
time compression is used. Invalid groups remain missing.

SG smoothing and angle selection use the whole **completed** 12 s block. Thus
the reconstructed sample trace is retrospective within that block; it is not a
sample-by-sample causal waveform. The block's rate never uses later blocks.
Tests verify that appending/changing later samples leaves completed-block output
unchanged. Existing causal replay and `joint-rate` behavior are untouched.

## Running the comparison

Create a local ignored manifest (for example under `data/`) with anonymous labels:

```json
{
  "schema_version": 1,
  "recordings": [
    {
      "key": "seated-trial-1",
      "label": "Seated trial 1",
      "input": "/absolute/path/to/verified/session",
      "starts_s": [10, 70],
      "empty_room": false
    }
  ]
}
```

```sh
.venv/bin/python -m wifi_csi_breathing compare-farsense \
  --manifest data/farsense-comparison.json \
  --output results/farsense-comparison-v1
```

Every input is a verified nine-file joint transfer plus receipt. Monitor is never
started. Each minute must fit the recording, paths/keys and duplicate windows
are validated, and existing output directories are never overwritten.

Outputs include an HTML overview, PNG/SVG comparison figures, CSV summary and
metrics, source/code hashes, original packet-shape/rate audits, and per-block:

- result and selected-group metadata;
- all 100 BNR direction scores for all 30 groups;
- BNR/angle/group selection CSV and fused ACF CSV;
- non-pickled numerical NPZ arrays (resampled complex ratios, selected projection
  traces, individual/fused ACFs, scores, spectra, grid timestamps and gap mask).

Raw recordings, Zotero PDFs, local manifests and generated results stay outside
version control. `report.json` is written only after all source hashes are
rechecked and the implementation hash set is unchanged.

## Comparison interpretation

LogRatio-FFT and LogRatio-ACF reuse the unchanged existing 60 s candidates,
including their data-completeness rules. They are shown **before** their combined
four-check acceptance gate; the original accepted output/status is also recorded.
FarSense uses its own complex feature, selection and ACF pipeline. This first
comparison preserves their distinct algorithms, not identical preprocessing.

The reference is the received manual event count in the same `[start,start+60)`
interval. There is no marker-derived projection, subcarrier choice, peak selection
or parameter tuning. Reference sequence/quality issues still suppress error
calculation. Counts are self-reports, not a synchronized respiratory belt.

Report coverage, available-case MAE, **common-window MAE**, every failed window,
and empty-control candidate counts together. A partial set of 12 s outputs is
listed as partial; its mean is not scored as a complete-minute estimate.

Look-back lengths and bands differ: baselines 60 s/6-30 per minute; FarSense
12 s/10-37 per minute. Lower error on this small, reused pilot dataset would not
establish superiority, generalization, a fair matched-latency benchmark, or the
paper's 8 m performance. Matched-window/band sensitivity checks and the omitted
motion detector are subsequent work, not silently claimed here.
