# VitalCSI: third-batch adapted reproduction

Source: Tom Michaelis et al., *VitalCSI: Contactless Respiratory Rate Estimation
Using Consumer-Grade Wi-Fi Channel State Information*, Sensors 26, 225 (2026),
DOI <https://doi.org/10.3390/s26010225>. The final 27-page article was read from
the local Zotero library. Methods on pages 11-18 and the displayed equations on
pages 14, 16, 17 and 18 were checked against the rendered PDF. No author source
implementation was verified. The code here was independently written.

This is **VitalCSI-adapted-v1**, an implementation of the published architecture
and stated equations with explicit choices where details are missing. It is not
a replication of the authors' calibrated system or reported accuracy. All
parameters below were fixed using the paper and synthetic checks before running
the existing real recordings. Manual events and recording labels never enter
selection, onset detection, quality scoring or fusion.

## Mathematical correspondence

The core is `src/wifi_csi_breathing/baselines/vitalcsi.py`.

| Paper operation | Function | Implemented convention |
| --- | --- | --- |
| CSI amplitudes | `amplitudes` | Absolute complex value, stream 1, fixed RX1, all 30 measured groups. No log ratio, RSSI scaling, packet normalization or group standardization. |
| Windowed PCA | `select_components` | Center each group within the 30 s window; covariance PCA via SVD. Components are ordered by variance. Fix sign using the largest-magnitude loading, without marker alignment. |
| Eq. (3) variance gate | `select_components` | Follow the prose's cumulative-variance elbow: retain the prefix before the first component contributing less than 1%, with at least one nonzero-variance component. See ambiguity below. |
| Eq. (4) SNR index | `power_spectrum`, `select_components` | Power within 5-35 cycles/min divided by total non-DC power of the unfiltered PC. Mean removal and Hann taper; one-sided energy weights; no zero padding. |
| Eqs. (5)-(6) CQI | `select_components` | Variance gate times SNR times a dominant-frequency >=5/min gate. Keep up to five positive-CQI components, never pad with zero-quality components. |
| Respiratory filtering | `filter_components` | Fourth-order Butterworth high-pass at 5/min followed by fourth-order low-pass at 35/min. Forward/backward evaluation inside a completed window. |
| Eq. (7) onset-count rate | `breath_onsets` | Boxed slope-sum function (BSSF) followed by adaptive onset selection; rate is 60(N-1)/(last onset minus first onset). |
| Frequency estimate | `power_spectrum`, `analyze_frame` | Hann-tapered 30 s filtered PC; strongest positive-frequency bin; no zero padding or peak interpolation. Reject a peak outside 5-35/min. Bin spacing is 2/min. |
| Eq. (8) component rate | `analyze_frame` | Arithmetic mean of FFT and BSSF rates; both required. |
| Eqs. (9)-(10) SQI | `analyze_frame` | SNR times binary agreement (absolute difference <=4/min) times spectral purity. Zero-SQI components are excluded from fusion. |
| Eqs. (11)-(12) SPI | `spectral_purity` | Filtered-PC spectral moments: m2 squared divided by m0 times m4. Full nonnegative-frequency power, with one-sided energy weighting. |
| Kalman source filtering | `KalmanFusion` | Separate scalar random-walk filters for eligible variance-ordered PCs. Parameters and component-identity guard below are adaptation choices. |
| Eqs. (13)-(14) fusion | `KalmanFusion` | Fusion variance = (floored absolute innovation / SQI) squared; weights proportional to inverse variance. This is algebraically equivalent to normalized products of the other source variances. |

## Ambiguities and explicit choices

The printed variance formula and prose disagree: the formula appears to impose
an individual-component threshold derived from a cumulative elbow. The example
retains 27 PCs explaining 97.2% cumulatively, which does not specify a consistent
individual-fraction comparison. This adaptation uses the prose interpretation
above. Centered covariance PCA rather than correlation PCA is also a convention.

The overview says 6-33/min while the detailed filter section says 5-35/min. We
use the detailed range throughout. The paper mentions an eighth-order IIR
approach but does not uniquely specify family, order allocation or phase handling.
Here each one-pass high/low filter has four poles; forward/backward application
doubles the effective order. The resulting waveform is retrospective within its
window, not a sample-by-sample causal filter output.

The BSSF section cites Zong (2003) and Villarroel's thesis (2017), without enough
numeric settings to reconstruct their exact implementation. Box formation and
threshold updates also follow the same author's published description in
[Villarroel et al. (2020)](https://www.nature.com/articles/s41598-020-75152-z):
sum positive slopes, replace each complete positive pulse by a constant-height
box, then adapt a detection threshold. For respiration, this implementation uses
a 0.4 s slope-sum window, initializes the threshold to the median complete-box
height within the completed 30 s frame, moves it 10% toward an above-threshold
height, and reduces it by 10% for a below-threshold event. Onsets must be at least
60/35 s apart. Boundary-clipped pulses are excluded. These respiratory timing
choices are not copied from the original PPG detector. Onset polarity is a PCA
convention, not an identified physiological inhalation onset.

The SPI formula is used as printed. Despite the paper's description, a flat
continuous spectrum on a finite symmetric frequency interval has SPI=5/9,
not zero. SQI is therefore not a calibrated probability of breathing or signal
quality. No undocumented correction is applied to make noise score zero.

The paper does not fully specify the Kalman state model, Q/R/P0, innovation
convention, zero-residual handling, initialization or PC identity across windows.
Our scalar state is cycles/min, random-walk Q=1 (cycles/min)^2 per second,
measurement R=4 and initial P=4. Innovation is the current component estimate
minus the predicted estimate, before updating. Its absolute value is floored at
0.5/min in the fusion variance to avoid singular weights. First observations
initialize at their measured value. A component reuses its previous state only
if its variance index and valid group set match and the absolute loading-vector
inner product is at least 0.8. Otherwise it starts a new state. Missing or
zero-quality sources are removed; missing measurements never become held
predictions. This guard is an adaptation, not a claimed original-paper feature.

## Input, time and comparison policy

The paper uses a single RX Raspberry Pi/Broadcom system with 256 groups, 80 MHz
and 50 Hz. We use stream 1, RX1 and the Intel 5300's 30 reported groups, resampled
onto 20 Hz. Packets with 1x3, 2x3 and 3x3 shapes are retained, with shape and rate
flags audited. Identical stream indices across configurations do not guarantee
an identical physical propagation channel. No cross-configuration calibration
is added, and no amplitude normalization is introduced to conceal this issue.

Interpolation follows actual timestamps with a maximum 0.5 s endpoint gap.
Grid points bridging more than 0.2 s are counted, with at most 5% per frame.
There is no extrapolation or time compression. Duplicate timestamps invalidate
the frame; invalid values exclude their group for any window containing them.
Unknown/non-CSI records remain explicit missing observations. Each result is
available only after its window and any required interpolation endpoint arrive.
Tests compare completed prefixes with the same frames from the full recording.

Each predefined 60 s interval contains 31 overlapping 30 s frames at 1 s hops.
The complete-minute value is the arithmetic mean of all 31 fused estimates,
only when all exist. Available-only and pre-SQI component means are exported
separately and never substituted. Kalman history starts afresh for each compared
minute. Overlapping windows are not independent trials. This minute aggregation
is a comparison convention, not an original-paper evaluation protocol.

The previous four methods rerun unchanged: LogRatio-FFT and LogRatio-ACF use
their 60 s signal; FarSense-core averages five 12 s blocks; WiResP-adapted averages
46 15 s frame candidates. FFT/ACF and WiResP values shown in the overview precede
their final quality gates, while SQI is intrinsic to VitalCSI fusion. Consequently
this is an exploratory end-to-end comparison, not an estimator-only ablation or
a fair detector-acceptance benchmark. Availability, partial estimates and raw
per-component FFT/BSSF estimates are visible alongside error on common windows.

Manual button counts are a self-reported reference, not an independently measured
physiological waveform. Empty controls have no assigned zero breathing-rate
truth. Periodic estimates, even with high SQI, do not establish person presence.
The current pilots have already informed development of the earlier methods;
new recordings are needed for held-out claims. No published accuracy or universal
method ranking is claimed from this hardware adaptation.

## Run and inspect

```bash
.venv/bin/python -m wifi_csi_breathing compare-vitalcsi \
  --manifest /absolute/path/to/comparison-manifest.json \
  --output results/vitalcsi-comparison-v1
```

The manifest is the same schema as `compare-farsense`/`compare-wiresp`. The command
does not start Monitor, connect to Ubuntu, modify source recordings or overwrite
an existing output directory. It reruns the four methods under `reference-methods/`
and adds the third-batch results. Source files, receipts, implementation, manifest
and artifacts are hashed; the completion report is written after verification.

The report contains a five-method comparison, availability/MAE tables, separate
empty controls, and per-minute diagnostics. The first four diagnostic panels
always show the first frame, never the best-looking frame: raw amplitudes,
filtered PCs, FFT spectra, and BSSF onsets/thresholds. The other two panels show
component estimates, SQIs and Kalman fusion across all frames. CSV/JSON files
contain each decision and weight; NPZ files contain amplitudes, PCA loadings,
scores, spectra, BSSF arrays and timestamps. PCA traces are standardized only for
plotting, not before estimation.

Real recordings, local manifests/configuration, PDFs and generated results remain
outside Git. Changes to numerical settings should receive a named version and
new output directory, rather than quietly fitting these same pilot references.
