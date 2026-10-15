# Wi-Fi CSI Breathing

The third batch adds an independent [VitalCSI adaptation](docs/VITALCSI_REPRODUCTION.md).
`compare-vitalcsi` compares amplitude PCA, FFT/breath counting and quality-weighted
Kalman fusion with the four existing methods on the same predefined minutes.
PCA scores, spectra, onset decisions, component quality and fusion weights are
exported. The paper's ambiguous variance gate and unspecified BSSF/filter/Kalman
settings are documented as adaptation choices, not an exact author replication.

For the second reproduction batch, see [WiResP adapted reproduction](docs/WIRESP_REPRODUCTION.md).
The `compare-wiresp` command compares its amplitude/ACF enhancement pipeline with
the unchanged FarSense core and LogRatio FFT/ACF candidates on predefined minutes.
Intermediate images and decisions are exported. Unspecified paper parameters,
the adapted continuity checker and the uncalibrated motion gate are documented;
this is not a replication of the original system's reported accuracy.

The first literature baseline is an independent, adapted
[FarSense core reproduction](docs/FARSENSE_REPRODUCTION.md). The `compare-farsense`
command compares its complex-ratio projection and multi-group autocorrelation
with the existing LogRatio FFT/ACF candidates on predefined minutes. It includes
per-block numerical audits and empty controls; the paper's large-motion exclusion
module is not implemented. Existing replay and analysis defaults are unchanged.

Independent **offline analysis and paced replay** of Intel 5300 CSI recordings.
Version 0.1.0 is a research-preview baseline, not a validated breathing detector.
See [baseline notes](docs/BASELINE.md) for the preserved scope and next milestone.

This project does not start capture, initialize Ubuntu hardware, connect to SSH,
or change `wifi-csi-monitor`. It reads the six-session B002 archive through a
hash-verified local catalog, and joint CSI/Marker sessions through verified
transfer receipts. Original CSI, logs, and metadata remain untouched.
The radar project is a reference, not a runtime dependency.

## What is available

- Intel 5300 decoding with antenna permutation, signal scaling, timestamps, and
  original file hashes retained in analysis provenance.
- Three fixed-channel exploratory features: scaled amplitude, log amplitude
  ratio between two RX channels, and same-TX inter-RX phase difference.
- Past-only preprocessing, a causal bandpass, and trailing-window spectral and
  autocorrelation diagnostics. Missing samples, long gaps, or spatial dimension
  changes reset the window rather than silently joining incompatible data.
- Offline CSV/JSON exports and a Qt desktop replay viewer, both using the same
  processor. Replay supports 1x/2x/4x/8x speed, pause/resume, stop, and replay again.
- Relative feature and filtered waveforms; optional **periodic candidate** rates.
  A candidate is not proof of breathing or person presence. Rates are cleared
  when paused, stopped, finished, failed, or waiting on stale replay data.
- Shared-scale static pilot comparisons, using the unchanged replay processor:
  three conditions by two repeats, publication-style PNG/PDF/SVG figures, spectra,
  quality tables, and a local HTML gallery. See [static analysis](docs/STATIC_ANALYSIS.md).
- Offline CSI / XBee manual-marker comparisons from verified joint transfers,
  with two-panel plots on the common Ubuntu host-receipt timeline. See
  [joint analysis](docs/JOINT_ANALYSIS.md); this is not physiological ground truth.

There is no live input, trained classifier, data augmentation, ground-truth
accuracy claim, or millimeter displacement output. The original replay uses
fixed feature channels; literature-comparison commands perform their documented
channel or component selection separately.

## Start on this Mac

The project is located at:

```text
/Users/sunchao/VSCodeProjects/wifi-csi-breathing
```

A dedicated `.venv` and a verified `data/catalog.json` have been prepared locally.
Double-click `run-gui.command` in Finder, or run:

```bash
cd /Users/sunchao/VSCodeProjects/wifi-csi-breathing
.venv/bin/python -m wifi_csi_breathing gui --catalog data/catalog.json
```

1. Choose a recording from the six-session list.
2. Start with `amplitude`, TX 1, RX 1, reference RX 2, subcarrier group 15. These
   are fixed exploratory defaults, not a proven optimal combination.
3. Select replay speed and click **Start replay**. File verification and decoding
   run in a worker; no source data is edited.
4. The initial continuous window needs approximately **40 recorded seconds**:
   10 seconds for settling plus 30 seconds for analysis. A gap or dimension change
   restarts this wait. Faster replay reduces wall-clock waiting only.
5. Use **Pause / Resume** or **Stop**. At EOF the historical curve remains but
   the current candidate is cleared. Click **Start replay** to replay again.
6. Compare `log_ratio` and `phase_difference` using different RX and reference RX
   channels. All values are feature units, not calibrated chest displacement.

The GUI intentionally has no capture or transfer button. Continue collecting
data with the existing `wifi-csi-monitor` application.

## Offline exports

The initial run is in the ignored local folder `results/b002-initial/`.
For a fresh run, choose a **new** output directory:

```bash
.venv/bin/python -m wifi_csi_breathing analyze \
  --catalog data/catalog.json --output results/b002-next
```

Default analysis evaluates all six recordings with all three fixed-channel
features (18 combinations). To select one feature or different channels:

```bash
.venv/bin/python -m wifi_csi_breathing analyze \
  --catalog data/catalog.json --output results/b002-phase-tx1 \
  --mode phase_difference --tx 1 --rx 1 --reference-rx 2 --subcarrier 15
```

Each combination exports:

- `observations.csv`: packet time, feature, invalid/boundary flags, packet index.
- `waveform.csv`: uniform-grid time, actual availability time, smoothed feature,
  causal filtered value, and segment ID.
- `estimates.csv`: eligible trailing-window diagnostics and candidate rate, if any.
- `summary.json`: source hashes, original metadata/status, configuration, timing,
  dimension changes, window counts, and descriptive candidate statistics.

`report.json` is written **last**, only on successful completion of the whole run.
An interrupted run may leave partial outputs but no completion report. Existing
output directories are never overwritten. Source hashes are checked before and
after analysis; no recovery status or experiment label is rewritten.

## Six-session static comparison

Install the updated dependencies in the existing virtual environment once:

```bash
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
```

Generate a new run (choose a different output folder if this one already exists):

```bash
.venv/bin/python -m wifi_csi_breathing static \
  --catalog data/catalog.json --output results/b002-static-v1
```

Open `results/b002-static-v1/index.html` in a browser. No server or Ubuntu
connection is needed. `SUMMARY.md` contains the numerical summary and caveats;
`quality.csv` and `spectra.csv` support further inspection. Fifteen six-panel
figures cover three features, each with smoothed input, filtered signal,
normalized shape, pre-bandpass spectrum, and trailing-window periodicity.
Every figure is exported as a 300 dpi PNG and vector PDF/SVG.

Defaults match the replay baseline: TX 1 / RX 1 / reference RX 2 / subcarrier
group 15. All records use the same configuration; there is no best-channel
search. The common interval is 0–120 s, but full-record numerical exports and
original status are retained in `numerical/`. The final completion marker is
`static-report.json`, written only after figures and final source verification
succeed. A nested `numerical/report.json` alone does not mean the static run
finished. See the [method and figure guide](docs/STATIC_ANALYSIS.md).
The [static validation record](docs/STATIC_VALIDATION.md) documents the test run,
source integrity checks, and cautious descriptive findings from the pilot.

## Joint CSI and manual Marker comparison

Use the individual session directory transferred by `wifi-csi-monitor`, including
its nine data files and `transfer.json` (not the old six-session catalog):

```bash
.venv/bin/python -m wifi_csi_breathing joint \
  --input /absolute/path/to/transferred-session \
  --output results/joint-comparison-v1
```

Open `results/joint-comparison-v1/index.html`. The three fixed feature views each
show CSI above and manual events below on the same axis from the stored common
T0. PNG/PDF/SVG, numerical CSVs and a hash-verified provenance report are produced.
This command is separate from the existing replay GUI and never connects to
hardware. Original recordings are read-only; existing output directories cannot
be overwritten. See [input validation, time semantics and limitations](docs/JOINT_ANALYSIS.md).

For a first quantitative comparison, use the separate [60 s Log Ratio rate comparison](docs/RATE_COMPARISON.md):

```bash
.venv/bin/python -m wifi_csi_breathing joint-rate \
  --input /absolute/path/to/transferred-session \
  --start 10 \
  --output results/rate-comparison-v1
```

This compares one predefined `[10, 70)` s window with the received manual button
count. It retains all reported rates within 3x3, reuses the causal filter and
FFT/autocorrelation checks, and reports unavailable estimates explicitly.
Pooling rates is an exploratory assumption. Manual counts are approximate
references, not independently measured physiological ground truth.

Each minute report also shows the FFT spectrum and autocorrelation, their
frequency candidates, and the four quality checks. Rejected candidates remain
visible as diagnostics. Full spectrum, correlation and detrended input CSVs
allow the displayed intermediate calculations to be inspected.

For a separate comparison that also uses 1x3 and 2x3 packets, add
`--shape-policy first-stream` and choose a new output directory. This opt-in
experiment always uses stream 1; configuration changes alone do not reset the
filter. It preserves shape/rate audit information without correcting spatial
mapping changes. The default 3x3 baseline, gap limits and estimator thresholds
are unchanged. See the [experiment and limitations](docs/RATE_COMPARISON.md#opt-in-first-stream-experiment).

To test short interruptions without another 10 s blackout, also add
`--gap-policy preserve-short`. Gaps over 0.2 s and at most 0.5 s preserve state
using audited internal interpolation; the plotted gap remains blank. Longer
gaps still reset, but settling is shown as dashed amber rather than hidden.
The original `--gap-policy reset` remains available as the default comparison.
See [gap handling](docs/RATE_COMPARISON.md#opt-in-short-gap-state-preservation).

## Recreate the environment / local catalog

For an explicit, opt-in analysis that treats packets from other reported radio
configurations as missing observations, see [fixed-configuration analysis](docs/FIXED_CONFIGURATION.md).
The `joint-fixed` command preserves the 0.2 s maximum interpolation gap and the
existing causal filter. It does not lock the router's rate, alter source files,
or change the default strict `joint` and replay behavior.

For a six-session joint batch with changing spatial dimensions, use the separate
[immutable archive and shape-separated diagnostics](docs/BATCH_DIAGNOSTICS.md).
This preserves the transfer files byte-for-byte, records confirmed label
corrections separately, and plots packet features for each shape without changing
the fixed-3x3 replay processor.

To inspect the frequency dimension, use the separate
[all-subcarrier diagnostics](docs/SUBCARRIER_DIAGNOSTICS.md). These retrospective
views show all 30 measured groups as heatmaps and 3D waterfalls, with fixed example
waveforms and manual markers. They preserve configuration changes and missing
observations and do not change the causal replay or estimate respiratory rate.

Python 3.11 or newer is declared; this initial build was tested on Apple Silicon
with Python 3.14.4. Qt is used because this Mac's Python does not provide a working
Tk runtime. A fresh environment can be created with:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
.venv/bin/python -m pip install -e .
```

`requirements-lock.txt` records the tested dependency versions; it is not a
cross-platform lock with wheel hashes. Other Python/platform combinations need
their own compatibility validation.

If no local catalog exists, build it from the existing archive's
`verification.json` (no raw files are copied):

```bash
.venv/bin/python -m wifi_csi_breathing import-archive \
  /Users/sunchao/VSCodeProjects/wifi-csi-monitor/data/experiments/B002/six-session-pilot \
  --output data/catalog.json
```

The importer currently targets that verified archive schema, not arbitrary CSI
folders. A catalog contains local absolute paths and source hashes, so `data/`
is excluded from Git tracking. Moving the archive requires a new catalog.
Cloning this repository does not download the recordings, generated results, or
the machine-specific catalog. The archive must be available locally before import.

## Tests

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
```

Tests cover decoder fixtures, integrity checks, timestamp rollover, changed
dimensions, causal prefix invariance, synthetic periodic signals, flat data,
pause/stop/EOF/error handling, and offscreen Qt rendering. They do not establish
physiological validity on real people.

## Research boundaries and next steps

1. Inspect the shared-scale six-session figures and quality tables before adding
   new estimators. Preserve replay as an auxiliary tool. Do not infer occupancy
   or breathing from a spectral peak alone or tune to an assumed target rate.
2. Add synchronized independent breathing reference measurements and more
   empty/motion/posture sessions; define evaluation at the recording level.
3. Develop channel selection and motion/quality rejection on development data;
   evaluate on separate recordings and participants. Do not split overlapping
   windows from one session between training and testing.
4. Only after offline validation, add a live source adapter feeding the same
   `Observation` interface. Keep acquisition responsibility in the collector.

See [reuse audit](docs/RADAR_REUSE_AUDIT.md), [processing details](docs/METHODS.md),
[initial validation](docs/INITIAL_VALIDATION.md), and
[third-party notices](THIRD_PARTY_NOTICES.md).

Environment setup does not create Git commits or publish anything. The baseline
source is maintained separately from the collector and radar repositories.
Do not commit raw recordings, participant identifiers, local catalogs, generated
results, credentials, or the virtual environment. Choose/review project licensing
before a future public release; existing third-party notices remain applicable.
