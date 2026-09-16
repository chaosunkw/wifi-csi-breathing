# Wi-Fi CSI Breathing

Independent **offline analysis and paced replay** of Intel 5300 CSI recordings.
Version 0.1.0 is a research-preview baseline, not a validated breathing detector.
See [baseline notes](docs/BASELINE.md) for the preserved scope and next milestone.

This project does not start capture, initialize Ubuntu hardware, connect to SSH,
or change `wifi-csi-monitor`. It reads the six-session B002 archive through a
hash-verified local catalog. Original CSI, logs, and metadata remain untouched.
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

There is no live input, trained classifier, automatic best-channel search,
data augmentation, ground-truth accuracy claim, or millimeter displacement output.
These are deliberately outside this initial scaffold.

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

## Recreate the environment / local catalog

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

1. Add unified-scale static analysis of the six sessions: separate amplitude,
   log-ratio, and phase-difference panels; filtered waveforms; spectra; and quality
   summaries. Use publication-style figures with shared axes within each feature,
   explicit invalid intervals, and separate raw-scale and normalized comparisons.
   Preserve this replay baseline as an auxiliary tool. These figures are the next
   milestone, not a feature of this baseline. Do not infer occupancy or breathing
   from a spectral peak alone or tune the analysis to an assumed target rate.
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
