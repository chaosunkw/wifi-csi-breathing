# Research-preview baseline — 2026-09-16

## Purpose

Preserve version 0.1.0 before developing unified-scale static analysis of the
six-session pilot. This is a source-control checkpoint, not a validated detector,
a clinical claim, or a formal tagged release. No new signal-processing or GUI
behavior is introduced by this checkpoint.

## Included capabilities

- Read-only Intel 5300 decoding, timestamp handling, and source-hash verification.
- Fixed-channel amplitude, log amplitude ratio, and inter-RX phase difference.
- Causal preprocessing and bandpass filtering, trailing-window spectral and
  autocorrelation diagnostics, and explicitly unvalidated periodic candidates.
- Offline CSV/JSON exports and paced Qt replay with pause/resume and stop.
- Source code, synthetic tests, dependency configuration, launcher, method notes,
  reuse audit, initial validation summary, and third-party attribution.

The replay viewer is retained as a diagnostic tool. Its rolling, independently
scaled plots are not a substitute for shared-scale comparisons across recordings.
Irregular curves are not proof of failed decoding, and band-limited oscillation
alone is not proof of respiration. EOF clears the current candidate while
retaining historical curves; an empty current estimate at EOF is not a summary
of every earlier analysis window.

## Verification at the checkpoint

All 34 automated tests passed on the existing Apple Silicon macOS environment
using Python 3.14.4 and the versions in `requirements-lock.txt`:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
```

The owner has also launched the desktop viewer and replayed empty and occupied
recordings. This confirms basic interactive operation, not physiological
accuracy. Earlier real-data processing and integration checks are documented
in [initial validation](INITIAL_VALIDATION.md); they were not repeated as new
experiments for this source-control checkpoint.

## Data and reproducibility boundaries

Raw recordings, local absolute-path catalogs, generated reports, virtual
environments, caches, and credentials are excluded from version control.
Original recordings remain in the collector archive; this checkpoint does not
move, modify, or upload them. A clone needs its own environment and a local
catalog rebuilt from the verified archive, as described in the README.

The initial validation document retains aggregate pilot diagnostics, not raw
recordings or participant identifiers. Source hashes and run configurations
remain in the locally retained catalog and analysis outputs. Git backs up the
source, not those excluded data or results; retain separate backups of them.

## Next milestone (not implemented here)

Continue in Python and add publication-style static comparisons: three conditions
by two repeats, consistent axes within each feature, spectra, and quality tables.
Separate original-scale comparisons from normalized morphology views. Mark gaps
and invalid segments rather than connecting them. Preserve the first empty
recording's original failed status and approximately 125-second span even when
using a common 0–120 second comparison interval.

Keep acquisition separate and do not add live input, automatic channel selection,
PCA, VMD, or new physiological claims as part of this baseline checkpoint.
