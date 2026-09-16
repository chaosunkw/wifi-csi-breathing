# Initial local validation — 2026-09-16

## Scope

Local version 0.1.0: offline Intel 5300 processing and recorded-data GUI replay.
No live detector, Ubuntu change, new capture, ground-truth comparison, Git commit,
or publication is part of this build.

Environment: Apple Silicon macOS, Python 3.14.4, NumPy 2.5.3, SciPy 1.18.1,
PySide6 Essentials 6.11.2, Shiboken6 6.11.2, in the project's own virtual environment.

Automated checks cover 34 tests: inherited decoder/scaling fixtures; source
hashes and path boundaries; timestamps and dimension changes; flat and synthetic
periodic inputs; causal prefix invariance; offline/replay numerical equality
at different speeds; interruptible pause/stop; terminal errors; current-rate
clearing; and offscreen Qt rendering. A synthetic 0.25 Hz signal is recovered
within the expected 30-second spectral-bin resolution. This is software testing,
not a measured respiration-accuracy result.

A separate Qt offscreen integration check replayed the actual R90 Trial 1
recording at 8x using the phase-difference feature. All 3,359 observations were
processed; pause/resume, source re-verification, EOF clearing, replay again, and
stop succeeded. Screenshots of ready, running, and finished states are saved
locally under `results/gui-qa/` and the running view was visually inspected for
layout/legibility. This is not yet the owner's interactive macOS acceptance test.

## Six real recordings

Input is the existing `B002/six-session-pilot` archive in the collector project.
All four raw files in every recording are checked against the saved archive
hashes. No raw file or original metadata is rewritten. The collector's saved
audit reports zero ping loss; this is not a new remote Ubuntu verification.

| Recording | CSI packets | Hardware-time span (s) | Spatial-shape resets | Eligible 30 s windows per feature |
| --- | ---: | ---: | ---: | ---: |
| Empty, Trial 1 | 3,402 | 124.983 | 6 | 19 |
| Empty, Trial 3 | 3,193 | 120.028 | 0 | 41 |
| F0, Trial 1 | 3,311 | 120.076 | 0 | 41 |
| F0, Trial 2 | 3,194 | 120.039 | 2 | 21 |
| R90, Trial 1 | 3,359 | 120.030 | 0 | 41 |
| R90, Trial 2 | 3,189 | 119.993 | 2 | 23 |

The first empty recording was recovered for diagnostic use. Its original
`failed` status and approximately 125-second sample span are preserved; it is
not relabeled as a normal 120-second acquisition or silently trimmed.
Dimension-change resets are conservative and materially reduce eligible windows.

## Fixed-parameter exploratory results

Run: `results/b002-initial/report.json`, all six sessions, TX 1 / RX 1 /
reference RX 2 / subcarrier group 15, with the defaults in `docs/METHODS.md`.
No channel search or scene-specific tuning was performed. Counts below are
periodic-candidate windows / eligible windows, **not accuracy or independent
trial counts**. Adjacent windows overlap strongly.

| Recording | Amplitude | Log amplitude ratio | Phase difference |
| --- | ---: | ---: | ---: |
| Empty, Trial 1 | 0 / 19 | 0 / 19 | 0 / 19 |
| Empty, Trial 3 | 0 / 41 | 0 / 41 | 0 / 41 |
| F0, Trial 1 | 0 / 41 | 0 / 41 | 15 / 41 |
| F0, Trial 2 | 9 / 21 | 15 / 21 | 1 / 21 |
| R90, Trial 1 | 9 / 41 | 8 / 41 | 32 / 41 |
| R90, Trial 2 | 0 / 23 | 11 / 23 | 0 / 23 |

Nonempty candidate series have median frequencies between 10 and 16 cycles/min.
Those values are plausible low-frequency periodicity, but no independent breath
reference was captured. The current candidate gates might select motion or
environmental oscillation and might miss genuine respiration.

The empty results are encouraging negative-control observations for **this one
fixed configuration**. They do not establish an absence detector or a zero
false-positive rate. Likewise, the mixed F0/R90 results do not prove one body
orientation is generally superior. These are two repeats per condition, one
participant for the breathing conditions, and no held-out validation set.

## Next research decision

The scaffold is ready for the owner to inspect and replay the recordings. The
next algorithmic step should compare CSI features and motion sensitivity with
independent reference measurements, not immediately connect this unvalidated
candidate score to a live breathing dashboard. Preserve these fixed-configuration
results before future channel-selection or threshold changes.
