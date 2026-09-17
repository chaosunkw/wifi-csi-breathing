# Offline CSI and manual Marker comparison

The `joint` command reads one completed or stopped, verified nine-file transfer
from `wifi-csi-monitor`. It does not contact Ubuntu, open a serial port, or start
acquisition. Existing `static`, `analyze`, catalogs and the desktop replay GUI
retain their original behavior. This first joint-data view is a separate CLI
command with a local HTML gallery, not an extension of the replay GUI.

## Run

From this project's directory with the existing dependencies installed:

```bash
.venv/bin/python -m wifi_csi_breathing joint \
  --input /absolute/path/to/transferred-session \
  --output results/joint-comparison-v1
```

Open `results/joint-comparison-v1/index.html` in a browser. Use a new output
directory on every run. Optional `--mode amplitude`, `--mode log_ratio`, or
`--mode phase_difference` selects a feature (repeat the option to select several).
The default is all three; default fixed channels are TX 1, RX 1, reference RX 2,
subcarrier group 15. These match the existing analysis baseline; no search for a
channel that agrees with button presses is performed.

The source is the **individual transferred session directory**, containing
`transfer.json` and these nine source files:

- `csi.dat`, `metadata.json`, `ping.log`, `receiver.log`
- `csi-times.csv`, `timing.json`
- `marker-records.jsonl`, `marker-events.csv`, `marker-status.json`

The transfer receipt must list exactly these files as verified. All nine sizes
and SHA-256 hashes are checked before loading and after rendering, and the
receipt itself must remain unchanged. Symbolic-link source files are refused.
Missing sidecars are an error, not permission to invent an alignment. A legacy
four-file recording remains usable by the existing commands but cannot be used
as a joint recording. Failed/unfinished sessions are deliberately refused by
this initial comparison tool. Accepted input sizes are bounded to 256 MiB per
file, 4 MiB per JSON document, and 250,000 CSV/journal rows.

## Time and integrity rules

Both streams use exactly:

```text
relative_seconds = (host_monotonic_ns - t0_ns) / 1e9
formal window   = t0_ns <= host_monotonic_ns < t1_ns
```

Nanoseconds are parsed as integers and subtracted before conversion to seconds.
`t0_ns` and `t1_ns` come from `timing.json`; identity, clock type, alignment mode
and final capture status are checked. The chipset's `timestamp_low`, the sender's
elapsed milliseconds, UTC and Marker process startup time are **not** substituted
for the common axis. A first packet at 0.013 s stays at 0.013 s, not zero. There
is no cross-correlation alignment, fitted lag, or drift correction. Nanosecond
storage does not imply nanosecond physical accuracy.

Every CSI index row is checked against its binary record: sequential index,
byte offset, framed length, record type and chipset timestamp. An incomplete
record, extra data, missing index row or backward host timestamp fails explicitly.
Non-CSI records remain included in index accounting but are not feature samples.
Duplicate CSI host times are marked unavailable and reset the causal processor;
they are not jittered into invented sample times. Duplicate accepted Marker
identities and backward Marker timestamps are refused. The accepted event CSV is
cross-checked against the raw hardware journal and final accepted/rejected/missing
counters. Empty event streams are valid and produce a clearly labeled empty panel.

Preparation/tail samples remain in their original files. Only formal-window CSI
and Marker events enter the figure. `markers.csv` exports **all** accepted events,
including relative time and an `in_window` flag, so an exclusion is auditable.
The source files are never trimmed, relabeled or modified.

## Figures and exports

Each selected feature produces a two-panel PNG/PDF/SVG:

- Above: packet-level CSI feature versus host receipt time. Scatter points avoid
  joining across missing data or dimension changes. Amplitude is the existing
  Intel-scaled relative amplitude; log ratio is unitless; phase difference is
  wrapped radians. These are not bandpass-filtered breathing waveforms.
- Below: accepted manual inhalation-onset events at their host receipt times.
  Sequence numbers are shown for up to 40 events; all ticks are plotted for
  larger counts. Sequence numbers may restart after a sender reboot; boot identity
  is retained in the exported event table.
- Both panels share the exact same 0-to-window-end horizontal limits.

`MODE-observations.csv` preserves formal-window packet features and quality/reset
reasons. The fixed 3x3 spatial mapping policy is unchanged: other dimensions are
unavailable rather than silently spliced into a different channel mapping.
`MODE-waveform.csv` separately exports the existing causal processor's smoothed
input and bandpass output on the common axis, with segment and availability
times. Filter warm-up and resets still apply; the initial 10 s of each segment
are provisional settling, and these filtered values are not the figure's points.
Marker times and experiment labels never enter feature extraction or filtering.

`joint-report.json` is written **last** as the completion marker. It includes
source/receipt hashes, common timing, inclusion/exclusion counts, original Marker
status, channel/processing settings, dependency versions, implementation hashes
and artifact hashes. A partial directory without this report is not a completed
run; use a new output directory after an interruption. The local `results/` folder
is ignored by Git; plots and reports may contain sensitive metadata and must not
be published without review. Keep independent backups of recordings.

## Interpretation

This is a **host-receipt reference**, not physical synchronization or a reference
respiration sensor. Human anticipation/reaction, button movement, radio/USB
buffering and host scheduling introduce unmeasured delay and possible motion
artifacts. A tick is a self-reported inhalation onset, not proof that a CSI peak
is a breath. Do not shift events until they visually match or use these plots
to claim respiratory accuracy. The purpose is a transparent exploratory reference
for later algorithm development and better controlled experiments.

## Validation

Synthetic tests cover integer precision beyond floating-point epoch resolution,
common T0 versus chipset resets, half-open boundaries, nonzero first-packet time,
shared plot axes/tick coordinates, invalid receipts/indexes/journals, empty markers,
duplicate host times, dimension resets, non-overwriting outputs, source integrity
and completion markers after failure. Run the complete suite with:

```bash
QT_QPA_PLATFORM=offscreen .venv/bin/python -m unittest discover -s tests -v
```

A local hardware transfer was also read without modification: 1,684 indexed
records, 1,683 CSI packets in the formal window, 11 accepted events in that window,
zero rejected events and zero observed missing sequences. One indexed record was
outside the window. Nine in-window packets were unavailable under the fixed
feature policy. The recorded window was approximately 60.118 s; it was not forced
to 60 s. These checks validate data handling, not physiological agreement.

Validation on 2026-09-17: all 54 tests passed (40 existing checks plus 14 joint
tests). The final real-data run produced 17 hashed artifacts plus its completion
report; every artifact hash and all original source hashes were rechecked.
Amplitude and phase-difference figures were visually inspected for shared axes,
all 11 event ticks, readable labels, and explicit unavailable-feature counts.
