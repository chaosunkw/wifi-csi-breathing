# Joint batch archive and packet diagnostics

These commands are offline only. They do not connect to Ubuntu, change radio
settings, or edit the original Monitor transfers. Data and results remain ignored
by Git. They are not a replacement for a separate data backup.

## Archive

Prepare an explicit correction JSON with `schema_version: 1`, `batch_id`, and a
`corrections` list. Each amendment contains `session_id`, `field: activityLabel`,
`original`, `corrected: Empty`, `reason`, and `confirmed_by`. Use an empty list
when all labels are already correct. A participant label alone does not imply an
empty-room recording. Session identifiers must match the verified receipts.

```bash
.venv/bin/python -m wifi_csi_breathing archive-joint \
  --source-root /absolute/path/to/wifi-csi-monitor/data/transfers \
  --batch B004 --corrections data/B004-label-corrections.json \
  --output data/archives/B004
```

The importer requires exactly two distinct trials for each condition: Empty,
F0, and R90. It copies all nine capture files and `transfer.json` per session,
checking their bytes and SHA-256 hashes before and after copying. Original
metadata and the receipt's historical paths are retained unchanged. Corrected
analysis labels are stored separately in `archive.json`, with the explicit
`label-corrections.json` amendment. No capture hashes are rewritten.

`archive.json` is written last, only on completion. Existing destinations are
refused. An interrupted run may leave a partial directory without this manifest;
do not treat it as a complete archive. The completed copy can be analyzed without
access to the original Monitor directory.

## Diagnostic figures

```bash
.venv/bin/python -m wifi_csi_breathing diagnose-joint \
  --archive data/archives/B004 --output results/b004-shape-diagnostics-v1
```

Open the output `index.html`. Each of the three fixed features has a six-session
overview and six enlarged individual figures, in PNG/PDF/SVG. Use `--mode
amplitude` to generate just one feature. Each session has five aligned rows:

1. Effective 1x3 CSI packet feature samples.
2. Effective 2x3 samples.
3. Effective 3x3 samples.
4. Accepted manual button events.
5. Transmit-stream count observed at each packet.

All features use stream 1, RX 1, reference RX 2 and subcarrier group 15. The
features are scaled CSI amplitude, log amplitude ratio, and wrapped inter-RX
phase difference. Axis limits are shared across shapes and sessions for each
feature. This does not make their effective channels physically interchangeable.
An effective transmit-stream index is not a guaranteed physical TX antenna.

Times are `(host_monotonic_ns - session_t0_ns) / 1e9`, with integer subtraction
before conversion. Only samples in the half-open `[T0, T1)` recording window are
plotted. Different sessions share relative axis limits, not an absolute start
time or breathing phase. The common host-receipt axis retains wireless, buffering
and human-response timing uncertainty; it is not hardware synchronization.

Dots are never connected across packets or shape changes. No interpolation,
filtering, normalization, best-channel selection, rate estimation, or automatic
causal interpretation is performed. NRX other than 3 is omitted from feature rows
and marked as state 0. The existing strict fixed-3x3 processor remains unchanged.
Manual events are approximate references, not physiological ground truth.

Per-session sample JSON and `diagnostics-report.json` retain numerical values,
counts, gaps, timing, archive hashes, implementation hashes and artifact hashes.
The report is written last after archive verification. Existing output folders
are refused. An output without the completion report is incomplete.

## Validation

`tests/test_batch_diagnostics.py` covers byte-preserving copies, explicit label
amendments, malformed corrections, corruption detection, incomplete-run behavior,
shape-local features, preservation of the old 3x3 policy, aligned marker ticks,
separate scatter panels, and synthetic end-to-end exports. Synthetic tests do not
establish breathing-detection accuracy.
