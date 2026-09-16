# Radar-to-CSI reuse audit

## Sources and scope

Reference repository: `iwr6843aop-vital-signs`, local sibling checkout, commit
`66929da86d3b47032bfac48b714f2730b2f30965`.

The complete `docs/csi-mac-handoff.md` was read first. Relevant Python host files
read in full: `live.py`, `gui.py`, `analysis.py`, `plotting.py`, `protocol.py`,
`session.py`, `ports.py`, and `cli.py` under `host/src/iwr6843aop_vitals/`.
The host README, package configuration, `test_live.py`, `test_gui.py`, and
`test_analysis.py` were also reviewed, together with `docs/analysis.md`,
`docs/live-demo.md`, `docs/baseline-validation.md`, and
`configs/vital-signs-20fps-front.cfg`.

Embedded review covered the output protocol header, the complete vital-sign
inter-frame processing function and its initialization in
`ccs-workspace/vital_signs_68xx_dss/dss/dss_data_path.c`, and the full
`dss_vitalSignsDemo_utilsFunc.c` file and its license. Unrelated EDMA/driver,
board flashing/build internals, generated artifacts, and unrelated protocol
tests were not exhaustively reviewed or ported.

## Central finding

The radar Python GUI primarily displays respiratory/heart-rate results already
computed by TI firmware. Its recorded UART stream is not raw radar ADC data,
and its host application is not a portable Python breathing estimator.
Therefore copying that GUI alone would not create a CSI breathing detector.

## Reuse decisions

| Source | Decision | Destination / reason |
| --- | --- | --- |
| `analysis.py`: `SeriesStatistics`, `summarize_values` | Reused with attribution | `stats.py`; generic finite-value descriptive statistics, not physiology |
| `live.py`: worker, queue, cancellation | Architectural adaptation | `replay.py`; bounded display queue, all packets processed, interruptible waits |
| `live.py`: radar replay pacing | Reimplemented for CSI | Actual hardware timestamp deltas, monotonic deadlines, pause-aware clock; no 20 FPS assumption |
| `gui.py`: rolling traces and current-value lifecycle | Architectural adaptation | `gui.py`; Qt widgets replace Tk; clear current rate on pause, EOF, stop, error, or stale input |
| `analysis.py` / `plotting.py`: export and diagnostic separation | Reimplemented | CSV/JSON provenance plus Qt plots; no radar millimeter units or firmware-rate fields |
| Radar host tests | Testing patterns reused | EOF/cancel/error/stale-state checks; new CSI integrity and causal-prefix tests |
| `protocol.py` | Not reused | TI UART magic word/TLV schema is unrelated to Intel 5300 packed CSI |
| `ports.py`, hardware `session.py`, radar CLI live commands | Not reused | Serial discovery, configuration ports, and firmware capture are not CSI input |
| `dss_data_path.c` vital-sign stages | Conceptual reference only | Independent SciPy/NumPy bandpass, spectrum, and autocorrelation on CSI features |
| `dss_vitalSignsDemo_utilsFunc.c` | Excluded from copying/adaptation | Its source license restricts use and derivatives to TI devices |
| TI filter coefficients, range-bin selection, wavelength-to-mm conversion | Not reused | Different observable, sample clock, and propagation geometry |

The radar code's parameter naming and buffer assumptions require careful review
even apart from modality differences. For example, the embedded breath/heart
peak-count conversion references different-named buffer lengths. The new code
does not port those calculations or assume firmware defaults are interchangeable.

## What comes from the CSI collector instead

`src/wifi_csi_breathing/vendor/intel5300.py` is an unchanged copy of the owner's
`wifi-csi-monitor/scripts/python/intel5300.py` from the checkout used to create
this project (`64fd63d` at intake). Its existing decoder tests were copied with
only the import path adjusted. They cover packed values, antenna permutation,
dimensions, scaling, timing, and invalid input.

This decoder traces back to the MIT-licensed Linux 802.11n CSI Tool utilities;
the existing full notice is preserved in `THIRD_PARTY_NOTICES.md`. New data
intake adds strict archive-hash checks and refuses partial CSI records rather
than silently treating a preview's tolerated tail as analysis-ready data.

## Independent CSI work in this first version

- Fixed-channel features: scaled amplitude, log amplitude ratio, and inter-RX
  phase difference. The latter two are exploratory variants, not a claim to
  reproduce a named published system.
- Relative hardware time with 32-bit rollover handling; no conversion to
  synchronized radar/UTC time and no fixed nominal packet-rate timing.
- Segmentation at unsupported shapes, dimension changes, invalid data, duplicate
  timestamps, and gaps. No zero fill across missing data.
- One shared causal processor for batch analysis and replay, with explicit
  right-endpoint availability time for interpolation.
- Heuristic spectral/ACF agreement displayed as **periodicity candidates**, never
  as confirmed breath or presence. No learned model or label-conditioned rules.

## Deferred rather than blindly transplanted

Channel/antenna/subcarrier selection, PCA, complex CSI ratios, calibrated phase
cleanup, motion rejection, validated quality thresholds, augmentation, and live
input all need CSI-specific experiments. An empty-room spectral peak is an
important negative-control finding, not a successful breathing detection.

There was no ready-to-use radar data-augmentation pipeline in the reviewed host
code. No artificial augmentation is added to this six-recording pilot. In
particular, duplicated or overlapping windows must not be presented as new
independent experimental subjects or used across train/test splits.

## Licensing and preservation

The owner explicitly requested local reuse of their host code. No repository-wide
license for that host code was found, so this scaffold does not invent one or
publish it. The TI-device-only utility source is not included. Dependency notices
remain with their installed packages. Review release licensing before a future
public repository or packaged application.

The radar repository, collector repository, and six original recordings are not
modified. No firmware, driver, Ubuntu helper, server, or capture configuration
is changed by this analysis project.
