# Fixed-configuration exploration

The default `joint` analysis resets on every reported spatial-dimension change.
Occasional packets with another stream count can therefore prevent a full
analysis window even when most packets use 3x3. `joint-fixed` is a separate,
explicit research option, not a change to the default policy or a radio driver.

```bash
.venv/bin/python -m wifi_csi_breathing joint-fixed \
  --input /absolute/path/to/verified-transferred-session \
  --rate-flags 275 \
  --output results/fixed-configuration-v1
```

The rate code is an explicit observed `rate_n_flags` integer, not Mbps or capture
frequency. Choose it from configuration diagnostics independently of apparent
breathing or manual-marker agreement; record this choice in the protocol. There
is no automatic dominant-rate selection, channel search or fitted marker lag.
The example is not a universal value for all routers or recordings.

## What changes

- Only packets with reported `(Ntx, Nrx, rate) == (3, 3, selected_rate)` provide
  feature values. Other shapes and rates are missing observations, not zeroes,
  and do not enter the filter. The selection CSV accounts for every formal-window
  indexed record with the original index and common-host timestamp.
- The existing causal Processor is unchanged. If matching samples resume within
  0.2 s, its existing interpolation may bridge the gap after receiving the right
  endpoint. The export's `available_at` records that delay. It never compresses
  the time axis or emits extrapolated samples through a trailing outage.
- More than 0.2 s between retained samples resets the history. Duplicate times,
  non-CSI records and invalid matching features are explicit hard barriers.
- The 10 s warm-up, 30 s window and periodicity criteria are unchanged. Warm-up
  is shaded and hidden in the filtered plot. Every window, including inconclusive
  ones, is exported. The original strict result is reported for comparison.

This assumes a comparable effective channel when the identical reported
configuration returns. Recorded flags cannot verify physical TX antenna choices,
spatial mapping or all hardware effects. This assumption requires separate
validation; do not describe gap handling as a hardware fix or a proven detector.
The full raw timeline and strict baseline remain available for comparison.

## Exports and interpretation

`index.html` links PNG/PDF/SVG packet and causal-filter plots. The three fixed
features use stream 1, RX 1, reference RX 2, group 15 unless explicitly changed.
There is no respiratory phase calibration; the causal filter has delay. Its
peak times must not be equated with inhalation onset. XBee manual button events
are self-reports on a host-receipt clock, not physiological ground truth.

Each mode exports a packet selection audit, retained observations, causal
waveform (including provisional warm-up) and all estimate windows. A candidate
is exploratory periodicity, not an accuracy claim. Source and implementation
hashes are checked before and after export. `fixed-configuration-report.json`
is written last; an incomplete directory without it is not a completed run.
Outputs never overwrite previous runs or enter the raw recording directory.

## Hardware scope

The Intel CSI Tool documents `Ntx` as transmitted space/time streams and `Nrx`
as receive antennas. Its debugfs fixed-rate control sets that Intel device's
**transmit** rate; it does not command a separate AP's downlink rate. See the
[official FAQ](https://dhalperi.github.io/linux-80211n-csitool/faq.html).
This analysis project never runs acquisition, connects to Ubuntu or reconfigures
the AP. Fixed channel/width alone must not be described as fixed MCS/NSS.

## Validation

Synthetic tests cover short and long interruptions, exact rate/shape selection,
causal prefix invariance, duplicate timestamps, invalid matching features,
trailing outages, strict-policy compatibility, export integrity and CLI errors.
Run the existing full suite before applying the option to experimental data.
