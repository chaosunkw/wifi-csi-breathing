# Third-party notices

## Linux 802.11n CSI Tool

The binary format decoding and signal scaling in
`src/wifi_csi_breathing/vendor/intel5300.py`
are adapted from `read_bfee.c`, `read_bf_file.m`, `get_scaled_csi.m`, and
`get_total_rss.m` in the Linux 802.11n CSI Tool supplementary utilities.

Source: https://github.com/dhalperi/linux-80211n-csitool-supplementary

The authors identify their own tool code as MIT-licensed:
https://dhalperi.github.io/linux-80211n-csitool/faq.html

Copyright (c) 2008-2011 Daniel Halperin

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
of the Software, and to permit persons to whom the Software is furnished to do
so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Owner's radar host code

`src/wifi_csi_breathing/stats.py` reuses `SeriesStatistics` and
`summarize_values` from `host/src/iwr6843aop_vitals/analysis.py` in the owner's
`iwr6843aop-vital-signs` repository at commit
`66929da86d3b47032bfac48b714f2730b2f30965`. Reuse was requested by the owner for
this local project. No repository-wide license was found there. This notice
does not grant a new public redistribution license for the owner's code.
Review licensing before a future public release.

Worker/queue separation, rolling plots, terminal-state clearing, and replay
tests are architectural adaptations of that project's Python host application.
The Qt implementation and CSI signal-processing pipeline are new code.

No TI embedded processing source or TI filter coefficients were copied.
In particular, `dss_vitalSignsDemo_utilsFunc.c` has a TI-device-only license and
is excluded from this Intel Wi-Fi project. See `docs/RADAR_REUSE_AUDIT.md`.

## Runtime dependencies

NumPy, SciPy, Matplotlib, PySide6 Essentials, and Shiboken6 are installed as separate
dependencies, not vendored in this source tree. Their installed distributions
retain their own license notices. PySide6/Qt and Shiboken distribution obligations
must be reviewed if a packaged application is distributed in the future. This
prototype is an unbundled local Python application; no public package is published.
