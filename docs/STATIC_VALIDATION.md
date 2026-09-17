# Static comparison validation — 2026-09-16

## Software and source integrity

- All 40 tests passed: the previous 34 checks plus six static-analysis test cases.
- The real B002 six-session catalog completed 18 recording/feature comparisons.
- Generated 15 six-panel figures in each of PNG, PDF, and SVG (45 figure files).
- All 122 artifacts listed by the completion manifest were hash-verified after
  generation. All six source records were reverified against their stored hashes.
- All 54 full-record numerical CSV files match the earlier `b002-initial` run
  byte-for-byte. The static view adds comparisons without changing the baseline
  causal processor or its outputs.
- Representative amplitude, phase, spectrum, and periodicity PNGs were visually
  inspected for shared axes, segment exclusions, titles, and legible footnotes.

Local run: `results/b002-static-v1/`; open `index.html` for the gallery and
`SUMMARY.md` for the complete numerical tables. Outputs remain ignored by Git.
Method details and reproduction commands are in `STATIC_ANALYSIS.md` and README.

## Descriptive observations, not physiological validation

The common-interval packet rates range from approximately 26.57 to 27.98 Hz.
The largest inter-packet intervals are approximately 45–55 ms, below the existing
200 ms gap threshold. Shape transitions still reset processing: the recovered
empty recording has six transitions, and F0 Trial 2 and R90 Trial 2 have two each.
After settling exclusions, processed coverage is approximately 74.6–91.7%.
Neither packet timing nor coverage alone establishes respiratory signal quality.

The phase-difference spectrum has input-band peaks at 16 and 14 cycles/min for
the two F0 recordings, and 12 and 14 cycles/min for R90. Empty recordings also
have band peaks at 10 and 16 cycles/min, which illustrates why a plausible
frequency is insufficient evidence of breathing. The R90 Trial 1 phase spectrum
has a comparatively pronounced peak near 0.2 Hz for this fixed channel; the
second R90 repeat does not reproduce the same level of candidate stability.

The unchanged phase candidate gate accepts 15/41 and 1/21 windows for F0, and
32/41 and 0/23 for R90, versus 0/16 and 0/41 for empty. These are overlapping
window counts, not independent examples, accuracy, or validated false-positive
rates. The recovered empty count differs from the original full-span analysis
because static comparison stops at 120 s instead of using approximately 125 s.

No single fixed feature is consistently strongest across all occupied records.
Do not declare one orientation superior, tune toward a desired rate, or interpret
the plots as calibrated chest displacement. Independent synchronized respiratory
reference measurements and additional recordings are needed for validation.
