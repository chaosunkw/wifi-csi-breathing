"""Inspect the exact predefined-minute estimator without changing its decision."""
from pathlib import Path
import html
import math

import numpy as np

from .analysis import write_csv, write_json
from .processing import inspect_window
from .subcarriers import new_figure, save_figure


SPECTRUM_FIELDS = ('frequency_hz', 'rate_per_min', 'filtered_power', 'pre_bandpass_power',
                   'normalized_filtered_power', 'normalized_pre_bandpass_power',
                   'in_search_band', 'in_concentration_bins', 'selected_fft_peak')
ACF_FIELDS = ('lag_samples', 'lag_s', 'normalized_acf', 'in_search_region',
              'local_peak_in_search_region', 'selected_acf_peak')
INPUT_FIELDS = ('time', 'available_at', 'pre_bandpass_detrended', 'filtered_detrended', 'hann_weight')


def number(value, digits=3):
    return 'Not available' if value is None else f'{value:.{digits}f}'


def prepare_diagnostics(points, result, config):
    """Use only the same eligible minute; never concatenate separated segments."""
    summary = {'computed': False, 'reason': result['reason'], 'status': result['status'],
               'start_s': result['start_s'], 'end_s': result['end_s'], 'checks': [],
               'sample_count': 0, 'fft_peak_bpm': None, 'acf_peak_bpm': None,
               'acf_peak_lag_s': None, 'accepted_candidate_bpm': result['candidate_bpm'],
               'failed_checks': [], 'manual_events_used_in_estimation': False,
               'fft_preprocessing': 'Linear detrend, Hann window, squared rFFT magnitude; no zero padding',
               'acf_preprocessing': 'Linear detrend; positive-lag full correlation divided by lag-zero value; no Hann window',
               'spectrum_display_normalization': 'Each curve divided by its own sum over all positive-frequency bins',
               'spectrum_display_range_per_min': [0, 60], 'acf_display_range_s': [0, 15],
               'full_arrays_exported': False,
               'internal_gap_interpolation_included': bool(result.get('bridged_grid_samples', 0)),
               'bridged_grid_samples': result.get('bridged_grid_samples', 0)}
    if result['status'] not in ('periodic_candidate', 'inconclusive'):
        # Insufficient coverage/settling/variation has no eligible spectral rate.
        return summary, [], None
    start, end = round(result['start_s'] * 1e9), round(result['end_s'] * 1e9)
    selected = [p for p in points if start <= round(p.time * 1e9) < end]
    if len(selected) != round(config.window_seconds * config.sample_hz):
        raise ValueError('Diagnostic input does not match the eligible minute')
    if len({p.segment for p in selected}) != 1 or any(
            not math.isclose(b.time-a.time, 1/config.sample_hz, abs_tol=1e-7)
            for a, b in zip(selected, selected[1:])):
        raise ValueError('Diagnostics require one uniformly sampled segment')
    estimate, arrays = inspect_window(selected, config)
    for key in ('status', 'candidate_bpm', 'spectral_peak_bpm', 'autocorrelation_bpm',
                'autocorrelation', 'concentration', 'band_fraction'):
        if getattr(estimate, key) != result[key]:
            raise ValueError('Diagnostic computation differs from the reported estimate: '+key)
    if arrays is None:
        raise ValueError('Eligible diagnostic window unexpectedly has no variation')
    summary.update(computed=True, sample_count=len(selected), full_arrays_exported=True,
                   fft_peak_bpm=estimate.spectral_peak_bpm, acf_peak_bpm=estimate.autocorrelation_bpm,
                   acf_peak_lag_s=arrays['lag']/config.sample_hz if arrays['lag'] is not None else None,
                   fft_bin_spacing_per_min=60*config.sample_hz/len(selected),
                   checks=arrays['checks'],
                   failed_checks=[c['name'] for c in arrays['checks'] if not c['passed']])
    return summary, selected, arrays


def draw_diagnostics(summary, arrays, config):
    fig = new_figure((12, 9.0))
    fft, acf = fig.subplots(2, 1)
    fig.subplots_adjust(left=.10, right=.97, top=.81, bottom=.12, hspace=.53)
    fig.suptitle(f"FFT and autocorrelation | {summary['start_s']:g}-{summary['end_s']:g} s", fontsize=17, y=.97)
    fig.text(.10, .915, f"FFT frequency candidate: {number(summary['fft_peak_bpm'], 2)} /min | "
             f"ACF frequency candidate: {number(summary['acf_peak_bpm'], 2)} /min", fontsize=11)
    fig.text(.10, .875, f"Accepted rate candidate: {number(summary['accepted_candidate_bpm'], 2)} /min | "
             f"Status: {summary['status']}", fontsize=11)
    if arrays is None:
        for ax, title in ((fft, 'FFT spectrum'), (acf, 'Autocorrelation')):
            ax.set_title(title, loc='left')
            ax.text(.5, .5, 'Not computed for this full minute\n'+summary['reason'].replace('_', ' '),
                    transform=ax.transAxes, ha='center', va='center', fontsize=12)
            ax.set_xticks([])
            ax.set_yticks([])
    else:
        rates = arrays['frequencies'] * 60
        power = arrays['power'] / max(float(arrays['power'][1:].sum()), 1e-30)
        raw_power = arrays['raw_power'] / max(float(arrays['raw_power'][1:].sum()), 1e-30)
        fft.axvspan(config.low_hz*60, config.high_hz*60, color='#DDEEF7', alpha=.6,
                    label='Search band: 6-30 /min')
        fft.plot(rates, raw_power, color='#777777', linewidth=1, alpha=.85,
                 label='Before bandpass (smoothed/resampled)')
        fft.plot(rates, power, color='#0072B2', linewidth=1.4, label='Filtered Log Ratio')
        near = arrays['near']
        fft.bar(rates[near], power[near], width=summary['fft_bin_spacing_per_min']*.7,
                color='#D55E00', alpha=.3, label='Bins used for spectral concentration')
        peak = arrays['peak']
        fft.scatter([rates[peak]], [power[peak]], color='#B53A39', zorder=5)
        fft.axvline(rates[peak], color='#B53A39', linestyle=':', linewidth=1)
        fft.set_title('FFT: which frequencies have the most energy?', loc='left', fontsize=11)
        fft.set_xlabel('Equivalent periodic rate (cycles/min); not yet a confirmed respiratory rate')
        fft.set_ylabel('Normalized spectral power / bin')
        fft.set_xlim(0, 60)
        fft.set_ylim(bottom=0)
        fft.legend(fontsize=8, loc='upper right')
        lags = np.arange(len(arrays['ac'])) / config.sample_hz
        acf.axvspan(arrays['min_lag']/config.sample_hz, arrays['max_lag']/config.sample_hz,
                    color='#DDEEF7', alpha=.6, label='Peak search: 2-10 s')
        acf.plot(lags, arrays['ac'], color='#8E6BBE', linewidth=1.4, label='Normalized autocorrelation')
        threshold = next(c['threshold'] for c in arrays['checks'] if c['name'] == 'Autocorrelation peak')
        acf.axhline(threshold, color='#B53A39', linestyle='--', linewidth=1, label=f'Peak threshold: {threshold:g}')
        lag = arrays['lag']
        if lag is not None:
            acf.scatter([lags[lag]], [arrays['ac'][lag]], color='#B53A39', zorder=5)
            acf.axvline(lags[lag], color='#B53A39', linestyle=':', linewidth=1,
                        label=f'Selected lag: {lags[lag]:.2f} s; 60/lag = {summary["acf_peak_bpm"]:.2f} /min')
        acf.set_title('Autocorrelation: after how many seconds does the waveform repeat?', loc='left', fontsize=11)
        acf.set_xlabel('Time lag (s); this is a shift between samples, not recording time')
        acf.set_ylabel('Normalized autocorrelation')
        acf.set_xlim(0, 15)
        acf.set_ylim(-1.05, 1.05)
        acf.legend(fontsize=8, loc='upper right')
    for ax in (fft, acf):
        ax.grid(alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.10, .035, 'Exactly the same minute and numerical estimator as the reported result; manual events do not enter these calculations.\n'
             'FFT view: 0-60 /min; ACF view: 0-15 s. CSV files retain the full arrays. Spectra are normalized separately, not probabilities.', fontsize=9)
    return fig


def diagnostics_html(summary):
    if summary['computed']:
        rows = ''.join('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                      (c['name'], number(c['value']), f"{c['operator']} {c['threshold']:g}",
                       'PASS' if c['passed'] else 'FAIL'))+'</tr>' for c in summary['checks'])
        reason = ('All four heuristic checks passed.' if not summary['failed_checks'] else
                  'Failed checks: '+', '.join(summary['failed_checks'])+'.')
        table = ('<table><tr><th>Check</th><th>Measured value</th><th>Requirement</th><th>Result</th></tr>'
                 + rows + '</table>')
    else:
        reason = 'The full-minute FFT/ACF was not computed: '+summary['reason'].replace('_', ' ')+'.'
        table = ''
    return ('<h2>2. FFT spectrum and autocorrelation</h2>'
            '<p>The spectrum shows which periodic frequencies are present. The autocorrelation compares the '
            'waveform with shifted copies; a selected lag T seconds corresponds to 60/T cycles per minute. '
            'Frequency candidates remain visible even when quality checks reject the final rate.</p>'
            '<img src="estimator-diagnostics.png" alt="FFT spectrum and normalized autocorrelation for the same minute">'
            '<p><a href="estimator-diagnostics.pdf">Diagnostic PDF</a> | '
            '<a href="estimator-diagnostics.svg">Diagnostic SVG</a> | '
            '<a href="spectrum.csv">Full spectrum CSV</a> | <a href="autocorrelation.csv">Full autocorrelation CSV</a> | '
            '<a href="diagnostic-input.csv">Detrended inputs</a> | '
            '<a href="estimator-diagnostics.json">Diagnostic details</a></p>'
            '<h2>3. Why the rate was accepted or withheld</h2><p>'+html.escape(reason)+'</p>'+table+
            '<p>Spectral concentration uses the peak and adjacent bins within the search band. '
            'Pre-bandpass energy fraction uses the smoothed/resampled signal before bandpass, '
            'with all positive-frequency bins in its denominator. FFT uses linear detrending and '
            'a Hann window; autocorrelation uses detrending without Hann tapering. '
            'The four thresholds are heuristic rules, not calibrated probabilities.</p>')


def export_diagnostics(output, points, result, config):
    output = Path(output)
    summary, selected, arrays = prepare_diagnostics(points, result, config)
    spectrum, autocorrelation, inputs = [], [], []
    if arrays is not None:
        total = max(float(arrays['power'][1:].sum()), 1e-30)
        raw_total = max(float(arrays['raw_power'][1:].sum()), 1e-30)
        for i, hz in enumerate(arrays['frequencies']):
            spectrum.append(dict(zip(SPECTRUM_FIELDS, (float(hz), float(hz*60),
                float(arrays['power'][i]), float(arrays['raw_power'][i]),
                float(arrays['power'][i]/total), float(arrays['raw_power'][i]/raw_total),
                bool(arrays['band'][i]), bool(arrays['near'][i]), i == arrays['peak']))))
        peaks = set(arrays['acf_peaks'])
        for i, value in enumerate(arrays['ac']):
            autocorrelation.append(dict(zip(ACF_FIELDS, (i, i/config.sample_hz, float(value),
                arrays['min_lag'] <= i <= arrays['max_lag'], i in peaks, i == arrays['lag']))))
        inputs = [dict(zip(INPUT_FIELDS, (p.time, p.available_at, float(arrays['raw_detrended'][i]),
                  float(arrays['filtered_detrended'][i]), float(arrays['taper'][i]))))
                  for i, p in enumerate(selected)]
    write_csv(output/'spectrum.csv', spectrum, SPECTRUM_FIELDS)
    write_csv(output/'autocorrelation.csv', autocorrelation, ACF_FIELDS)
    write_csv(output/'diagnostic-input.csv', inputs, INPUT_FIELDS)
    write_json(output/'estimator-diagnostics.json', summary)
    save_figure(draw_diagnostics(summary, arrays, config), output, 'estimator-diagnostics')
    return summary, diagnostics_html(summary)
