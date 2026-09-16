"""Read-only batch analysis with explicit provenance and non-overwriting exports."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import json

from . import __version__
from .data import FeatureConfig, MODES, read_observations, check_cancel
from .processing import Processor, ProcessingConfig
from .stats import summarize_values


def write_json(path, content):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(content, stream, indent=2, allow_nan=False)
        stream.write('\n')


def write_csv(path, rows, columns):
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def process_observations(observations, mode, config=None, cancel=None):
    processor = Processor(mode, config)
    points, estimates = [], []
    for observation in observations:
        check_cancel(cancel)
        new_points, new_estimates = processor.feed(observation)
        points.extend(new_points)
        estimates.extend(new_estimates)
    return processor, points, estimates


def analyze(datasets, output, modes=MODES, tx=1, rx=1, reference_rx=2, subcarrier=15, cancel=None):
    output = Path(output).resolve()
    if any(output.is_relative_to(dataset.directory.resolve()) for dataset in datasets):
        raise ValueError('Output must not be inside a raw recording')
    modes = tuple(modes)
    if not modes or len(set(modes)) != len(modes):
        raise ValueError('Select at least one non-duplicate feature mode')
    configs = [FeatureConfig(mode, tx, rx, reference_rx, subcarrier) for mode in modes]
    processing = ProcessingConfig()
    output.mkdir(parents=True, exist_ok=False)
    results = []
    for index, dataset in enumerate(datasets, 1):
        for feature_config in configs:
            check_cancel(cancel)
            print(f'Analyzing {dataset.id}: {feature_config.mode}', flush=True)
            observations, audit = read_observations(dataset, feature_config, cancel)
            processor, points, estimates = process_observations(observations, feature_config.mode, processing, cancel)
            destination = output / f'{index:02d}-{feature_config.mode}'
            destination.mkdir()
            write_csv(destination / 'observations.csv', (asdict(p) for p in observations),
                      ('time', 'value', 'boundary', 'reason', 'index'))
            write_csv(destination / 'waveform.csv', (asdict(p) for p in points),
                      ('time', 'available_at', 'raw', 'filtered', 'segment'))
            write_csv(destination / 'estimates.csv', (asdict(e) for e in estimates),
                      ('time', 'available_at', 'segment', 'status', 'candidate_bpm', 'spectral_peak_bpm',
                       'autocorrelation_bpm', 'autocorrelation', 'concentration', 'band_fraction'))
            candidates = [e.candidate_bpm for e in estimates if e.candidate_bpm is not None]
            stats = summarize_values(candidates)
            summary = {
                'dataset_id': dataset.id, 'operator_label': dataset.label,
                'feature': asdict(feature_config), 'processing': asdict(processing),
                'input': audit, 'source_directory': str(dataset.directory),
                'source_hashes': list(dataset.files), 'output_folder': destination.name,
                'uniform_samples': len(points), 'filter_resets': processor.resets,
                'eligible_windows': len(estimates), 'periodic_candidate_windows': len(candidates),
                'candidate_window_fraction': len(candidates) / len(estimates) if estimates else None,
                'candidate_bpm_statistics': asdict(stats) if stats else None,
                'interpretation': 'Exploratory periodicity only; no validated breathing or occupancy decision.',
            }
            dataset.verify(cancel)
            write_json(destination / 'summary.json', summary)
            results.append(summary)
    report = {
        'schema_version': 1, 'software_version': __version__,
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'complete': True,
        'raw_data_modified': False,
        'scope': 'Six-session pilot exploration is not independent physiological validation. '
                 'Operator labels never enter the signal processor. Window fractions are not accuracy.',
        'results': results,
    }
    # This completion record is written last. A partially interrupted directory
    # has no report.json and must not be mistaken for a completed run.
    write_json(output / 'report.json', report)
    return report
