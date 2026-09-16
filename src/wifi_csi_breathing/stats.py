"""Reused descriptive statistics from the operator's radar host analysis.py.

Source: iwr6843aop-vital-signs at 66929da86d3b47032bfac48b714f2730b2f30965.
No radar estimator, protocol or TI embedded implementation is included.
"""
from dataclasses import dataclass
import math
import statistics
from typing import Iterable


@dataclass(frozen=True, slots=True)
class SeriesStatistics:
    count: int
    minimum: float
    maximum: float
    mean: float
    median: float
    population_standard_deviation: float
    median_absolute_deviation: float


def summarize_values(values: Iterable[float]) -> SeriesStatistics | None:
    finite_values = [float(value) for value in values if math.isfinite(float(value))]
    if not finite_values:
        return None
    median = statistics.median(finite_values)
    deviations = [abs(value - median) for value in finite_values]
    return SeriesStatistics(
        count=len(finite_values), minimum=min(finite_values), maximum=max(finite_values),
        mean=statistics.fmean(finite_values), median=median,
        population_standard_deviation=statistics.pstdev(finite_values),
        median_absolute_deviation=statistics.median(deviations),
    )
