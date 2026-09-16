import math
import unittest

from wifi_csi_breathing.stats import summarize_values


class StatisticsTests(unittest.TestCase):
    def test_generic_radar_statistics_ignore_nonfinite_values(self):
        summary = summarize_values([1, 2, 3, math.nan, math.inf])
        self.assertEqual(summary.count, 3)
        self.assertEqual(summary.mean, 2)
        self.assertEqual(summary.median_absolute_deviation, 1)
        self.assertIsNone(summarize_values([math.nan]))
        self.assertIsNone(summarize_values([]))
