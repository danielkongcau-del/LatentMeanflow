import unittest

import numpy as np

from scripts.eval_mask_prior import _compare_distribution, _summarize_distribution


def _make_reference_masks():
    ring = np.zeros((8, 8), dtype=np.int64)
    ring[1:7, 1:7] = 1
    ring[3:5, 3:5] = 2

    road = np.zeros((8, 8), dtype=np.int64)
    road[1:7, 3] = 1
    road[4, 1:7] = 1
    road[2:4, 5:7] = 2
    return [ring, road]


def _make_generated_masks():
    ring = np.zeros((8, 8), dtype=np.int64)
    ring[1:7, 1:7] = 1
    ring[2:6, 2:6] = 2

    road = np.zeros((8, 8), dtype=np.int64)
    road[1:7, 3] = 1
    road[4, 2:6] = 1
    road[1:3, 5:7] = 2
    return [ring, road]


class MaskPriorEvalMetricTest(unittest.TestCase):
    def test_distribution_summary_includes_structure_metrics(self):
        stats = _summarize_distribution(
            _make_reference_masks(),
            num_classes=3,
            ignore_index=None,
            small_region_threshold_ratio=0.1,
            thin_structure_class_ids=[1],
        )

        self.assertEqual(tuple(np.asarray(stats["adjacency_frequency_mean"]).shape), (3, 3))
        self.assertTrue(np.isfinite(np.asarray(stats["adjacency_frequency_mean"])).all())
        self.assertIn(1, stats["largest_component_class_share_stats"])
        self.assertIn(1, stats["hole_count_stats"])
        self.assertIn(1, stats["hole_area_ratio_stats"])
        self.assertEqual(stats["thin_structure_class_ids"], [1])
        self.assertIn(1, stats["thin_structure_skeleton_length_ratio_stats"])
        self.assertIn(1, stats["thin_structure_endpoint_count_stats"])
        self.assertIn(1, stats["thin_structure_fragment_count_stats"])

    def test_distribution_compare_reports_new_gap_metrics(self):
        reference_stats = _summarize_distribution(
            _make_reference_masks(),
            num_classes=3,
            ignore_index=None,
            small_region_threshold_ratio=0.1,
            thin_structure_class_ids=[1],
        )
        generated_stats = _summarize_distribution(
            _make_generated_masks(),
            num_classes=3,
            ignore_index=None,
            small_region_threshold_ratio=0.1,
            thin_structure_class_ids=[1],
        )

        gap = _compare_distribution(
            generated_stats,
            reference_stats,
            num_classes=3,
            thin_structure_class_ids=[1],
        )

        for key in (
            "adjacency_matrix_l1_mean",
            "adjacency_matrix_jsd",
            "largest_component_class_share_l1_mean",
            "hole_count_l1_mean",
            "hole_area_ratio_l1_mean",
            "thin_structure_skeleton_length_gap_mean",
            "thin_structure_endpoint_count_gap_mean",
            "thin_structure_fragment_count_gap_mean",
        ):
            self.assertIn(key, gap)
            self.assertIsNotNone(gap[key])
            self.assertTrue(np.isfinite(float(gap[key])))


if __name__ == "__main__":
    unittest.main()
