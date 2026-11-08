import random
import unittest

from adgb.simulator import FEATURES, GRANULARITIES, evaluate_granularity, generate_dataset, make_episode


class SimulatorTests(unittest.TestCase):
    def test_reproducible_generation(self):
        self.assertEqual(generate_dataset([17], 4), generate_dataset([17], 4))

    def test_disjoint_seeds_change_rows(self):
        self.assertNotEqual(generate_dataset([17], 2), generate_dataset([29], 2))

    def test_feature_schema_and_label(self):
        row = generate_dataset([17], 1)[0]
        self.assertTrue(all(name in row for name in FEATURES))
        self.assertIn(int(row["oracle_granularity"]), GRANULARITIES)

    def test_objective_is_finite_and_metrics_bounded(self):
        row = make_episode(random.Random(4), 4, 0)
        for g in GRANULARITIES:
            metrics = evaluate_granularity(row, g)
            self.assertGreaterEqual(metrics["objective"], 0.0)
            self.assertGreaterEqual(metrics["quality_score"], 0.0)
            self.assertLessEqual(metrics["quality_score"], 1.0)
            self.assertGreaterEqual(metrics["merge_conflict_rate"], 0.0)
            self.assertLessEqual(metrics["merge_conflict_rate"], 1.0)

    def test_invalid_granularity_rejected(self):
        row = make_episode(random.Random(4), 4, 0)
        with self.assertRaises(ValueError):
            evaluate_granularity(row, 3)


if __name__ == "__main__":
    unittest.main()
