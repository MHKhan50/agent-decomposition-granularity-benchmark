import unittest

from adgb.model import TinyMLP
from adgb.simulator import generate_dataset


class ModelTests(unittest.TestCase):
    def test_model_fits_and_predicts_valid_choice(self):
        rows = generate_dataset([17, 29], 30)
        model = TinyMLP(hidden=8, seed=5).fit(rows, epochs=3)
        self.assertIn(model.predict(rows[0]), (1, 2, 4, 8))
        self.assertGreaterEqual(model.accuracy(rows), 0.0)
        self.assertLessEqual(model.accuracy(rows), 1.0)

    def test_empty_training_data_rejected(self):
        with self.assertRaises(ValueError):
            TinyMLP().fit([], epochs=2)


if __name__ == "__main__":
    unittest.main()
