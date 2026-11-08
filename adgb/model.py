"""A small dependency-free one-hidden-layer neural classifier."""
from __future__ import annotations

import math
import random
from typing import Dict, List, Sequence

from .simulator import FEATURES, GRANULARITIES


class TinyMLP:
    def __init__(self, hidden: int = 12, seed: int = 123):
        self.hidden = hidden
        self.rng = random.Random(seed)
        self.means = [0.0] * len(FEATURES)
        self.stds = [1.0] * len(FEATURES)
        scale1 = math.sqrt(2.0 / len(FEATURES))
        scale2 = math.sqrt(2.0 / hidden)
        self.w1 = [[self.rng.uniform(-scale1, scale1) for _ in FEATURES] for _ in range(hidden)]
        self.b1 = [0.0] * hidden
        self.w2 = [[self.rng.uniform(-scale2, scale2) for _ in range(hidden)] for _ in GRANULARITIES]
        self.b2 = [0.0] * len(GRANULARITIES)

    def _raw(self, row: Dict[str, float]) -> List[float]:
        return [float(row[k]) for k in FEATURES]

    def _forward(self, row: Dict[str, float]):
        x = [(v - m) / s for v, m, s in zip(self._raw(row), self.means, self.stds)]
        h = []
        for j in range(self.hidden):
            z = self.b1[j] + sum(self.w1[j][i] * x[i] for i in range(len(x)))
            h.append(math.tanh(z))
        logits = [self.b2[k] + sum(self.w2[k][j] * h[j] for j in range(self.hidden)) for k in range(len(GRANULARITIES))]
        peak = max(logits)
        exps = [math.exp(max(-40.0, v - peak)) for v in logits]
        total = sum(exps)
        probs = [v / total for v in exps]
        return x, h, probs

    def fit(self, rows: Sequence[Dict[str, float]], epochs: int = 30, learning_rate: float = 0.035) -> "TinyMLP":
        if not rows:
            raise ValueError("training rows cannot be empty")
        if epochs < 1:
            raise ValueError("epochs must be >= 1")
        raw = [self._raw(r) for r in rows]
        self.means = [sum(v[i] for v in raw) / len(raw) for i in range(len(FEATURES))]
        self.stds = []
        for i in range(len(FEATURES)):
            variance = sum((v[i] - self.means[i]) ** 2 for v in raw) / len(raw)
            self.stds.append(max(1e-6, math.sqrt(variance)))

        order = list(range(len(rows)))
        for epoch in range(epochs):
            self.rng.shuffle(order)
            rate = learning_rate / (1.0 + 0.025 * epoch)
            for idx in order:
                row = rows[idx]
                target = GRANULARITIES.index(int(row["oracle_granularity"]))
                x, h, probs = self._forward(row)
                delta2 = list(probs)
                delta2[target] -= 1.0
                grad_w2 = [[delta2[k] * h[j] for j in range(self.hidden)] for k in range(len(GRANULARITIES))]
                grad_b2 = delta2
                delta1 = []
                for j in range(self.hidden):
                    back = sum(self.w2[k][j] * delta2[k] for k in range(len(GRANULARITIES)))
                    delta1.append(back * (1.0 - h[j] * h[j]))
                for k in range(len(GRANULARITIES)):
                    self.b2[k] -= rate * grad_b2[k]
                    for j in range(self.hidden):
                        self.w2[k][j] -= rate * grad_w2[k][j]
                for j in range(self.hidden):
                    self.b1[j] -= rate * delta1[j]
                    for i in range(len(FEATURES)):
                        self.w1[j][i] -= rate * delta1[j] * x[i]
        return self

    def predict(self, row: Dict[str, float]) -> int:
        _, _, probs = self._forward(row)
        return GRANULARITIES[max(range(len(probs)), key=lambda i: probs[i])]

    def accuracy(self, rows: Sequence[Dict[str, float]]) -> float:
        if not rows:
            return 0.0
        correct = sum(self.predict(r) == int(r["oracle_granularity"]) for r in rows)
        return correct / len(rows)
