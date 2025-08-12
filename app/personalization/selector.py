import json
import numpy as np
from collections import defaultdict


class ThompsonBandit:
    def __init__(self, arms):
        self.alpha = defaultdict(lambda: 1.0)
        self.beta = defaultdict(lambda: 1.0)
        self.arms = list(arms)

    def select(self) -> str:
        samples = {a: np.random.beta(self.alpha[a], self.beta[a]) for a in self.arms}
        return max(samples, key=samples.get)

    def update(self, arm: str, reward: float):
        reward = float(max(0.0, min(1.0, reward)))
        self.alpha[arm] += reward
        self.beta[arm] += (1.0 - reward)

    def to_dict(self):
        return {
            "alpha": dict(self.alpha),
            "beta": dict(self.beta),
            "arms": self.arms,
        }

    @classmethod
    def from_dict(cls, data: dict):
        inst = cls(data.get("arms", []))
        inst.alpha.update(data.get("alpha", {}))
        inst.beta.update(data.get("beta", {}))
        return inst


def choose_modules(user_context_vec=None):
    # In a production system, you would bias bandits with nearest-neighbor
    # successes for similar contexts. Here we keep it simple.
    chunker_bandit = ThompsonBandit(["adaptive", "fixed_length", "semantic"])
    retriever_bandit = ThompsonBandit(["dense", "hybrid", "advanced"])
    gen_bandit = ThompsonBandit(["none", "huggingface", "ollama"])
    return {
        "chunker": chunker_bandit.select(),
        "retriever": retriever_bandit.select(),
        "generator": gen_bandit.select(),
    }

