import json
import numpy as np
from collections import defaultdict
from typing import Dict
from app.personalization.models import create_session, BanditState


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

    def save(self, stage: str):
        Session = create_session()
        s = Session()
        state = s.query(BanditState).filter_by(stage=stage).one_or_none()
        payload = self.to_dict()
        if state is None:
            state = BanditState(stage=stage, arms=payload["arms"], alpha=payload["alpha"], beta=payload["beta"])
            s.add(state)
        else:
            state.arms = payload["arms"]
            state.alpha = payload["alpha"]
            state.beta = payload["beta"]
        s.commit()
        s.close()

    @classmethod
    def load(cls, stage: str, default_arms: list) -> "ThompsonBandit":
        Session = create_session()
        s = Session()
        state = s.query(BanditState).filter_by(stage=stage).one_or_none()
        if state is None:
            s.close()
            return cls(default_arms)
        inst = cls(state.arms)
        inst.alpha.update(state.alpha or {})
        inst.beta.update(state.beta or {})
        s.close()
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

def update_bandits_from_feedback(modules_used: Dict[str, str], rating: int):
    reward = 1.0 if rating >= 4 else 0.0
    # Load, update, save bandits
    chunker = ThompsonBandit.load("chunker", ["adaptive","fixed_length","semantic"])
    retriever = ThompsonBandit.load("retriever", ["dense","hybrid","advanced","langextract"])
    generator = ThompsonBandit.load("generator", ["none","huggingface","ollama"])
    if "chunker" in modules_used:
        chunker.update(modules_used["chunker"], reward); chunker.save("chunker")
    if "retriever" in modules_used:
        retriever.update(modules_used["retriever"], reward); retriever.save("retriever")
    if "generator" in modules_used:
        generator.update(modules_used["generator"], reward); generator.save("generator")

