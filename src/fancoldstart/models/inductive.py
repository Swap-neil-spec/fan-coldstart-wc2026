"""Inductive cold-start head (paper Section 4.5, H6).

A linear GraphSAGE-mean regressor: each fan is represented by its own position
features concatenated with the mean of its neighbors' features (one message-
passing step). A ridge head maps that representation to predicted future
engagement. The model is trained on a training fold and applied inductively to
fans held out entirely from training, whose representation is still computable
from their neighbors. It is compared against the population-average fallback that
a plain BG/NBD supplies for a fan with no usable individual signal.

Linear and closed-form on purpose: it is inductive and reproducible with no
training-loop randomness. A deeper PyTorch Geometric version is a drop-in
replacement and is noted in the README, but is not required for the claim.
"""
import numpy as np


def _aggregate(G, fan_ids, feat_by_fan):
    """Mean of neighbor features for each fan (zeros if isolated)."""
    dim = next(iter(feat_by_fan.values())).shape[0]
    agg = np.zeros((len(fan_ids), dim), float)
    for i, fan in enumerate(fan_ids):
        neigh = [v for v in G.neighbors(fan) if v in feat_by_fan]
        if neigh:
            agg[i] = np.mean([feat_by_fan[v] for v in neigh], axis=0)
    return agg


class InductiveHead:
    def __init__(self, ridge=1.0):
        self.ridge = ridge
        self.mean_ = None
        self.std_ = None
        self.w_ = None
        self.baseline_ = 0.0

    def _design(self, G, fan_ids, feat_by_fan):
        self_feat = np.array([feat_by_fan[f] for f in fan_ids], float)
        agg = _aggregate(G, fan_ids, feat_by_fan)
        Z = np.hstack([self_feat, agg])
        Z = (Z - self.mean_) / self.std_
        return np.column_stack([np.ones(len(Z)), Z])

    def fit(self, G, train_ids, feat_by_fan, y_train):
        self_feat = np.array([feat_by_fan[f] for f in train_ids], float)
        agg = _aggregate(G, train_ids, feat_by_fan)
        Z = np.hstack([self_feat, agg])
        self.mean_ = Z.mean(axis=0)
        self.std_ = Z.std(axis=0) + 1e-8
        Zc = np.column_stack([np.ones(len(Z)), (Z - self.mean_) / self.std_])
        k = Zc.shape[1]
        R = self.ridge * np.eye(k)
        R[0, 0] = 0.0  # do not penalize the intercept
        self.w_ = np.linalg.solve(Zc.T @ Zc + R, Zc.T @ np.asarray(y_train, float))
        self.baseline_ = float(np.mean(y_train))
        return self

    def predict(self, G, fan_ids, feat_by_fan):
        Zc = self._design(G, fan_ids, feat_by_fan)
        return np.maximum(Zc @ self.w_, 0.0)

    def baseline(self, fan_ids):
        return np.full(len(fan_ids), self.baseline_, float)
