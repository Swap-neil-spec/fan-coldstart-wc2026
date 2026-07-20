"""Temporal calibration/holdout split and the standard customer-base-analysis
transform to per-fan (x, t_x, T), realized future engagement, and persistence.

x is the repeat-event count in the calibration window (total calibration events
minus the first), t_x is the time of the last calibration event measured from the
fan's first event, and T is the observation length from the fan's first event to
the cutoff. The cold-start stratum is defined by a pre-registered threshold on x.
"""
from collections import defaultdict

import numpy as np


class Splits:
    def __init__(self, fan_ids, x, t_x, T, y_future, persist, cold_threshold):
        self.fan_ids = fan_ids
        self.x = x
        self.t_x = t_x
        self.T = T
        self.y_future = y_future
        self.persist = persist
        self.cold_threshold = cold_threshold

    @property
    def cold_mask(self):
        return self.x <= self.cold_threshold


def make_splits(events, t_cutoff, tau, cold_threshold=1):
    cal = defaultdict(list)
    fut = defaultdict(int)
    for e in events:
        t = e["t"]
        if t <= t_cutoff:
            cal[e["fan"]].append(t)
        elif t <= t_cutoff + tau:
            fut[e["fan"]] += 1

    fan_ids = sorted(cal.keys())
    x = np.zeros(len(fan_ids), float)
    t_x = np.zeros(len(fan_ids), float)
    T = np.zeros(len(fan_ids), float)
    y_future = np.zeros(len(fan_ids), float)
    for i, fan in enumerate(fan_ids):
        ts = sorted(cal[fan])
        birth = ts[0]
        x[i] = len(ts) - 1
        t_x[i] = ts[-1] - birth
        T[i] = t_cutoff - birth
        y_future[i] = fut.get(fan, 0)
    persist = (y_future > 0).astype(float)
    return Splits(fan_ids, x, t_x, T, y_future, persist, cold_threshold)
