"""Multi-frame voting: a vehicle is judged on many frames, never on a single one.

For every (vehicle, check) pair we keep the last `window` observations. A decision is made only
after `min_votes` observations, and a violation needs at least `violation_ratio` of the
confidence-weighted votes. Once flagged, a violation is latched so it is reported exactly once.
"""
from __future__ import annotations

from collections import defaultdict, deque


class VoteBox:
    def __init__(self, window: int, min_votes: int, ratio: float, min_conf: float):
        self.window = window
        self.min_votes = min_votes
        self.ratio = ratio
        self.min_conf = min_conf
        self.votes: dict[tuple[int, str], deque] = defaultdict(lambda: deque(maxlen=self.window))
        self.latched: set[tuple[int, str]] = set()

    @classmethod
    def from_config(cls, cfg) -> "VoteBox":
        v = cfg.voting
        return cls(int(v.window), int(v.min_votes), float(v.violation_ratio), float(v.min_conf))

    def add(self, vid: int, kind: str, label: str, conf: float) -> None:
        if conf < self.min_conf:
            return
        self.votes[(vid, kind)].append((label == "violation", conf))

    def status(self, vid: int, kind: str) -> str:
        """'violation', 'ok', or 'unknown' (not enough evidence yet)."""
        key = (vid, kind)
        if key in self.latched:
            return "violation"
        q = self.votes.get(key)
        if not q or len(q) < self.min_votes:
            return "unknown"
        bad = sum(c for v, c in q if v)
        total = sum(c for _, c in q)
        return "violation" if total > 0 and bad / total >= self.ratio else "ok"

    def score(self, vid: int, kind: str) -> float:
        q = self.votes.get((vid, kind))
        if not q:
            return 0.0
        bad = [c for v, c in q if v]
        return sum(bad) / len(bad) if bad else 0.0

    def try_latch(self, vid: int, kind: str) -> bool:
        """True exactly once — the first time this vehicle is confirmed as violating."""
        key = (vid, kind)
        if key in self.latched:
            return False
        if self.status(vid, kind) == "violation":
            self.latched.add(key)
            return True
        return False

    def count(self, vid: int, kind: str) -> int:
        q = self.votes.get((vid, kind))
        return len(q) if q else 0
