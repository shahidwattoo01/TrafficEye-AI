"""Core data structures shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class Detection:
    box: np.ndarray            # xyxy, float
    cls: int
    name: str
    conf: float
    track_id: Optional[int] = None


@dataclass
class VehicleState:
    """Everything the pipeline knows about one (relinked) vehicle."""

    vid: int                       # stable vehicle ID shown to the user
    name: str                      # Bike / Car / Jeep / Van
    box: np.ndarray                # smoothed box
    conf: float
    first_frame: int
    last_seen: int
    class_votes: dict = field(default_factory=dict)   # name -> count, majority vote of vehicle type
    in_zone: bool = False
    direction: Optional[str] = None                   # IN / OUT after crossing the counting line
    helmet: str = "unknown"                           # ok / violation / unknown
    seatbelt: str = "unknown"
    riders: int = 0
    speed_kmh: Optional[float] = None
    plate: Optional[str] = None
    plate_conf: float = 0.0
    violations: list = field(default_factory=list)    # list of violation type strings
    snapshots: dict = field(default_factory=dict)     # violation type -> path
    trail: list = field(default_factory=list)         # recent bottom-centre points

    @property
    def vehicle_type(self) -> str:
        if not self.class_votes:
            return self.name
        return max(self.class_votes.items(), key=lambda kv: kv[1])[0]

    @property
    def is_two_wheeler(self) -> bool:
        return self.vehicle_type == "Bike"


@dataclass
class ViolationEvent:
    vid: int
    vehicle_type: str
    violation: str              # No-Helmet / No-Seatbelt / Wrong-Way / Over-Speed / Red-Light / Triple-Riding
    frame: int
    timestamp_s: float
    confidence: float
    box: np.ndarray
    snapshot: Optional[str] = None
    plate: Optional[str] = None
    details: dict = field(default_factory=dict)
