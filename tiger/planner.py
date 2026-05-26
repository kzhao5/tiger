"""TIGER planner — fixed-round iteration.

The loop runs exactly max_steps rounds. Each round repairs f* = argmax r(f).
No action selection, no early stopping. Control is entirely in pipeline.py.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PlannerConfig:
    max_steps: int = 3
