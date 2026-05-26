"""TigerInferenceEngine — Traceable Inference with Graph-based Evidence Routing.

Public re-exports for the most common entry points. For full access, import
the underlying modules (e.g., `from tiger.evidence import FactExtractor`).
"""

from .models import (
    BaseBackbone,
    DummyBackbone,
    GenerationSettings,
    QwenOmniBackbone,
    build_backbone,
)
from .pipeline import TigerInferenceEngine
from .planner import PlannerConfig
from .schemas import CandidateOutput, UnifiedSample

__all__ = [
    "BaseBackbone",
    "DummyBackbone",
    "GenerationSettings",
    "QwenOmniBackbone",
    "build_backbone",
    "TigerInferenceEngine",
    "PlannerConfig",
    "CandidateOutput",
    "UnifiedSample",
]
