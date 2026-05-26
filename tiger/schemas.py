from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Tuple


class TaskType(str, Enum):
    IMAGE_TEXT_TO_TEXT_MCQ = "image_text_to_text_mcq"
    IMAGE_TEXT_TO_TEXT_FREEFORM = "image_text_to_text_freeform"
    AUDIO_TO_TEXT_CAPTION = "audio_to_text_caption"
    AUDIO_TO_TEXT_ASR = "audio_to_text_asr"
    TEXT_TO_AUDIO = "text_to_audio"
    TEXT_TO_LAYOUT = "text_to_layout"
    IMAGE_TEXT_TO_LAYOUT = "image_text_to_layout"
    LOCAL_CASE_STUDY = "local_case_study"


class FactStatus(str, Enum):
    SUPPORTED = "SUPPORTED"
    CONTRADICTED = "CONTRADICTED"
    UNKNOWN = "UNKNOWN"
    ENTAILED = "ENTAILED"


class ActionType(str, Enum):
    REPAIR = "REPAIR"    # Repair highest-risk fact via span replacement


@dataclass
class UnifiedSample:
    sample_id: str
    task_type: TaskType
    prompt: str
    image_paths: List[str] = field(default_factory=list)
    audio_paths: List[str] = field(default_factory=list)
    video_paths: List[str] = field(default_factory=list)
    target_text: Optional[str] = None
    target_audio_path: Optional[str] = None
    target_layout: Optional[Dict[str, Any]] = None
    choices: Optional[List[str]] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["task_type"] = self.task_type.value
        return data


@dataclass
class Fact:
    fact_id: str
    subject: str
    predicate: str
    obj: str
    modality: str
    evidence_span: Optional[str] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    time_range: Optional[Tuple[float, float]] = None
    confidence: float = 1.0

    def text(self) -> str:
        return f"{self.subject} | {self.predicate} | {self.obj}"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact_id": self.fact_id,
            "subject": self.subject,
            "predicate": self.predicate,
            "obj": self.obj,
            "modality": self.modality,
            "evidence_span": self.evidence_span,
            "bbox": list(self.bbox) if self.bbox is not None else None,
            "time_range": list(self.time_range) if self.time_range is not None else None,
            "confidence": self.confidence,
        }


@dataclass
class ObservationGraph:
    facts: List[Fact]
    edges: List[Tuple[str, str, str]] = field(default_factory=list)

    def by_modality(self) -> Dict[str, List[Fact]]:
        output: Dict[str, List[Fact]] = {}
        for fact in self.facts:
            output.setdefault(fact.modality, []).append(fact)
        return output


@dataclass
class ClaimGraph:
    facts: List[Fact]
    edges: List[Tuple[str, str, str]] = field(default_factory=list)


@dataclass
class FactAssessment:
    fact: Fact
    status_by_modality: Dict[str, FactStatus]
    support_by_modality: Dict[str, float]
    conflict_by_modality: Dict[str, float]
    risk: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fact": self.fact.to_dict(),
            "status_by_modality": {k: v.value for k, v in self.status_by_modality.items()},
            "support_by_modality": self.support_by_modality,
            "conflict_by_modality": self.conflict_by_modality,
            "risk": self.risk,
        }


@dataclass
class CandidateOutput:
    text: Optional[str] = None
    audio_path: Optional[str] = None
    layout: Optional[Dict[str, Any]] = None
    raw: Dict[str, Any] = field(default_factory=dict)
    generation_trace: List[Dict[str, Any]] = field(default_factory=list)
    candidate_id: Optional[str] = None

    def display_text(self) -> str:
        if self.text:
            return self.text
        if self.layout is not None:
            return str(self.layout)
        if self.audio_path:
            return self.audio_path
        return str(self.raw)


@dataclass
class CriticOutput:
    global_score: float
    bad_fact_id: Optional[str]
    edit: Optional[Dict[str, Any]]
    rationale: Optional[str] = None


@dataclass
class PlannerState:
    step_idx: int = 0
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class InferenceOutput:
    sample: UnifiedSample
    prediction: CandidateOutput
    observation_graph: ObservationGraph
    claim_graph: ClaimGraph
    assessments: List[FactAssessment]
    risk: float
    planner_state: PlannerState
    metrics: Dict[str, float]
    evidence_trace: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample": self.sample.to_dict(),
            "prediction": {
                "text": self.prediction.text,
                "audio_path": self.prediction.audio_path,
                "layout": self.prediction.layout,
                "raw": self.prediction.raw,
                "generation_trace": self.prediction.generation_trace,
                "candidate_id": self.prediction.candidate_id,
            },
            "observation_graph": [f.to_dict() for f in self.observation_graph.facts],
            "claim_graph": [f.to_dict() for f in self.claim_graph.facts],
            "assessments": [a.to_dict() for a in self.assessments],
            "risk": self.risk,
            "planner_state": {
                "max_steps": self.planner_state.step_idx,
                "step_idx": self.planner_state.step_idx,
                "history": self.planner_state.history,
            },
            "metrics": self.metrics,
            "evidence_trace": self.evidence_trace,
        }
