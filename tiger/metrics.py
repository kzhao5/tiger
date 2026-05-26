from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    from jiwer import cer, wer
except Exception:
    def wer(reference, prediction):
        ref_words = reference.split()
        pred_words = prediction.split()
        if not ref_words:
            return float(bool(pred_words))
        mismatches = sum(1 for a, b in zip(ref_words, pred_words) if a != b) + abs(len(ref_words) - len(pred_words))
        return mismatches / max(1, len(ref_words))

    def cer(reference, prediction):
        if not reference:
            return float(bool(prediction))
        mismatches = sum(1 for a, b in zip(reference, prediction) if a != b) + abs(len(reference) - len(prediction))
        return mismatches / max(1, len(reference))

try:
    from rouge_score import rouge_scorer
except Exception:
    rouge_scorer = None

try:
    import sacrebleu
except Exception:
    sacrebleu = None

from .schemas import CandidateOutput, InferenceOutput, TaskType, UnifiedSample
from .utils import exact_match, normalize_text, option_normalize


def text_metrics(prediction: str, reference: str) -> Dict[str, float]:

    if rouge_scorer is not None:
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        rouge = scorer.score(reference, prediction)["rougeL"].fmeasure
    else:
        rouge = exact_match(prediction, reference)
    if sacrebleu is not None:
        bleu = sacrebleu.sentence_bleu(prediction, [reference]).score / 100.0
    else:
        bleu = exact_match(prediction, reference)
    em = exact_match(prediction, reference)
    return {"rougeL": float(rouge), "bleu": float(bleu), "em": float(em)}


def asr_metrics(prediction: str, reference: str) -> Dict[str, float]:
    return {
        "wer": float(wer(reference, prediction)),
        "cer": float(cer(reference, prediction)),
        "em": float(exact_match(prediction, reference)),
    }


def bbox_iou(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def parse_first_bbox(layout) -> Optional[Tuple[float, float, float, float]]:
    if not layout:
        return None
    # Handle both dict ({"elements": [...]}) and list ([x1,y1,x2,y2]) formats
    if isinstance(layout, (list, tuple)):
        if len(layout) >= 4:
            return tuple(float(v) for v in layout[:4])
        return None
    if not isinstance(layout, dict):
        return None
    elements = layout.get("elements", [])
    if not elements:
        return None
    e = elements[0]
    if isinstance(e, (list, tuple)):
        if len(e) >= 4:
            return tuple(float(v) for v in e[:4])
        return None
    if not isinstance(e, dict):
        return None
    x1 = float(e.get("x1", e.get("x", 0.0)))
    y1 = float(e.get("y1", e.get("y", 0.0)))
    x2 = float(e.get("x2", x1 + float(e.get("width", 0.0))))
    y2 = float(e.get("y2", y1 + float(e.get("height", 0.0))))
    return (x1, y1, x2, y2)


def layout_metrics(pred_layout: Optional[Dict[str, Any]], target_layout: Optional[Dict[str, Any]]) -> Dict[str, float]:
    pred_box = parse_first_bbox(pred_layout)
    tgt_box = parse_first_bbox(target_layout)
    if pred_box is None or tgt_box is None:
        return {"iou": 0.0}
    return {"iou": float(bbox_iou(pred_box, tgt_box))}


def task_metrics(sample: UnifiedSample, prediction: CandidateOutput) -> Dict[str, float]:
    if sample.task_type == TaskType.IMAGE_TEXT_TO_TEXT_MCQ:
        pred = option_normalize(prediction.text)
        ref = option_normalize(sample.answer)
        return {"accuracy": float(pred == ref), "em": float(pred == ref)}
    if sample.task_type == TaskType.IMAGE_TEXT_TO_TEXT_FREEFORM:
        if sample.target_text is None:
            return {}
        return text_metrics(prediction.text or "", sample.target_text)
    if sample.task_type == TaskType.AUDIO_TO_TEXT_CAPTION:
        if sample.target_text is None:
            return {}
        return text_metrics(prediction.text or "", sample.target_text)
    if sample.task_type == TaskType.AUDIO_TO_TEXT_ASR:
        if sample.target_text is None:
            return {}
        return asr_metrics(prediction.text or "", sample.target_text)
    if sample.task_type in {TaskType.TEXT_TO_LAYOUT, TaskType.IMAGE_TEXT_TO_LAYOUT}:
        return layout_metrics(prediction.layout, sample.target_layout)
    return {}


def aggregate_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    bucket: Dict[str, List[float]] = {}
    for row in rows:
        for k, v in row.items():
            bucket.setdefault(k, []).append(float(v))
    return {k: float(np.mean(vs)) for k, vs in bucket.items()}


def evidence_metrics(inference: InferenceOutput) -> Dict[str, float]:
    total = len(inference.assessments)
    if total == 0:
        return {"FSR": 0.0, "UFR": 0.0, "CR": 0.0}
    fsr, ufr, cr = 0, 0, 0
    for a in inference.assessments:
        statuses = list(a.status_by_modality.values())
        if any(s.value in {"SUPPORTED", "ENTAILED"} for s in statuses):
            fsr += 1
        if all(s.value == "UNKNOWN" for s in statuses):
            ufr += 1
        if any(s.value == "CONTRADICTED" for s in statuses):
            cr += 1
    return {
        "FSR": fsr / total,
        "UFR": ufr / total,
        "CR": cr / total,
    }


def budget_efficiency(task_gain: float, compute_cost: float) -> float:
    return float(task_gain / max(compute_cost, 1e-8))
