"""Evidence graph extraction and cross-graph entailment.

New algorithm: deterministic tools extract G_x and G_y,
coref edges propagate support scores, risk includes
cross-modal inconsistency term ν·δ(f).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .models import BaseBackbone, GenerationSettings
from .schemas import (
    CandidateOutput,
    ClaimGraph,
    Fact,
    FactAssessment,
    FactStatus,
    ObservationGraph,
    TaskType,
    UnifiedSample,
)
from .tools import extract_graph_with_tools, run_dependency_parser, detections_to_facts
from .utils import normalize_text, try_extract_json


# ---------------------------------------------------------------------------
# Layout helper (kept for RefCOCO layout tasks)
# ---------------------------------------------------------------------------

def parse_layout_facts(layout: Dict[str, Any]) -> List[Fact]:
    elements = layout.get("elements", [])
    facts: List[Fact] = []
    for idx, element in enumerate(elements):
        label = str(element.get("label", element.get("type", f"elem_{idx}")))
        x1 = float(element.get("x1", element.get("x", 0.0)))
        y1 = float(element.get("y1", element.get("y", 0.0)))
        x2 = float(element.get("x2", x1 + float(element.get("width", 0.0))))
        y2 = float(element.get("y2", y1 + float(element.get("height", 0.0))))
        facts.append(Fact(
            fact_id=f"layout_{idx}",
            subject=label,
            predicate="located_at",
            obj=f"[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]",
            modality="layout",
            bbox=(x1, y1, x2, y2),
        ))
    return facts


# ---------------------------------------------------------------------------
# Graph extraction using deterministic tools
# ---------------------------------------------------------------------------

def _infer_source_modality(sample: UnifiedSample) -> str:
    """Determine the dominant input modality from a sample."""
    if sample.image_paths:
        return "image"
    if sample.audio_paths:
        return "audio"
    return "text"


def _infer_target_modality(candidate: CandidateOutput) -> str:
    """Determine the output modality from a candidate."""
    if candidate.layout is not None:
        return "layout"
    if candidate.audio_path:
        return "audio"
    return "text"


class FactExtractor:
    """Extract fact graphs via LLM self-extraction (default) or deterministic tools.

    use_llm=True  → backbone prompts extract (subject, relation, object) triples
    use_llm=False → legacy deterministic tools (Grounding DINO, Stanza, PANNs)
    """

    # ── G_x extraction prompt: multimodal input (image/audio/text) ──
    # Standardized to make subject/predicate/object roles unambiguous so that
    # the field-level support and conflict rules in compare_fact_to_observations
    # can fire reliably. Predicate vocabulary is constrained for spatial,
    # attribute, and count facts to match the rule dictionaries (_ASYMMETRIC_PREDS,
    # _ATTRIBUTE_PREDS) downstream.
    _LLM_EXTRACT_PROMPT_GX = (
        "You are a multimodal fact extraction system. "
        "Extract ALL observable facts from the provided input across all modalities "
        "(image, audio, text). Each fact must be a triple (subject, predicate, object).\n\n"
        "STRICT OUTPUT FORMAT — numbered list, one triple per line, parentheses required:\n"
        "1. (subject, predicate, object)\n"
        "2. (subject, predicate, object)\n"
        "...\n\n"
        "FIELD ROLES — keep these unambiguous so downstream verification can compare facts correctly:\n"
        "- subject: the head entity, written as a bare noun phrase WITHOUT attributes\n"
        "  GOOD: (car, is, red)        BAD: (red car, is, parked)\n"
        "- predicate: the relation. Use the standardized forms below when applicable.\n"
        "- object: the tail entity, attribute value, or count.\n\n"
        "STANDARDIZED PREDICATES — prefer these exact strings:\n"
        "- Attributes (color/size/material/shape):  is  (e.g., (car, is, red))\n"
        "- Counts:                                  count  (e.g., (lights, count, 3))\n"
        "- Spatial (DIRECTIONAL, subject is the figure, object is the ground):\n"
        "    on / under / above / below / left of / right of / in front of / behind / inside\n"
        "- Possession / wearing / carrying:         holding / wearing / carrying / riding\n"
        "- Existence (object present in scene):     exists in  (e.g., (dog, exists in, image))\n"
        "- Actions: use the bare verb               jogging / cooking / talking\n\n"
        "Example — given an image of a park with a dog and a man jogging:\n"
        "1. (dog, exists in, image)\n"
        "2. (dog, is, golden)\n"
        "3. (man, exists in, image)\n"
        "4. (man, jogging on, path)\n"
        "5. (man, wearing, red shirt)\n"
        "6. (dog, left of, man)\n"
        "7. (trees, count, 5)\n"
        "8. (sky, is, blue)\n\n"
        "Example — given an audio clip of a busy street:\n"
        "1. (cars, honking, loudly)\n"
        "2. (people, talking, nearby)\n"
        "3. (engine, running, idle)\n"
        "4. (music, playing from, shop)\n\n"
        "Example — given text 'The president announced a new policy on Tuesday':\n"
        "1. (president, announced, new policy)\n"
        "2. (announcement, happened on, Tuesday)\n\n"
        "Rules:\n"
        "- Extract every object, attribute, spatial relation, action, and count you can verify.\n"
        "- One fact per triple. Do NOT combine multiple claims into one line.\n"
        "- Always put the bare entity in subject; never bundle adjectives into subject.\n"
        "- For spatial predicates, subject is the figure that is positioned relative to object.\n"
        "- Aim for 8-20 triples. Be thorough but only include facts grounded in the input.\n"
        "- Output ONLY the numbered triple list. No explanation, no headers, no prose.\n\n"
    )

    # ── G_y / GT extraction prompt: plain text ──
    # Mirrors the G_x prompt's field-role discipline so that G_y and G_x facts
    # share the same shape and field-level comparison in _flat_similarity is meaningful.
    _LLM_EXTRACT_PROMPT_TEXT = (
        "You are a fact extraction system. Read the text below and decompose it into "
        "ALL factual claims as structured triples (subject, predicate, object).\n\n"
        "STRICT OUTPUT FORMAT — numbered list, one triple per line, parentheses required:\n"
        "1. (subject, predicate, object)\n"
        "2. (subject, predicate, object)\n"
        "...\n\n"
        "FIELD ROLES:\n"
        "- subject: the head entity, written as a bare noun WITHOUT adjectives.\n"
        "  GOOD: (car, is, red)        BAD: (red car, is, parked)\n"
        "- predicate: the relation. Prefer the standardized forms below.\n"
        "- object: tail entity, attribute value, or count.\n\n"
        "STANDARDIZED PREDICATES:\n"
        "- Attributes:              is              e.g., (car, is, red)\n"
        "- Counts:                  count           e.g., (lights, count, 3)\n"
        "- Existence:               exists in       e.g., (dog, exists in, image)\n"
        "- Spatial (directional):   on / under / above / below / left of / right of /\n"
        "                           in front of / behind / inside\n"
        "  (subject is the figure, object is the ground)\n"
        "- Possession/action:       holding / wearing / carrying / riding / using\n"
        "- Free actions:            bare verb (cooking, jogging, talking, ...)\n\n"
        "Example 1 — input: 'A red car is parked near a tall building on a sunny day.'\n"
        "1. (car, is, red)\n"
        "2. (car, parked near, building)\n"
        "3. (building, is, tall)\n"
        "4. (day, is, sunny)\n\n"
        "Example 2 — input: 'The fire hydrant cap is yellow.'\n"
        "1. (fire hydrant cap, is, yellow)\n\n"
        "Example 3 — input: 'There are three traffic lights in the image.'\n"
        "1. (traffic lights, exists in, image)\n"
        "2. (traffic lights, count, 3)\n\n"
        "Example 4 — input: 'A man in a white shirt is cooking in the kitchen while holding a knife.'\n"
        "1. (man, exists in, image)\n"
        "2. (man, wearing, white shirt)\n"
        "3. (man, cooking in, kitchen)\n"
        "4. (man, holding, knife)\n\n"
        "Example 5 — input: 'The dog is on top of the car.'\n"
        "1. (dog, on, car)\n"
        "   (note: the dog is the figure; the car is the ground.)\n\n"
        "Rules:\n"
        "- Extract every claim, even from a short single sentence.\n"
        "- Keep the subject a bare noun; never bundle attributes into subject.\n"
        "- Each claim = one triple. Do NOT combine multiple facts.\n"
        "- For spatial predicates, subject is the figure positioned relative to object.\n"
        "- Output ONLY the numbered triple list. No prose.\n\n"
        "Now extract from this text:\n"
    )

    def __init__(self, backbone: BaseBackbone, settings: Optional[GenerationSettings] = None,
                 use_llm: bool = True):
        self.backbone = backbone
        self.use_llm = use_llm
        self.settings = settings or GenerationSettings(
            max_new_tokens=256, temperature=0.0, do_sample=False
        )
        self._extract_settings = GenerationSettings(
            max_new_tokens=512, temperature=0.0, top_p=1.0,
            do_sample=False, return_audio=False, voice="Chelsie", num_candidates=1,
        )

    # ── LLM-based extraction ──

    def _llm_extract_gx(self, sample: UnifiedSample, context_text: str = "",
                        modality: str = "text") -> List[Fact]:
        """Extract facts from multimodal input (G_x). Sample carries image/audio."""
        prompt = self._LLM_EXTRACT_PROMPT_GX
        if context_text:
            prompt += f"Additional text context: {context_text[:400]}\n\n"
        prompt += "Extract all facts now:"
        out = self.backbone.generate(sample, prompt, self._extract_settings)
        raw = (out.text or "").strip()
        return self._parse_triple_lines(raw, modality)

    def _llm_extract_text(self, sample: UnifiedSample, text_content: str,
                          modality: str = "text") -> List[Fact]:
        """Extract facts from plain text (G_y and GT). Stronger prompt for text-only."""
        if not text_content or not text_content.strip():
            return []
        prompt = self._LLM_EXTRACT_PROMPT_TEXT + f'"{text_content[:600]}"'
        out = self.backbone.generate(sample, prompt, self._extract_settings)
        raw = (out.text or "").strip()
        facts = self._parse_triple_lines(raw, modality)

        # Fallback: if LLM returns 0 facts from non-empty text, use Stanza
        if not facts and len(text_content.strip()) > 10:
            fallback_facts, _ = extract_graph_with_tools(text_content, modality="text")
            if fallback_facts:
                facts = fallback_facts

        return facts

    @staticmethod
    def _parse_triple_lines(raw: str, modality: str = "text") -> List[Fact]:
        """Parse (subject, relation, object) triples from LLM output.

        Handles multiple formats:
        - (s, r, o) per line
        - s | r | o per line
        - Comma-separated stream: s, r, o, s, r, o, ...
        - Chinese comma variants: s，r，o
        """
        import re

        # Normalize chinese commas/punctuation
        raw = raw.replace("，", ",").replace("、", ",").replace("；", ";")

        # Filter out known garbage lines
        _SKIP = {"subject, relation, object", "subject | relation | object",
                 "subject,relation,object", "(subject, relation, object)"}

        facts: List[Fact] = []
        lines = raw.split("\n")

        for line in lines:
            line = line.strip().lstrip("-•*0123456789.)")
            line = line.strip()
            if not line or line.lower() in _SKIP:
                continue

            parsed_from_line = []

            # Format 1: (s, r, o) with parentheses — most reliable
            if "(" in line:
                for m in re.finditer(r'\(([^)]+)\)', line):
                    parts = [p.strip() for p in m.group(1).split(",", 2)]
                    if len(parts) >= 2:
                        parsed_from_line.append(parts)

            # Format 2: s | r | o with pipes
            if not parsed_from_line and "|" in line:
                parts = [p.strip() for p in line.split("|", 2)]
                if len(parts) >= 2:
                    parsed_from_line.append(parts)

            # Format 3: "s - r - o" with dashes (some LLMs use this)
            if not parsed_from_line and " - " in line:
                parts = [p.strip() for p in line.split(" - ", 2)]
                if len(parts) >= 2:
                    parsed_from_line.append(parts)

            # Format 4: comma-separated stream (s, r, o, s, r, o, ...)
            if not parsed_from_line and line.count(",") >= 5:
                tokens = [t.strip() for t in line.split(",") if t.strip()]
                for i in range(0, len(tokens) - 2, 3):
                    parsed_from_line.append(tokens[i:i+3])

            # Format 5: single triple: s, r, o
            if not parsed_from_line and "," in line:
                parts = [p.strip() for p in line.split(",", 2)]
                if len(parts) >= 2:
                    parsed_from_line.append(parts)

            # Format 6: colon-separated "subject: relation: object"
            if not parsed_from_line and line.count(":") >= 2:
                parts = [p.strip() for p in line.split(":", 2)]
                if len(parts) >= 2 and all(len(p) < 60 for p in parts):
                    parsed_from_line.append(parts)

            for parts in parsed_from_line:
                subj = parts[0]
                pred = parts[1] if len(parts) > 1 else ""
                obj = parts[2] if len(parts) > 2 else ""
                if not subj or len(subj) > 80:
                    continue
                # Skip if it looks like a header/template
                if subj.lower() in ("subject", "entity", "s") and pred.lower() in ("relation", "predicate", "r"):
                    continue
                facts.append(Fact(
                    fact_id=f"f{len(facts)}",
                    subject=subj, predicate=pred, obj=obj,
                    modality=modality, confidence=0.9,
                ))

        return facts

    # ── Public extraction methods ──

    def extract_observation_graph(self, sample: UnifiedSample) -> ObservationGraph:
        """Extract G_x from input using multimodal LLM prompt or tools."""
        if not self.use_llm:
            return self._extract_observation_graph_tools(sample)

        all_facts: List[Fact] = []

        # Image modality: extract from image only (when present), tagged as "image".
        if sample.image_paths:
            image_only = UnifiedSample(
                sample_id=sample.sample_id + "__gx_img",
                task_type=sample.task_type,
                prompt="",  # do not let prompt text leak into image extraction
                image_paths=sample.image_paths,
                audio_paths=[],
            )
            facts = self._llm_extract_gx(image_only, context_text="", modality="image")
            all_facts.extend(facts)

        # Audio modality: extract from audio only (when present), tagged as "audio".
        # Run a separate LLM call so the resulting facts land in their own modality
        # bucket, which is required for the cross-modal disagreement term δ(f) to
        # produce a non-zero signal.
        if sample.audio_paths:
            audio_only = UnifiedSample(
                sample_id=sample.sample_id + "__gx_aud",
                task_type=sample.task_type,
                prompt="",
                image_paths=[],
                audio_paths=sample.audio_paths,
            )
            facts = self._llm_extract_gx(audio_only, context_text="", modality="audio")
            all_facts.extend(facts)

        # Text modality: extract from the prompt text.
        if sample.prompt:
            dummy = UnifiedSample(
                sample_id="gx_text", task_type=TaskType.IMAGE_TEXT_TO_TEXT_FREEFORM,
                prompt=sample.prompt,
            )
            facts = self._llm_extract_text(dummy, sample.prompt, modality="text")
            all_facts.extend(facts)

        # Re-number fact IDs
        for i, f in enumerate(all_facts):
            f.fact_id = f"f{i}"

        # Edges: cross-modal coref (existing) plus within-modal coref so
        # _graph_propagated_support has real neighbors to walk.
        edges = _build_cross_modal_coref(all_facts)
        edges += _build_within_modal_coref(all_facts)
        return ObservationGraph(facts=all_facts, edges=edges)

    def extract_claim_graph(
        self, candidate: CandidateOutput, default_modality: str = "text"
    ) -> ClaimGraph:
        """Extract G_y from output text using text-specific LLM prompt."""
        if candidate.layout is not None:
            facts = parse_layout_facts(candidate.layout)
            return ClaimGraph(facts=facts, edges=[])

        if not self.use_llm:
            return self._extract_claim_graph_tools(candidate, default_modality)

        content = candidate.text or ""
        if candidate.audio_path:
            from .tools import run_asr_and_acoustic_analysis
            try:
                asr_results = run_asr_and_acoustic_analysis(candidate.audio_path)
                # Extract transcript text from ASR results
                transcript_parts = [r.get("object", r.get("entity", ""))
                                    for r in asr_results if r.get("relation") == "transcribed_as"]
                if not transcript_parts:
                    transcript_parts = [r.get("entity", "") for r in asr_results]
                content = " ".join(p for p in transcript_parts if p) or content
            except Exception:
                pass  # keep original content

        # Use the stronger text extraction prompt (with examples + fallback)
        dummy = UnifiedSample(
            sample_id="extract", task_type=TaskType.IMAGE_TEXT_TO_TEXT_FREEFORM,
            prompt=content,
        )
        facts = self._llm_extract_text(dummy, content, modality=default_modality)
        # Build within-graph coref edges so that _graph_propagated_support over G_y
        # has actual neighbors to propagate through (previous default was edges=[]).
        edges = _build_within_modal_coref(facts)
        return ClaimGraph(facts=facts, edges=edges)

    # ── Legacy deterministic tool extraction ──

    def _extract_observation_graph_tools(self, sample: UnifiedSample) -> ObservationGraph:
        """Legacy: Extract G_x using deterministic tools."""
        all_facts: List[Fact] = []
        all_edges: List[Tuple[str, str, str]] = []
        offset = 0

        for img_path in sample.image_paths:
            facts, edges = extract_graph_with_tools(img_path, modality="image")
            facts, edges = _offset_ids(facts, edges, offset)
            all_facts.extend(facts)
            all_edges.extend(edges)
            offset += len(facts)

        for audio_path in sample.audio_paths:
            facts, edges = extract_graph_with_tools(audio_path, modality="audio")
            facts, edges = _offset_ids(facts, edges, offset)
            all_facts.extend(facts)
            all_edges.extend(edges)
            offset += len(facts)

        if sample.prompt:
            facts, edges = extract_graph_with_tools(sample.prompt, modality="text")
            facts, edges = _offset_ids(facts, edges, offset)
            all_facts.extend(facts)
            all_edges.extend(edges)
            offset += len(facts)

        all_edges.extend(_build_cross_modal_coref(all_facts))
        return ObservationGraph(facts=all_facts, edges=all_edges)

    def _extract_claim_graph_tools(
        self, candidate: CandidateOutput, default_modality: str = "text"
    ) -> ClaimGraph:
        """Legacy: Extract G_y using deterministic tools."""
        content = candidate.text or ""
        if candidate.audio_path:
            facts, edges = extract_graph_with_tools(
                candidate.audio_path, modality="audio"
            )
        else:
            facts, edges = extract_graph_with_tools(content, modality="text")
        return ClaimGraph(facts=facts, edges=edges)


def _offset_ids(
    facts: List[Fact],
    edges: List[Tuple[str, str, str]],
    offset: int,
) -> Tuple[List[Fact], List[Tuple[str, str, str]]]:
    """Offset fact IDs to prevent collisions when merging multiple sources."""
    if offset == 0:
        return facts, edges
    id_map = {}
    new_facts = []
    for f in facts:
        new_id = f"f{offset + int(f.fact_id.lstrip('f').split('_')[0]) if f.fact_id.startswith('f') and f.fact_id[1:].split('_')[0].isdigit() else hash(f.fact_id) % 10000}"
        id_map[f.fact_id] = new_id
        new_facts.append(Fact(
            fact_id=new_id,
            subject=f.subject,
            predicate=f.predicate,
            obj=f.obj,
            modality=f.modality,
            evidence_span=f.evidence_span,
            bbox=f.bbox,
            time_range=f.time_range,
            confidence=f.confidence,
        ))
    new_edges = [(id_map.get(a, a), id_map.get(b, b), t) for a, b, t in edges]
    return new_facts, new_edges


def _build_cross_modal_coref(facts: List[Fact]) -> List[Tuple[str, str, str]]:
    """Build coref edges between facts from different modalities that share entity names."""
    entity_to_facts: Dict[str, List[Fact]] = defaultdict(list)
    for f in facts:
        key = f.subject.lower().strip()
        if key:
            entity_to_facts[key].append(f)

    edges = []
    for key, group in entity_to_facts.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                if group[i].modality != group[j].modality:
                    edges.append((group[i].fact_id, group[j].fact_id, "coref"))
    return edges


def _build_within_modal_coref(facts: List[Fact]) -> List[Tuple[str, str, str]]:
    """Build coref edges between facts that share a normalized entity name,
    regardless of modality. Used inside a single graph (e.g., G_y) so that
    BFS-based support propagation can actually walk between facts about the
    same entity (the previous default of edges=[] left _graph_propagated_support
    with a trivial 1-node BFS)."""
    entity_to_facts: Dict[str, List[Fact]] = defaultdict(list)
    for f in facts:
        key = f.subject.lower().strip()
        if key:
            entity_to_facts[key].append(f)

    edges: List[Tuple[str, str, str]] = []
    for key, group in entity_to_facts.items():
        if len(group) < 2:
            continue
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                edges.append((group[i].fact_id, group[j].fact_id, "coref"))
    return edges


# ---------------------------------------------------------------------------
# Similarity functions
# ---------------------------------------------------------------------------

def _token_set(text: str) -> set:
    return {tok for tok in normalize_text(text).split() if tok}


_NUM_RE = re.compile(r"\b\d+\b")


def _numeric_tokens(text: str) -> List[str]:
    return _NUM_RE.findall(text)


def _jaccard(a: str, b: str) -> float:
    sa, sb = _token_set(a), _token_set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / max(1, len(sa | sb))


def _bbox_iou(
    a: Optional[Tuple[float, float, float, float]],
    b: Optional[Tuple[float, float, float, float]],
) -> float:
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


_sem_model = None

def _get_sem_model():
    global _sem_model
    if _sem_model is None:
        from sentence_transformers import SentenceTransformer
        _sem_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    return _sem_model


# LRU cache for individual text → embedding. Many calls share the same
# short subject/predicate/object strings ("bed", "is", "exists in", ...),
# so caching gets us a ~10-100× speedup on per-fact entailment loops.
from functools import lru_cache

@lru_cache(maxsize=32768)
def _encode_cached(text: str):
    return _get_sem_model().encode([text], normalize_embeddings=True)[0]


def _semantic_similarity(text_a: str, text_b: str) -> float:
    """Cosine similarity between sentence embeddings (cached per text)."""
    if not text_a or not text_b:
        return 0.0
    a = _encode_cached(text_a)
    b = _encode_cached(text_b)
    return float(a @ b)


# Field-level support weights (subject is the gate; predicate and object carry signal).
_SUBJ_GATE = 0.4    # below this, treat facts as unrelated entities (return 0)
_W_SUBJ = 0.4       # contribution from subject match
_W_PRED = 0.3       # contribution from predicate match
_W_OBJ = 0.3        # contribution from object match


def _flat_similarity(claim: Fact, obs: Fact) -> float:
    """Field-level support score between two facts.

    Earlier versions collapsed (subject, predicate, object) into one string and
    took sentence-transformer cosine over the joined form. That hid the role of
    each field — identical subjects/predicates would inflate similarity even
    when the object disagreed (the attribute / relation blind spot).

    Now: gate on subject (require entities to plausibly co-refer), then sum a
    weighted contribution from each field. Discrete-value slots (count,
    attribute predicates) need tighter object agreement because cosine over
    short strings like "red" vs "blue" or "2" vs "5" stays around 0.6-0.7
    — too lax to reject genuinely different values.

    Falls back to whole-text cosine if subject is missing (e.g., layout facts).
    BBox IoU is still consulted when both facts carry geometry (tools path).
    """
    # If field decomposition is unavailable on either side, fall back to whole-text cosine.
    if not claim.subject or not obs.subject:
        sem_sim = _semantic_similarity(claim.text(), obs.text())
    else:
        subj_sim = _semantic_similarity(
            normalize_text(claim.subject), normalize_text(obs.subject))

        # Subject gate: if subjects do not plausibly refer to the same entity,
        # this observation cannot support the claim, regardless of predicate/object overlap.
        if subj_sim < _SUBJ_GATE:
            sem_sim = 0.0
        else:
            pred_sim = (_semantic_similarity(
                normalize_text(claim.predicate), normalize_text(obs.predicate))
                if claim.predicate and obs.predicate else 0.5)
            obj_sim = (_semantic_similarity(
                normalize_text(claim.obj), normalize_text(obs.obj))
                if claim.obj and obs.obj else 0.5)

            claim_pred_n = _norm_pred(claim.predicate)
            obs_pred_n = _norm_pred(obs.predicate)

            # Count predicate: require exact numeric agreement, otherwise no support.
            if "count" in claim_pred_n or "count" in obs_pred_n:
                c_nums = _numeric_tokens(claim.obj or "")
                o_nums = _numeric_tokens(obs.obj or "")
                if c_nums and o_nums and c_nums != o_nums:
                    return 0.0
                if c_nums and o_nums and c_nums == o_nums:
                    obj_sim = 1.0

            # Attribute predicate: cosine on attribute values (red/blue, big/small)
            # tends to stay above 0.6 even when values differ. Tighten by remapping
            # [0, 0.85] -> 0 and [0.85, 1] -> [0, 1] so only near-identical
            # attribute values produce real support.
            if (claim_pred_n in _ATTRIBUTE_PREDS
                    and obs_pred_n in _ATTRIBUTE_PREDS):
                obj_sim = max(0.0, (obj_sim - 0.85) / 0.15)

            sem_sim = _W_SUBJ * subj_sim + _W_PRED * pred_sim + _W_OBJ * obj_sim

    # BBox IoU when both facts carry geometry (tools path only — LLM path has bbox=None).
    if claim.bbox is not None and obs.bbox is not None:
        iou = _bbox_iou(claim.bbox, obs.bbox)
        sem_sim = max(sem_sim, iou)

    return sem_sim


# ---------------------------------------------------------------------------
# Coref edge propagation (Step 4.2)
# ---------------------------------------------------------------------------

def _build_adjacency(graph) -> Dict[str, List[Tuple[str, str]]]:
    """Build adjacency list from graph edges: fact_id -> [(neighbor_id, edge_type)]."""
    adj: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
    for a, b, etype in graph.edges:
        adj[a].append((b, etype))
        adj[b].append((a, etype))
    return adj


def _graph_propagated_support(
    claim: Fact,
    claim_graph: ClaimGraph,
    obs_facts_in_modality: List[Fact],
    gamma: float = 0.8,
    max_hops: int = 2,
) -> float:
    """Compute s_graph(f) with coref edge propagation.

    s_graph(f) = max over f' in {f} ∪ N_k(f) of γ^d(f,f') · s_flat(f', m)
    """
    adj = _build_adjacency(claim_graph)

    # BFS from claim up to max_hops
    best_score = 0.0
    visited = {claim.fact_id: 0}  # fact_id -> distance
    queue = [(claim.fact_id, 0)]

    fact_by_id = {f.fact_id: f for f in claim_graph.facts}

    while queue:
        current_id, dist = queue.pop(0)
        current_fact = fact_by_id.get(current_id)
        if current_fact is None:
            continue

        # Compute flat similarity for this neighbor
        for obs in obs_facts_in_modality:
            sim = _flat_similarity(current_fact, obs)
            propagated = (gamma ** dist) * sim
            best_score = max(best_score, propagated)

        # Expand neighbors
        if dist < max_hops:
            for neighbor_id, etype in adj.get(current_id, []):
                if neighbor_id not in visited:
                    visited[neighbor_id] = dist + 1
                    queue.append((neighbor_id, dist + 1))

    return best_score


# ---------------------------------------------------------------------------
# Predicate dictionaries used by conflict rules C/D/E
# ---------------------------------------------------------------------------

# Asymmetric (directional) predicates: swapping subject/object flips meaning.
_ASYMMETRIC_PREDS = {
    "on", "above", "below", "under", "underneath", "beneath",
    "left of", "right of", "in front of", "behind",
    "holding", "carrying", "wearing", "riding", "pushing",
    "pulling", "feeding", "kicking", "throwing",
    "inside", "contains", "owns", "has",
}

# Attribute-style predicates: object slot is the attribute value (color/size/...).
_ATTRIBUTE_PREDS = {
    "is", "is a", "are", "has color", "has size", "has shape",
    "looks", "appears", "color", "size", "material",
}


def _norm_pred(p: str) -> str:
    return (p or "").lower().strip()


# ---------------------------------------------------------------------------
# Cross-graph entailment (Step 4) + Risk (Step 5)
# ---------------------------------------------------------------------------

def compare_fact_to_observations(
    claim: Fact,
    observation_graph: ObservationGraph,
    claim_graph: ClaimGraph,
    lambda_conflict: float = 0.5,
    nu_inconsistency: float = 0.1,
    gamma: float = 0.95,
) -> FactAssessment:
    """Compute continuous support/conflict scores and risk for a claim fact.

    No discrete labels. Directly outputs:
      s(f,m) ∈ [0,1]: semantic support score (via coref-propagated similarity)
      c(f,m) ∈ [0,1]: conflict score (rule-based detection)
      s + c ≤ 1 enforced

    Risk: r(f) = (1 - s̄(f)) + λ·c̄(f) + ν·δ(f)
    """
    support_by_modality: Dict[str, float] = {}
    conflict_by_modality: Dict[str, float] = {}

    grouped = observation_graph.by_modality()
    for modality, obs_facts in grouped.items():
        # s(f,m): graph-propagated semantic support score
        s = _graph_propagated_support(
            claim, claim_graph, obs_facts, gamma=gamma
        )

        # c(f,m): rule-based conflict score. Five rules:
        #   A: numeric mismatch on same entity
        #   B: same subject, semantically different object (legacy)
        #   C: same subject + same object, different predicate (relation-type error)
        #   D: subject/object swapped, asymmetric predicate (relation-direction error)
        #   E: attribute predicate + different object (explicit attribute conflict)
        c = 0.0
        claim_text = claim.text()
        claim_nums = _numeric_tokens(claim_text)
        claim_subj_norm = normalize_text(claim.subject)
        claim_obj_norm = normalize_text(claim.obj)
        claim_pred_n = _norm_pred(claim.predicate)
        is_attribute = claim_pred_n in _ATTRIBUTE_PREDS
        is_asymmetric = claim_pred_n in _ASYMMETRIC_PREDS

        if claim.predicate != "is_a":  # skip NER structural annotations
            for obs in obs_facts:
                if obs.predicate == "is_a":
                    continue
                obs_subj_norm = normalize_text(obs.subject)
                obs_obj_norm = normalize_text(obs.obj)
                obs_pred_n = _norm_pred(obs.predicate)
                obs_nums = _numeric_tokens(obs.text())

                subj_sim = (_semantic_similarity(claim_subj_norm, obs_subj_norm)
                            if claim_subj_norm and obs_subj_norm else 0.0)
                obj_sim = (_semantic_similarity(claim_obj_norm, obs_obj_norm)
                           if claim_obj_norm and obs_obj_norm else 0.0)
                pred_sim = (_semantic_similarity(claim_pred_n, obs_pred_n)
                            if claim_pred_n and obs_pred_n else 0.0)

                # Rule A — numerical contradiction on same entity
                if claim_nums and obs_nums and claim_nums != obs_nums and subj_sim > 0.4:
                    c = max(c, 0.8 * subj_sim)

                # Rule B — same subject, semantically different object (legacy)
                if subj_sim > 0.5 and claim.obj and obs.obj and obj_sim < 0.3:
                    c = max(c, subj_sim * (1.0 - obj_sim))

                # Rule C — predicate conflict: subject & object align, predicate differs
                if (subj_sim > 0.5 and obj_sim > 0.5
                        and claim.predicate and obs.predicate
                        and pred_sim < 0.4):
                    c = max(c, min(subj_sim, obj_sim) * (1.0 - pred_sim))

                # Rule D — directional reversal: claim's subject ↔ obs's object,
                # claim's object ↔ obs's subject, predicate matches, predicate is asymmetric.
                if is_asymmetric and obs_pred_n in _ASYMMETRIC_PREDS:
                    cross_subj_obj = (_semantic_similarity(claim_subj_norm, obs_obj_norm)
                                      if claim_subj_norm and obs_obj_norm else 0.0)
                    cross_obj_subj = (_semantic_similarity(claim_obj_norm, obs_subj_norm)
                                      if claim_obj_norm and obs_subj_norm else 0.0)
                    if (cross_subj_obj > 0.7 and cross_obj_subj > 0.7
                            and pred_sim > 0.5):
                        c = max(c, 0.9 * min(cross_subj_obj, cross_obj_subj))

                # Rule E — explicit attribute conflict (color/size/...)
                if is_attribute and subj_sim > 0.5 and obs_pred_n in _ATTRIBUTE_PREDS:
                    if claim.obj and obs.obj and obj_sim < 0.5:
                        c = max(c, subj_sim * (1.0 - obj_sim))

        # Enforce s + c <= 1. When a strong contradiction fires (c large),
        # the support score must reflect it — clipping c by s (the previous
        # behavior) silently suppressed Rule A/D signals whenever the field-level
        # support happened to be high. Instead, let conflict take precedence:
        # clip s by c, then clip any tiny residual c by the remaining headroom.
        if c >= 0.3:
            s = min(s, 1.0 - c)
        c = min(c, 1.0 - s)

        support_by_modality[modality] = float(s)
        conflict_by_modality[modality] = float(c)

    if not support_by_modality:
        support_by_modality = {"text": 0.0}
        conflict_by_modality = {"text": 0.0}

    # Derive labels for FSR/UFR/CR reporting (not used in risk)
    status_by_modality: Dict[str, FactStatus] = {}
    for mod in support_by_modality:
        s_val = support_by_modality[mod]
        c_val = conflict_by_modality[mod]
        if s_val >= 0.5:
            status_by_modality[mod] = FactStatus.SUPPORTED
        elif c_val >= 0.3:
            status_by_modality[mod] = FactStatus.CONTRADICTED
        elif s_val >= 0.2:
            status_by_modality[mod] = FactStatus.ENTAILED
        else:
            status_by_modality[mod] = FactStatus.UNKNOWN

    # Risk: r(f) = (1 - s̄) + λ·c̄ + ν·δ
    s_bar = sum(support_by_modality.values()) / max(1, len(support_by_modality))
    c_bar = max(conflict_by_modality.values())

    support_vals = list(support_by_modality.values())
    delta = (max(support_vals) - min(support_vals)) if len(support_vals) >= 2 else 0.0

    risk = (1.0 - s_bar) + lambda_conflict * c_bar + nu_inconsistency * delta

    return FactAssessment(
        fact=claim,
        status_by_modality=status_by_modality,
        support_by_modality=support_by_modality,
        conflict_by_modality=conflict_by_modality,
        risk=float(risk),
    )


def assess_candidate(
    claim_graph: ClaimGraph,
    observation_graph: ObservationGraph,
    lambda_conflict: float = 0.5,
    nu_inconsistency: float = 0.1,
    gamma: float = 0.95,
) -> Tuple[List[FactAssessment], float]:
    """Assess all claims against observations. Returns (assessments, total_risk).

    Risk(G_y|G_x) = Σ r(f), w_f = 1.0 for all f.
    r(f) = (1 - s̄) + λ·c̄ + ν·δ  (no μ term)
    """
    assessments: List[FactAssessment] = []
    total_risk = 0.0
    for fact in claim_graph.facts:
        assessment = compare_fact_to_observations(
            fact, observation_graph, claim_graph,
            lambda_conflict=lambda_conflict,
            nu_inconsistency=nu_inconsistency,
            gamma=gamma,
        )
        total_risk += assessment.risk
        assessments.append(assessment)
    return assessments, float(total_risk)


def assess_candidate_flat(
    claim_graph: ClaimGraph,
    observation_graph: ObservationGraph,
    lambda_conflict: float = 0.5,
    nu_inconsistency: float = 0.1,
) -> Tuple[List[FactAssessment], float]:
    """Flat ablation: same risk formula but NO coref propagation.

    r(f) = (1 - s̄) + λ·c̄ + ν·δ
    s = max flat_similarity(f, g) for g in G_x  (no graph propagation)

    Isolates graph structure contribution: only difference from full method
    is that coref edges are ignored for support propagation.
    """
    assessments: List[FactAssessment] = []
    total_risk = 0.0

    grouped = observation_graph.by_modality()

    for claim in claim_graph.facts:
        support_by_modality: Dict[str, float] = {}
        conflict_by_modality: Dict[str, float] = {}

        for modality, obs_facts in grouped.items():
            # Flat similarity — NO graph propagation
            s = 0.0
            for obs in obs_facts:
                s = max(s, _flat_similarity(claim, obs))

            # Conflict (same rules as graph method)
            c = 0.0
            if claim.predicate != "is_a":
                for obs in obs_facts:
                    if obs.predicate == "is_a":
                        continue
                    sim = _flat_similarity(claim, obs)
                    claim_subj = normalize_text(claim.subject)
                    obs_subj = normalize_text(obs.subject)
                    subj_sim = _semantic_similarity(claim_subj, obs_subj) if claim_subj and obs_subj else 0.0
                    claim_nums = _numeric_tokens(claim.text())
                    obs_nums = _numeric_tokens(obs.text())
                    if claim_nums and obs_nums and claim_nums != obs_nums and subj_sim > 0.4:
                        c = max(c, 0.8 * subj_sim)
                    if subj_sim > 0.5 and claim.obj and obs.obj:
                        obj_sim = _semantic_similarity(normalize_text(claim.obj), normalize_text(obs.obj))
                        if obj_sim < 0.3:
                            c = max(c, subj_sim * (1.0 - obj_sim))

            c = min(c, 1.0 - s)  # enforce s + c <= 1

            support_by_modality[modality] = float(s)
            conflict_by_modality[modality] = float(c)

        if not support_by_modality:
            support_by_modality = {"text": 0.0}
            conflict_by_modality = {"text": 0.0}

        # Derived labels for reporting
        status_by_modality: Dict[str, FactStatus] = {}
        for mod in support_by_modality:
            sv = support_by_modality[mod]
            cv = conflict_by_modality[mod]
            if sv >= 0.5: status_by_modality[mod] = FactStatus.SUPPORTED
            elif cv >= 0.3: status_by_modality[mod] = FactStatus.CONTRADICTED
            elif sv >= 0.2: status_by_modality[mod] = FactStatus.ENTAILED
            else: status_by_modality[mod] = FactStatus.UNKNOWN

        s_bar = sum(support_by_modality.values()) / max(1, len(support_by_modality))
        c_bar = max(conflict_by_modality.values())
        support_vals = list(support_by_modality.values())
        delta = (max(support_vals) - min(support_vals)) if len(support_vals) >= 2 else 0.0

        risk = (1.0 - s_bar) + lambda_conflict * c_bar + nu_inconsistency * delta

        assessment = FactAssessment(
            fact=claim,
            status_by_modality=status_by_modality,
            support_by_modality=support_by_modality,
            conflict_by_modality=conflict_by_modality,
            risk=float(risk),
        )
        total_risk += assessment.risk
        assessments.append(assessment)

    return assessments, float(total_risk)


# ---------------------------------------------------------------------------
# Summary helpers
# ---------------------------------------------------------------------------

def summarize_assessments(assessments: Sequence[FactAssessment]) -> Dict[str, float]:
    if not assessments:
        return {"fsr": 0.0, "ufr": 0.0, "cr": 0.0}
    supported_or_entailed = 0
    all_unknown = 0
    contradicted = 0
    total = len(assessments)
    for item in assessments:
        statuses = list(item.status_by_modality.values())
        if any(status in {FactStatus.SUPPORTED, FactStatus.ENTAILED} for status in statuses):
            supported_or_entailed += 1
        if all(status == FactStatus.UNKNOWN for status in statuses):
            all_unknown += 1
        if any(status == FactStatus.CONTRADICTED for status in statuses):
            contradicted += 1
    return {
        "fsr": supported_or_entailed / total,
        "ufr": all_unknown / total,
        "cr": contradicted / total,
    }
