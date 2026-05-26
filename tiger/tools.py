"""Deterministic perceptual tools for fact extraction.

v4: Image  — Grounding DINO (open-vocabulary detection, attributes)
    Audio  — Whisper-medium (WER ~3.5%) + Stanza SVO
    Text   — Stanza (Stanford NLP, LAS=94%)
    All modalities output unified SVO format.
"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import Fact, ObservationGraph, ClaimGraph


# ---------------------------------------------------------------------------
# Lazy-loaded singletons
# ---------------------------------------------------------------------------

_gdino_model = None
_gdino_processor = None
_whisper_pipe = None
_stanza_nlp = None
_panns_tagger = None


def _get_device():
    """Get best available device, with fallback if GPU is busy."""
    import torch
    if not torch.cuda.is_available():
        return "cpu"
    try:
        # Test that GPU is actually usable (not just detected)
        torch.zeros(1, device="cuda")
        return "cuda"
    except Exception:
        return "cpu"


def _load_gdino():
    """Load Grounding DINO — open-vocabulary object detector, prefer GPU."""
    global _gdino_model, _gdino_processor
    if _gdino_model is not None:
        return
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    _gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-base")
    # Try GPU first, fall back to CPU if CUDA is busy
    for device in ("cuda", "cpu"):
        if device == "cuda" and not torch.cuda.is_available():
            continue
        try:
            _gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
                "IDEA-Research/grounding-dino-base",
            ).to(device).eval()
            return
        except Exception:
            continue
    raise RuntimeError("Failed to load Grounding DINO on any device")


def _load_whisper():
    """Load Whisper-medium, prefer GPU (WER ~3.5%)."""
    global _whisper_pipe
    if _whisper_pipe is not None:
        return
    import torch
    from transformers import pipeline
    device = _get_device()
    # transformers pipeline expects device index for cuda
    dev_arg = 0 if device == "cuda" else -1
    _whisper_pipe = pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-medium",
        chunk_length_s=30,
        device=dev_arg,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )


def _load_stanza():
    """Load Stanza, prefer GPU with fallback to CPU. Fully offline."""
    global _stanza_nlp
    if _stanza_nlp is not None:
        return
    import os
    # Fix OpenSSL MD5 issue on some cluster nodes
    os.environ["OPENSSL_CONF"] = "/dev/null"
    # Stanza: disable resource download (models already cached on login node)
    os.environ["STANZA_RESOURCES_DIR"] = os.path.expanduser("~/stanza_resources")
    import stanza
    device = _get_device()
    _stanza_nlp = stanza.Pipeline(
        "en",
        processors="tokenize,pos,lemma,depparse,ner",
        use_gpu=(device == "cuda"),
        logging_level="ERROR",
        download_method=None,  # Do NOT try to download anything
    )


# ===================================================================
# Helper: build full noun phrase from a head token (Stanza word)
# ===================================================================

def _full_np(head, words_by_id):
    """Collect the full noun phrase rooted at `head` — determiners,
    adjectives, compounds, nummod, possessives. Excludes case markers
    (prepositions) since those go into the relation."""
    deps = {w.id: w for w in words_by_id.values() if w.head == head.id}
    parts = []
    for w in sorted(deps.values(), key=lambda x: x.id):
        if w.deprel in ("det", "amod", "nummod", "compound", "nmod:poss",
                         "flat", "fixed"):
            parts.append(w.text)
    parts.append(head.text)
    return " ".join(parts)


# ===================================================================
# Text: Stanza-based extraction with recursive prep-chain traversal
# ===================================================================

def _extract_stanza(doc) -> List[Dict[str, Any]]:
    """Extract facts from a Stanza Document.

    Returns list of dicts with entity/relation/object/confidence keys.
    """
    triples: List[Dict[str, Any]] = []
    seen: set = set()

    for sent in doc.sentences:
        wid = {w.id: w for w in sent.words}

        # Named entities
        for ent in sent.ents:
            key = ("ner", ent.text, ent.type)
            if key not in seen:
                seen.add(key)
                triples.append({
                    "entity": ent.text,
                    "relation": "is_a",
                    "object": ent.type,
                    "confidence": 0.95,
                })

        # ---- Verb-centred extraction ----
        for verb in sent.words:
            if verb.upos not in ("VERB", "AUX"):
                continue

            # Gather subjects
            subjects = []
            for w in sent.words:
                if w.head == verb.id and w.deprel in (
                    "nsubj", "nsubj:pass", "csubj", "expl"
                ):
                    subjects.append(w)
                    # conjoined subjects
                    for cc in sent.words:
                        if cc.head == w.id and cc.deprel == "conj":
                            subjects.append(cc)
            if not subjects:
                continue

            vlemma = verb.lemma

            # Gather ALL objects / complements / preps under the verb
            obj_tuples = _collect_objects(verb, sent.words, wid)

            for subj_w in subjects:
                subj_text = _full_np(subj_w, wid)
                if not obj_tuples:
                    key = (subj_text, vlemma, "")
                    if key not in seen:
                        seen.add(key)
                        triples.append({
                            "entity": subj_text,
                            "relation": vlemma,
                            "object": "",
                            "confidence": 0.82,
                        })
                for obj_text, rel_text, conf in obj_tuples:
                    key = (subj_text, rel_text, obj_text)
                    if key not in seen:
                        seen.add(key)
                        triples.append({
                            "entity": subj_text,
                            "relation": rel_text,
                            "object": obj_text,
                            "confidence": conf,
                        })

        # ---- Copula / predicate adjective ----
        for w in sent.words:
            if w.deprel in ("acomp", "root") and w.upos == "ADJ":
                cop = None
                subj = None
                for c in sent.words:
                    if c.head == w.id and c.deprel == "cop":
                        cop = c
                    if c.head == w.id and c.deprel in ("nsubj", "nsubj:pass"):
                        subj = c
                if subj:
                    key = (_full_np(subj, wid), "is", w.text)
                    if key not in seen:
                        seen.add(key)
                        triples.append({
                            "entity": _full_np(subj, wid),
                            "relation": "is",
                            "object": w.text,
                            "confidence": 0.88,
                        })

    # ---- Participial clauses / fragments without explicit subject ----
    # E.g., "A dog and a cat sitting on a couch" — no finite verb
    for sent in doc.sentences:
        wid_s = {w.id: w for w in sent.words}
        has_finite_verb = any(
            w.upos == "VERB" and w.deprel in ("root",)
            and any(c.deprel in ("nsubj", "nsubj:pass")
                    for c in sent.words if c.head == w.id)
            for w in sent.words
        )
        if has_finite_verb:
            continue
        # Find the root (often a participle or noun)
        root = next((w for w in sent.words if w.deprel == "root"), None)
        if root is None:
            continue
        # Collect subject-like NPs (nsubj or the whole clause)
        root_subjs = [w for w in sent.words
                      if w.head == root.id and w.deprel in ("nsubj", "nsubj:pass")]
        if not root_subjs:
            # Use the root as subject placeholder
            root_subjs = [root]
        for subj_w in root_subjs:
            subj_text = _full_np(subj_w, wid_s)
            objs = _collect_objects(root, sent.words, wid_s)
            for obj_text, rel_text, conf in objs:
                key = (subj_text, rel_text, obj_text)
                if key not in seen:
                    seen.add(key)
                    triples.append({
                        "entity": subj_text,
                        "relation": rel_text,
                        "object": obj_text,
                        "confidence": conf * 0.9,
                    })

    # Fallback if nothing extracted
    if not triples:
        for sent in doc.sentences:
            triples.append({
                "entity": sent.text.strip(),
                "relation": "states",
                "object": "",
                "confidence": 0.70,
            })

    return triples


def _collect_objects(verb, all_words, wid) -> List[Tuple[str, str, float]]:
    """Recursively collect all objects/complements under a verb.

    Returns list of (object_text, relation_text, confidence).
    Handles: dobj, iobj, prep chains, xcomp, ccomp, advcl, conj.
    """
    results: List[Tuple[str, str, float]] = []
    vlemma = verb.lemma

    for w in all_words:
        if w.head != verb.id:
            continue

        # Direct / indirect objects
        if w.deprel in ("obj", "dobj", "iobj", "obl", "attr"):
            np_text = _full_np(w, wid)
            # Check for case marker (preposition)
            case = None
            for c in all_words:
                if c.head == w.id and c.deprel == "case":
                    case = c.text
            if case:
                results.append((np_text, f"{vlemma} {case}", 0.90))
            else:
                results.append((np_text, vlemma, 0.90))
            # Conjoined objects
            for cc in all_words:
                if cc.head == w.id and cc.deprel == "conj":
                    cc_np = _full_np(cc, wid)
                    if case:
                        results.append((cc_np, f"{vlemma} {case}", 0.88))
                    else:
                        results.append((cc_np, vlemma, 0.88))

            # Recursive: prep chains under this object
            # "sitting on a chair NEAR THE WINDOW"
            _collect_prep_chain(w, all_words, wid, vlemma, results)

        # oblique nominal with case: "on a chair", "near the window"
        if w.deprel in ("obl", "obl:npmod", "obl:tmod"):
            np_text = _full_np(w, wid)
            case = None
            for c in all_words:
                if c.head == w.id and c.deprel == "case":
                    case = c.text
            if case:
                results.append((np_text, f"{vlemma} {case}", 0.88))
            else:
                results.append((np_text, vlemma, 0.85))
            _collect_prep_chain(w, all_words, wid, vlemma, results)

        # Clausal complement: "taught her to play piano"
        if w.deprel in ("xcomp", "ccomp", "advcl"):
            sub_objs = _collect_objects(w, all_words, wid)
            for obj_text, sub_rel, conf in sub_objs:
                results.append((obj_text, f"{vlemma} {w.lemma}", conf * 0.95))
            if not sub_objs:
                results.append((w.lemma, f"{vlemma}", 0.80))

    return results


def _collect_prep_chain(head_w, all_words, wid, base_verb, results,
                        depth: int = 0, max_depth: int = 3):
    """Recursively follow prepositional modifiers hanging off a noun.

    E.g., "on a chair near the window beside the door"
       → (window, sit near, 0.85), (door, sit beside, 0.83)
    """
    if depth >= max_depth:
        return
    for w in all_words:
        if w.head != head_w.id:
            continue
        if w.deprel in ("nmod", "obl", "nmod:poss"):
            np_text = _full_np(w, wid)
            case = None
            for c in all_words:
                if c.head == w.id and c.deprel == "case":
                    case = c.text
            if case:
                conf = max(0.75, 0.90 - depth * 0.05)
                results.append((np_text, f"{base_verb} {case}", conf))
            _collect_prep_chain(w, all_words, wid, base_verb, results,
                                depth + 1, max_depth)


def run_dependency_parser(text: str) -> List[Dict[str, Any]]:
    """Parse text with Stanza and extract (subject, relation, object) triples.

    Uses Stanford NLP's dependency parser with recursive prep-chain
    traversal to capture all modifiers and arguments.
    """
    text = text.strip()
    if not text:
        return []
    _load_stanza()
    doc = _stanza_nlp(text)
    return _extract_stanza(doc)


# ===================================================================
# Image: DETR object detection + spatial relations + counts
# ===================================================================

def run_object_detector(image_path: str, threshold: float = 0.5, max_detections: int = 10) -> List[Dict[str, Any]]:
    """Run Grounding DINO on an image with a broad open-vocabulary query.

    Unlike DETR (91 fixed classes), Grounding DINO can detect ANY object
    described in text. We use a comprehensive query covering common
    visual entities, attributes, and scene elements.
    """
    _load_gdino()
    import torch
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    img_w, img_h = image.size

    # Broad query covering common visual elements
    # Grounding DINO matches these phrases against image regions
    query = (
        "person. man. woman. child. face. hand. "
        "car. truck. bus. bicycle. motorcycle. "
        "dog. cat. bird. horse. "
        "chair. table. couch. bed. desk. "
        "tree. building. house. window. door. wall. "
        "food. bottle. cup. plate. "
        "phone. laptop. book. bag. "
        "text. sign. number."
    )

    inputs = _gdino_processor(images=image, text=query, return_tensors="pt")
    inputs = {k: v.to(_gdino_model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = _gdino_model(**inputs)

    results = _gdino_processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        threshold=threshold,
        text_threshold=threshold,
        target_sizes=[(img_h, img_w)],
    )[0]

    objects = []
    for score, label, box in zip(
        results["scores"].tolist(),
        results["text_labels"],
        results["boxes"].tolist(),
    ):
        x1, y1, x2, y2 = box
        objects.append({
            "entity": label.strip(),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
            "confidence": round(score, 4),
        })

    # Limit to top-N detections by confidence to avoid noisy Gx
    objects.sort(key=lambda x: x["confidence"], reverse=True)
    objects = objects[:max_detections]

    detections: List[Dict[str, Any]] = []

    for obj in objects:
        # Existence fact
        detections.append({
            "entity": obj["entity"],
            "relation": "exists_in",
            "object": "image",
            "bbox": obj["bbox"],
            "confidence": obj["confidence"],
        })
        # Position fact
        cx = (obj["bbox"][0] + obj["bbox"][2]) / 2 / max(img_w, 1)
        cy = (obj["bbox"][1] + obj["bbox"][3]) / 2 / max(img_h, 1)
        region = _get_spatial_region(cx, cy)
        detections.append({
            "entity": obj["entity"],
            "relation": "located_in",
            "object": region,
            "bbox": obj["bbox"],
            "confidence": obj["confidence"],
        })

    # Spatial relations between pairs
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            rel = _compute_spatial_relation(objects[i]["bbox"], objects[j]["bbox"])
            if rel:
                detections.append({
                    "entity": objects[i]["entity"],
                    "relation": rel,
                    "object": objects[j]["entity"],
                    "confidence": min(objects[i]["confidence"],
                                      objects[j]["confidence"]) * 0.9,
                })

    # Count facts
    entity_counts: Dict[str, int] = {}
    for obj in objects:
        entity_counts[obj["entity"]] = entity_counts.get(obj["entity"], 0) + 1
    for entity, count in entity_counts.items():
        if count > 1:
            detections.append({
                "entity": f"{count} {entity}s",
                "relation": "count",
                "object": str(count),
                "confidence": 0.85,
            })

    return detections


def _get_spatial_region(cx: float, cy: float) -> str:
    if cy < 0.33:
        v = "top"
    elif cy > 0.67:
        v = "bottom"
    else:
        v = "center"
    if cx < 0.33:
        h = "left"
    elif cx > 0.67:
        h = "right"
    else:
        h = "center"
    if v == "center" and h == "center":
        return "center"
    return f"{v}-{h}"


def _compute_spatial_relation(bbox_a: List[float], bbox_b: List[float]) -> Optional[str]:
    ax1, ay1, ax2, ay2 = bbox_a
    bx1, by1, bx2, by2 = bbox_b
    acx, acy = (ax1 + ax2) / 2, (ay1 + ay2) / 2
    bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2

    dx = bcx - acx
    dy = bcy - acy
    diag_a = ((ax2 - ax1) ** 2 + (ay2 - ay1) ** 2) ** 0.5
    diag_b = ((bx2 - bx1) ** 2 + (by2 - by1) ** 2) ** 0.5
    min_dist = (diag_a + diag_b) / 4
    dist = (dx ** 2 + dy ** 2) ** 0.5
    if dist < min_dist * 0.3:
        return "overlaps"
    if abs(dx) > abs(dy) * 1.5:
        return "left_of" if dx > 0 else "right_of"
    if abs(dy) > abs(dx) * 1.5:
        return "above" if dy > 0 else "below"
    return "near"


# ===================================================================
# Audio: Whisper ASR → Stanza SVO (unified format with text)
# ===================================================================

def _load_audio_array(audio_path: str, target_sr: int = 16000):
    """Load audio file as numpy array. Tries multiple backends."""
    import numpy as np
    # Try librosa first (most robust, handles many formats)
    try:
        import librosa
        audio, sr = librosa.load(audio_path, sr=target_sr, mono=True)
        return {"array": audio, "sampling_rate": sr}
    except Exception:
        pass
    # Fallback: soundfile
    try:
        import soundfile as sf
        audio, sr = sf.read(audio_path, dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # stereo to mono
        if sr != target_sr:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
            sr = target_sr
        return {"array": audio, "sampling_rate": sr}
    except Exception:
        pass
    # Last resort: return silence (don't crash the whole pipeline)
    return {"array": np.zeros(target_sr, dtype=np.float32), "sampling_rate": target_sr}


def _load_panns():
    """Load PANNs audio tagging model (AudioSet 527 classes)."""
    global _panns_tagger
    if _panns_tagger is not None:
        return
    from panns_inference import AudioTagging
    device = _get_device()
    _panns_tagger = AudioTagging(checkpoint_path=None, device=device)


def run_panns_event_detection(audio_path: str, top_k: int = 10, threshold: float = 0.15) -> List[Dict[str, Any]]:
    """Run PANNs to detect acoustic events in audio.

    Returns top-k AudioSet event labels above threshold.
    Essential for environmental sounds (Clotho) where Whisper produces nothing.
    """
    _load_panns()
    import numpy as np
    import librosa

    audio, sr = librosa.load(audio_path, sr=32000, mono=True)
    audio = audio[np.newaxis, :]  # (1, samples)

    clipwise_output, _ = _panns_tagger.inference(audio)
    probs = clipwise_output[0]  # (527,)

    # Get AudioSet label names
    import csv
    import os
    labels_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "..", "..", "data", "audioset_labels.csv")
    if os.path.exists(labels_path):
        with open(labels_path) as f:
            reader = csv.reader(f)
            labels = [row[2] for row in reader if len(row) > 2]
    else:
        # Fallback: use index as label
        labels = [f"event_{i}" for i in range(527)]

    # Sort by probability, take top_k above threshold
    sorted_indices = np.argsort(probs)[::-1]
    events = []
    for idx in sorted_indices[:top_k]:
        prob = float(probs[idx])
        if prob < threshold:
            break
        label = labels[idx] if idx < len(labels) else f"event_{idx}"
        events.append({
            "entity": label,
            "relation": "exists_in",
            "object": "audio",
            "confidence": round(prob, 4),
            "timestamp": 0.0,
        })

    return events


def run_asr_and_acoustic_analysis(audio_path: str) -> List[Dict[str, Any]]:
    """Complete audio analysis: PANNs events + Whisper ASR + Stanza SVO.

    Pipeline:
    1. PANNs → acoustic event labels (bird, water, siren, etc.)
    2. Whisper → speech transcript (if any speech present)
    3. Stanza → SVO triples from transcript
    4. Merge all into unified fact list

    This handles both environmental sounds (Clotho) and speech (LibriTTS).
    """
    detections: List[Dict[str, Any]] = []

    # --- Step 1: PANNs acoustic event detection ---
    try:
        panns_events = run_panns_event_detection(audio_path, top_k=10, threshold=0.15)
        detections.extend(panns_events)
    except Exception:
        pass  # PANNs failure is non-fatal

    # --- Step 2: Whisper ASR ---
    _load_whisper()
    audio_input = _load_audio_array(audio_path, target_sr=16000)
    result = _whisper_pipe(audio_input, return_timestamps=True)
    full_text = (result.get("text") or "").strip()

    # --- Step 3: Stanza SVO from transcript (if speech found) ---
    if full_text and len(full_text) > 5:
        text_facts = run_dependency_parser(full_text)
        # Tag with timestamps
        chunks = result.get("chunks", [])
        for det in text_facts:
            det["timestamp"] = 0.0
            entity_lower = det.get("entity", "").lower()
            for chunk in chunks:
                chunk_text = (chunk.get("text") or "").lower()
                if entity_lower in chunk_text:
                    ts = chunk.get("timestamp", (0.0, None))
                    det["timestamp"] = ts[0] if ts and ts[0] is not None else 0.0
                    break
        detections.extend(text_facts)

        # Full transcript as context
        detections.append({
            "entity": full_text,
            "relation": "transcript",
            "object": "",
            "confidence": 0.92,
            "timestamp": 0.0,
        })

    return detections


# ===================================================================
# Convert raw detections → Fact list with coref edges
# ===================================================================

def detections_to_facts(
    detections: List[Dict[str, Any]],
    modality: str,
) -> Tuple[List[Fact], List[Tuple[str, str, str]]]:
    """Convert raw tool detections into Fact objects and coref edges."""
    facts: List[Fact] = []
    edges: List[Tuple[str, str, str]] = []

    for idx, det in enumerate(detections):
        fid = f"f{idx}"
        entity = det.get("entity", "")
        relation = det.get("relation", "detected")
        obj = det.get("object", "")

        bbox = det.get("bbox")
        bbox_tuple = tuple(bbox) if isinstance(bbox, (list, tuple)) and len(bbox) == 4 else None
        ts = det.get("timestamp")
        time_range = (ts, ts) if isinstance(ts, (int, float)) else None

        facts.append(Fact(
            fact_id=fid,
            subject=str(entity),
            predicate=str(relation),
            obj=str(obj),
            modality=modality,
            bbox=bbox_tuple,
            time_range=time_range,
            confidence=float(det.get("confidence", 1.0)),
        ))

    _build_coref_edges(facts, edges)

    # Spatial edges for overlapping bboxes
    bbox_facts = [(f.fact_id, f.bbox) for f in facts if f.bbox is not None]
    for i in range(len(bbox_facts)):
        for j in range(i + 1, len(bbox_facts)):
            fid_a, bb_a = bbox_facts[i]
            fid_b, bb_b = bbox_facts[j]
            ix1 = max(bb_a[0], bb_b[0])
            iy1 = max(bb_a[1], bb_b[1])
            ix2 = min(bb_a[2], bb_b[2])
            iy2 = min(bb_a[3], bb_b[3])
            if ix2 > ix1 and iy2 > iy1:
                edges.append((fid_a, fid_b, "spatial"))

    # Temporal edges for nearby audio facts
    time_facts = [(f.fact_id, f.time_range[0]) for f in facts
                  if f.time_range is not None]
    time_facts.sort(key=lambda x: x[1])
    for i in range(len(time_facts) - 1):
        fid_a, t_a = time_facts[i]
        fid_b, t_b = time_facts[i + 1]
        if abs(t_b - t_a) < 5.0:
            edges.append((fid_a, fid_b, "temporal"))

    return facts, edges


def _build_coref_edges(facts: List[Fact], edges: List[Tuple[str, str, str]]):
    """Build coref edges with fuzzy entity matching.

    Connects facts that share overlapping entity names:
    'her daughter' ↔ 'daughter', 'red truck' ↔ 'truck'.
    """
    for i in range(len(facts)):
        for j in range(i + 1, len(facts)):
            si = facts[i].subject.lower().strip()
            sj = facts[j].subject.lower().strip()
            if not si or not sj or len(si) < 2 or len(sj) < 2:
                continue
            # Exact match
            if si == sj:
                edges.append((facts[i].fact_id, facts[j].fact_id, "coref"))
                continue
            # Containment: "her daughter" ↔ "daughter"
            if si in sj or sj in si:
                edges.append((facts[i].fact_id, facts[j].fact_id, "coref"))
                continue
            # Word overlap: share at least one content word (>2 chars)
            words_i = {w for w in si.split() if len(w) > 2}
            words_j = {w for w in sj.split() if len(w) > 2}
            if words_i and words_j and words_i & words_j:
                edges.append((facts[i].fact_id, facts[j].fact_id, "coref"))


# ===================================================================
# High-level: extract graph from any modality
# ===================================================================

def extract_graph_with_tools(
    content: Any,
    modality: str,
) -> Tuple[List[Fact], List[Tuple[str, str, str]]]:
    """Run deterministic tools and return (facts, edges)."""
    if modality == "image":
        detections = run_object_detector(content)
        return detections_to_facts(detections, modality="image")
    elif modality == "audio":
        detections = run_asr_and_acoustic_analysis(content)
        return detections_to_facts(detections, modality="audio")
    elif modality == "text":
        detections = run_dependency_parser(content)
        return detections_to_facts(detections, modality="text")
    elif modality == "layout":
        from .evidence import parse_layout_facts
        facts = parse_layout_facts(content) if isinstance(content, dict) else []
        return facts, []
    else:
        detections = run_dependency_parser(str(content))
        return detections_to_facts(detections, modality="text")
