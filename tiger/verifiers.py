"""External verifiers for baseline comparisons.

B3: VisualPRM-style scoring via CLIP/SigLIP similarity.
B4: CycleReward-style scoring via round-trip text consistency.
FlatBaseline: Ablation — embedding similarity without graph structure.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .schemas import CandidateOutput, UnifiedSample
from .utils import normalize_text


# ---------------------------------------------------------------------------
# B3: CLIP-based verifier (proxy for VisualPRM)
# ---------------------------------------------------------------------------

class CLIPVerifier:
    """Score candidates by CLIP similarity between input and generated text.

    For image+text tasks: measures alignment between source image and output text.
    For text-only / audio tasks: falls back to text overlap heuristic.
    """

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        self.model_name = model_name
        self._model = None
        self._processor = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import CLIPModel, CLIPProcessor
        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        # Try GPU first, fall back to CPU if CUDA is busy/unavailable
        for device in ("cuda", "cpu"):
            if device == "cuda" and not torch.cuda.is_available():
                continue
            try:
                self._model = CLIPModel.from_pretrained(self.model_name).to(device)
                self._model.eval()
                self._device = device
                return
            except Exception:
                continue
        raise ImportError(f"CLIP model loading failed on all devices.")

    def score(self, sample: UnifiedSample, candidate: CandidateOutput) -> float:
        """Score a candidate output against the input. Higher = better alignment."""
        text = candidate.display_text()
        if not text:
            return 0.0

        # If we have images, use CLIP image-text similarity
        if sample.image_paths:
            return self._score_image_text(sample.image_paths[0], text)

        # Fallback: text overlap with prompt (simple proxy)
        return self._score_text_overlap(sample.prompt, text)

    def _score_image_text(self, image_path: str, text: str) -> float:
        """CLIP similarity between image and text."""
        self._load()
        try:
            import torch
            from PIL import Image
            image = Image.open(image_path).convert("RGB")
            inputs = self._processor(
                text=[text[:77]],  # CLIP max token length
                images=image,
                return_tensors="pt",
                padding=True,
                truncation=True,
            ).to(self._device)
            with torch.no_grad():
                outputs = self._model(**inputs)
                # Normalized similarity in [0, 1]
                score = outputs.logits_per_image.item() / 100.0
                return max(0.0, min(1.0, score))
        except Exception:
            return self._score_text_overlap("", text)

    def _score_text_overlap(self, reference: str, prediction: str) -> float:
        """Simple token-overlap score as fallback."""
        ref_tokens = set(normalize_text(reference).split())
        pred_tokens = set(normalize_text(prediction).split())
        if not ref_tokens or not pred_tokens:
            return 0.0
        intersection = ref_tokens & pred_tokens
        return len(intersection) / max(len(ref_tokens | pred_tokens), 1)


# ---------------------------------------------------------------------------
# B4: Cycle consistency verifier (proxy for CycleReward)
# ---------------------------------------------------------------------------

class CycleVerifier:
    """Score candidates by round-trip cycle consistency.

    For each candidate output, ask the backbone to regenerate a description
    of the original input from the output. Then measure text similarity
    between the cycle-generated description and the original prompt/evidence.
    """

    def __init__(self, backbone, generation_settings=None):
        from .models import BaseBackbone, GenerationSettings
        self.backbone: BaseBackbone = backbone
        self.settings = generation_settings or GenerationSettings(
            max_new_tokens=128, temperature=0.0, do_sample=False,
        )

    def score(self, sample: UnifiedSample, candidate: CandidateOutput) -> float:
        """Cycle consistency score. Higher = more consistent."""
        output_text = candidate.display_text()
        if not output_text:
            return 0.0

        # Step 1: Generate reverse description from the candidate output
        cycle_prompt = (
            "Based on the following output, describe what the original input "
            "must have contained. Be specific about entities, quantities, "
            "and spatial relations.\n\n"
            f"Output: {output_text}\n\n"
            "Original input description:"
        )
        cycle_output = self.backbone.generate(sample, cycle_prompt, self.settings)
        cycle_text = cycle_output.text or ""

        # Step 2: Measure similarity between cycle text and original prompt
        return self._text_similarity(sample.prompt, cycle_text)

    def _text_similarity(self, text_a: str, text_b: str) -> float:
        """Compute similarity between two texts using token F1."""
        tokens_a = set(normalize_text(text_a).split())
        tokens_b = set(normalize_text(text_b).split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        precision = len(intersection) / max(len(tokens_b), 1)
        recall = len(intersection) / max(len(tokens_a), 1)
        if precision + recall == 0:
            return 0.0
        f1 = 2 * precision * recall / (precision + recall)
        return f1


# ---------------------------------------------------------------------------
# Ablation: FlatBaseline — embedding similarity, no graph structure
# ---------------------------------------------------------------------------

class FlatBaseline:
    """Replace graph-based entailment with flat embedding similarity.

    Uses sentence-transformers to compute cosine similarity between
    each claim fact and all observation facts independently.
    No coref edge propagation — isolates the contribution of graph structure.
    """

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._model = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from sentence_transformers import SentenceTransformer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model = SentenceTransformer(self.model_name, device=device)

    def compute_support_score(self, claim_text: str, obs_texts: list) -> float:
        """Max cosine similarity between claim and any observation fact."""
        if not obs_texts:
            return 0.0
        self._load()
        import numpy as np
        all_texts = [claim_text] + obs_texts
        embeddings = self._model.encode(all_texts, normalize_embeddings=True)
        claim_emb = embeddings[0]
        obs_embs = embeddings[1:]
        sims = np.dot(obs_embs, claim_emb)
        return float(np.max(sims))

    def compute_risk(self, claim_text: str, obs_texts: list) -> float:
        """Simplified risk: 1 - max_similarity."""
        s = self.compute_support_score(claim_text, obs_texts)
        return 1.0 - s
