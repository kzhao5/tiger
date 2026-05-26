"""TIGER inference pipeline.

New algorithm: no critic. Repair uses backbone directly with K-sample
selection. Graph extraction uses deterministic tools.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from .evidence import FactExtractor, assess_candidate
from .metrics import evidence_metrics as compute_evidence_metrics
from .metrics import task_metrics
from .models import BaseBackbone, GenerationSettings
from .planner import PlannerConfig
from .schemas import (
    CandidateOutput,
    ClaimGraph,
    InferenceOutput,
    ObservationGraph,
    PlannerState,
    UnifiedSample,
)
from .utils import try_extract_json
from .verifiers import CLIPVerifier, CycleVerifier


class TigerInferenceEngine:
    def __init__(
        self,
        backbone: BaseBackbone,
        planner_config: Optional[PlannerConfig] = None,
        generation_settings: Optional[GenerationSettings] = None,
        k_repair: int = 1,
        lambda_conflict: float = 1.5,
        nu_inconsistency: float = 0.1,
        gamma: float = 0.8,
        alpha_batch: float = 0.0,
        use_llm_extract: bool = True,
    ):
        self.backbone = backbone
        self.planner = planner_config or PlannerConfig()
        self.generation_settings = generation_settings or GenerationSettings()

        # New algorithm hyperparameters
        self.k_repair = k_repair
        self.lambda_conflict = lambda_conflict
        self.nu_inconsistency = nu_inconsistency
        self.gamma = gamma
        self.alpha_batch = alpha_batch  # batch repair: fix α·|G_x| facts per round
        self.use_llm_extract = use_llm_extract

        self.fact_extractor = FactExtractor(backbone, use_llm=use_llm_extract)
        self._clip_verifier: Optional[CLIPVerifier] = None
        self._cycle_verifier: Optional[CycleVerifier] = None

    def _initial_generate(self, sample: UnifiedSample, greedy: bool = False) -> CandidateOutput:
        """Generate initial candidate. Use greedy=True for TIGER mode
        (best quality initial output), greedy=False for Best-of-N (diversity)."""
        if greedy:
            greedy_settings = GenerationSettings(
                max_new_tokens=self.generation_settings.max_new_tokens,
                temperature=0.0,
                top_p=1.0,
                do_sample=False,
                return_audio=self.generation_settings.return_audio,
                voice=self.generation_settings.voice,
                num_candidates=1,
            )
            return self.backbone.generate(sample, sample.prompt, greedy_settings)
        return self.backbone.generate(sample, sample.prompt, self.generation_settings)

    def _parse_layout_if_needed(
        self, sample: UnifiedSample, candidate: CandidateOutput
    ) -> CandidateOutput:
        if sample.task_type.value.endswith("layout") and candidate.text and candidate.layout is None:
            try:
                parsed = try_extract_json(candidate.text)
                if isinstance(parsed, dict):
                    candidate.layout = parsed
            except Exception:
                pass
        return candidate

    def _assess(
        self, claim_graph: ClaimGraph, observation_graph: ObservationGraph
    ):
        return assess_candidate(
            claim_graph, observation_graph,
            lambda_conflict=self.lambda_conflict,
            nu_inconsistency=self.nu_inconsistency,
            gamma=self.gamma,
        )

    # ------------------------------------------------------------------
    # Backbone-direct repair (replaces Critic)
    # ------------------------------------------------------------------

    # Lazy-loaded semantic model for repair
    _sem_model = None

    @staticmethod
    def _get_sem_model():
        if TigerInferenceEngine._sem_model is None:
            from sentence_transformers import SentenceTransformer
            TigerInferenceEngine._sem_model = SentenceTransformer(
                'sentence-transformers/all-MiniLM-L6-v2')
        return TigerInferenceEngine._sem_model

    def _semantic_sim(self, text_a: str, text_b: str) -> float:
        """Semantic similarity between two texts (cosine of embeddings)."""
        if not text_a or not text_b:
            return 0.0
        model = self._get_sem_model()
        embs = model.encode([text_a, text_b], normalize_embeddings=True)
        return float(embs[0] @ embs[1])

    def _find_worst_fact_semantic(self, assessments, observation_graph):
        """Find highest-risk fact using SEMANTIC similarity with G_x.

        Unlike jaccard-based risk (which gives 90% UNKNOWN), semantic
        similarity actually identifies which facts LACK support vs which
        are grounded. This makes f* selection much more accurate.
        """
        if not assessments:
            return None

        obs_texts = [f.text() for f in observation_graph.facts]
        if not obs_texts:
            return max(assessments, key=lambda a: a.risk)

        model = self._get_sem_model()
        obs_embs = model.encode(obs_texts, normalize_embeddings=True)

        best_worst = None
        best_semantic_risk = -1.0

        for a in assessments:
            claim_text = a.fact.text()
            claim_emb = model.encode([claim_text], normalize_embeddings=True)[0]
            max_sim = float(max(claim_emb @ obs_embs.T))

            # Semantic risk: 1 - max_semantic_sim
            # Low sim = high risk = this fact has no support in G_x
            sem_risk = 1.0 - max_sim

            # Also consider the original jaccard-based risk
            combined_risk = 0.5 * sem_risk + 0.5 * a.risk

            if combined_risk > best_semantic_risk:
                best_semantic_risk = combined_risk
                best_worst = a

        return best_worst

    def _repair(
        self,
        sample: UnifiedSample,
        candidate: CandidateOutput,
        assessments: list,
        observation_graph: ObservationGraph,
    ) -> CandidateOutput:
        """Backward-compatible single-fact repair shim.

        Selects the top-1 highest-risk fact from ``assessments`` and delegates
        to :meth:`_repair_batch`. Returns only the repaired candidate, matching
        the legacy interface used by ``tiger_no_graph`` and
        ``tiger_no_{lambda,nu}`` ablations.
        """
        if not assessments:
            return candidate
        targets = sorted(assessments, key=lambda a: a.risk, reverse=True)[:1]
        repaired, _, _, _ = self._repair_batch(
            sample, candidate, targets, observation_graph, current_risk=0.0
        )
        return repaired

    def _repair_batch(
        self,
        sample: UnifiedSample,
        candidate: CandidateOutput,
        targets: list,
        observation_graph: ObservationGraph,
        current_risk: float,
    ):
        """Batch repair: y^(t+1) = LLM(x, y^(t), {f*_1,...,f*_B}, evidence).

        Fixes ALL target facts in a single LLM call.
        - x: original input (image/audio via sample), LLM re-perceives it
        - y^(t): full previous output text
        - {f*}: list of high-risk facts to fix (our localization contribution)
        - evidence: G_x facts related to each f* (打破幻觉循环的结构化信号)

        Generates K candidates, keeps the one with lowest risk.

        Returns: (best_candidate, best_claim_graph, best_assessments, best_risk)
        """
        if not targets:
            cg = self.fact_extractor.extract_claim_graph(
                candidate, default_modality="layout" if candidate.layout else "text")
            a, r = self._assess(cg, observation_graph)
            return candidate, cg, a, r

        original_text = candidate.text or ""

        # ── Step 1: For each f*, gather evidence from G_x ──
        fact_lines = []
        for i, a in enumerate(targets):
            fact_desc = f"{a.fact.subject} {a.fact.predicate} {a.fact.obj}"

            # G_x facts semantically related to f* → structured evidence
            evidence = []
            for obs_f in observation_graph.facts:
                sim = self._semantic_sim(a.fact.text(), obs_f.text())
                if sim > 0.3:
                    evidence.append(obs_f.text())
            if not evidence:
                evidence = [f.text() for f in observation_graph.facts[:3]]

            evidence_str = "; ".join(evidence[:3]) if evidence else "(no closely related observations available)"
            fact_lines.append(
                f"  [{i+1}] Claim: \"{fact_desc}\"\n"
                f"      Related observations from the input: {evidence_str}"
            )

        facts_block = "\n".join(fact_lines)
        original_word_count = max(1, len(original_text.split()))

        # ── Step 2: Build the repair prompt ──
        # y^(t+1) = LLM(x, y^(t), {f*}, evidence)
        # Frame the flagged claims as high-risk (not inaccurate), letting the
        # LLM verify them against x and keep, correct, or rephrase as needed.
        repair_prompt = (
            f"Below is your previous response, followed by a list of claims that "
            f"have been flagged as high-risk under the current evidence. A claim "
            f"is flagged when it has weak support, possible conflict, or "
            f"cross-modal disagreement with the input.\n\n"
            f"Your goal: produce a revised response with high factual reliability. "
            f"For each flagged claim:\n"
            f"  (a) If the claim contradicts the input, correct it to match the "
            f"      input.\n"
            f"  (b) If the claim is supported by the input, keep it.\n"
            f"  (c) If the claim cannot be verified from the input, rephrase it "
            f"      with appropriate hedging or remove it — whichever keeps the "
            f"      surrounding text coherent.\n\n"
            f"Prefer factual reliability over length: a slightly shorter but "
            f"well-supported response is better than a longer one containing "
            f"unsupported claims. Preserve the original tone and the overall "
            f"answer to the task prompt.\n\n"
            f"Previous response:\n\"{original_text}\"\n\n"
            f"High-risk claims to re-examine:\n{facts_block}\n\n"
            f"Revised response:"
        )

        repair_settings = GenerationSettings(
            max_new_tokens=self.generation_settings.max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            return_audio=self.generation_settings.return_audio,
            voice=self.generation_settings.voice,
            num_candidates=1,
        )

        # ── Step 3: Generate K candidates, keep best ──
        best_candidate = candidate
        best_cg = self.fact_extractor.extract_claim_graph(
            candidate, default_modality="layout" if candidate.layout else "text")
        best_assessments, best_risk = self._assess(best_cg, observation_graph)

        for _ in range(self.k_repair):
            repaired = copy.deepcopy(candidate)
            try:
                # LLM sees: x (image/audio via sample) + repair prompt
                out = self.backbone.generate(sample, repair_prompt, repair_settings)
                new_text = (out.text or "").strip()

                # Clean up meta-prefixes
                for prefix in ["Revised response:", "Corrected response:",
                               "Here's the revised response:",
                               "Here's the corrected response:",
                               "Here is the revised", "Here is the corrected",
                               "Sure,", "Here's"]:
                    if new_text.lower().startswith(prefix.lower()):
                        new_text = new_text[len(prefix):].strip()
                new_text = new_text.strip('"').strip()

                # Validation
                if (not new_text
                        or len(new_text) < len(original_text) * 0.3
                        or len(new_text) > len(original_text) * 3
                        or new_text == original_text):
                    continue

                repaired.text = new_text
                repaired = self._parse_layout_if_needed(sample, repaired)
            except Exception:
                continue

            # Evaluate: extract G_y from new y, compute risk
            cg = self.fact_extractor.extract_claim_graph(
                repaired, default_modality="layout" if repaired.layout else "text")

            # Reject empty G_y — risk=0 from no facts is not a real improvement
            if not cg.facts:
                continue

            assessments, risk = self._assess(cg, observation_graph)

            if risk < best_risk:
                best_risk = risk
                best_candidate = repaired
                best_cg = cg
                best_assessments = assessments

        return best_candidate, best_cg, best_assessments, best_risk

    # ------------------------------------------------------------------
    # Main inference loop
    # ------------------------------------------------------------------

    def run(
        self, sample: UnifiedSample, mode: str = "tiger"
    ) -> InferenceOutput:

        # Step 1: Extract G_x with deterministic tools
        observation_graph = self.fact_extractor.extract_observation_graph(sample)

        state = PlannerState(
            step_idx=0, history=[]
        )

        # Step 2: LLM translation
        # TIGER and frozen: greedy initial generation (best quality)
        # Best-of-N baselines: sampling (diversity needed)
        use_greedy = mode in ("tiger", "tiger_random_batch", "frozen",
                              "flat_baseline", "tiger_no_graph",
                              "tiger_no_lambda", "tiger_no_nu",
                              "woodpecker", "degf", "tiger_no_gy", "volcano",
                              "tiger_feedback_raw", "tiger_feedback_text")
        best_candidate = self._parse_layout_if_needed(
            sample, self._initial_generate(sample, greedy=use_greedy)
        )

        # Step 3: Extract G_y with deterministic tools
        best_claim_graph = self.fact_extractor.extract_claim_graph(
            best_candidate,
            default_modality="layout" if best_candidate.layout is not None else "text",
        )
        best_assessments, best_risk = self._assess(best_claim_graph, observation_graph)

        # ---- Mode: frozen (B1) ----
        if mode == "frozen":
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph,
                claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph,
                claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- Mode: flat_baseline (A1 ablation — flat entailment, no graph) ----
        if mode == "flat_baseline":
            from .evidence import assess_candidate_flat
            best_assessments, best_risk = assess_candidate_flat(
                best_claim_graph, observation_graph,
                lambda_conflict=self.lambda_conflict,
                nu_inconsistency=self.nu_inconsistency,
            )
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph,
                claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph,
                claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- A1: tiger_no_graph — flat entailment + same repair loop ----
        if mode == "tiger_no_graph":
            from .evidence import assess_candidate_flat
            best_assessments, best_risk = assess_candidate_flat(
                best_claim_graph, observation_graph,
                lambda_conflict=self.lambda_conflict,
                nu_inconsistency=self.nu_inconsistency,
            )
            max_steps = self.planner.max_steps
            history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]
            for step in range(max_steps):
                if not best_assessments:
                    break
                new_candidate = self._repair(
                    sample, best_candidate, best_assessments, observation_graph
                )
                new_cg = self.fact_extractor.extract_claim_graph(
                    new_candidate,
                    default_modality="layout" if new_candidate.layout is not None else "text",
                )
                new_a, new_r = assess_candidate_flat(
                    new_cg, observation_graph,
                    lambda_conflict=self.lambda_conflict,
                    nu_inconsistency=self.nu_inconsistency,
                )
                state.history.append({"step": step, "old_risk": best_risk, "new_risk": new_r})
                state.step_idx += 1
                history.append((new_candidate, new_cg, new_a, new_r))
                if new_r < best_risk:
                    best_candidate, best_claim_graph, best_assessments, best_risk = new_candidate, new_cg, new_a, new_r
            best_entry = min(history, key=lambda h: h[3])
            best_candidate, best_claim_graph, best_assessments, best_risk = best_entry
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- A2 ablations: risk function variants ----
        # tiger_no_lambda: r(f) = (1 - s̄) + ν·δ  (no contradiction term)
        # tiger_no_nu:     r(f) = (1 - s̄) + λ·c̄  (no cross-modal disagreement)
        if mode in ("tiger_no_lambda", "tiger_no_nu"):
            ablation_lambda = 0.0 if mode == "tiger_no_lambda" else self.lambda_conflict
            ablation_nu = 0.0 if mode == "tiger_no_nu" else self.nu_inconsistency

            best_assessments, best_risk = assess_candidate(
                best_claim_graph, observation_graph,
                lambda_conflict=ablation_lambda,
                nu_inconsistency=ablation_nu, gamma=self.gamma,
            )
            max_steps = self.planner.max_steps
            history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]
            for step in range(max_steps):
                if not best_assessments:
                    break
                new_candidate = self._repair(
                    sample, best_candidate, best_assessments, observation_graph
                )
                new_cg = self.fact_extractor.extract_claim_graph(
                    new_candidate,
                    default_modality="layout" if new_candidate.layout is not None else "text",
                )
                new_a, new_r = assess_candidate(
                    new_cg, observation_graph,
                    lambda_conflict=ablation_lambda,
                    nu_inconsistency=ablation_nu, gamma=self.gamma,
                )
                state.history.append({"step": step, "old_risk": best_risk, "new_risk": new_r})
                state.step_idx += 1
                history.append((new_candidate, new_cg, new_a, new_r))
                if new_r < best_risk:
                    best_candidate, best_claim_graph, best_assessments, best_risk = new_candidate, new_cg, new_a, new_r
            best_entry = min(history, key=lambda h: h[3])
            best_candidate, best_claim_graph, best_assessments, best_risk = best_entry
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- B5: Woodpecker-style — whole-text repair (coarse granularity) ----
        # Woodpecker does NOT have access to G_x evidence graph.
        # It uses the backbone to self-review and rewrite the entire output.
        # This is the key difference from TIGER: no fact-level localization.
        if mode == "woodpecker":
            max_steps = self.planner.max_steps
            history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]
            for step in range(max_steps):
                if not best_assessments:
                    break
                original_text = best_candidate.text or ""
                original_word_count = max(1, len(original_text.split()))
                # Anti-shrinkage rewrite: produce a faithful response of comparable
                # length, preserving informational coverage rather than collapsing
                # content. This neutralizes the "say less to avoid hallucination"
                # shortcut.
                rewrite_prompt = (
                    f"{sample.prompt}\n\n"
                    f"Provide a detailed, factually accurate response based on the "
                    f"input. Cover the same level of detail as a typical full answer "
                    f"to this prompt (roughly {original_word_count} words). Do not "
                    f"shorten the response to avoid uncertainty: describe everything "
                    f"you can verify from the input. Preserve tone and style."
                )
                repair_settings = GenerationSettings(
                    max_new_tokens=self.generation_settings.max_new_tokens,
                    temperature=0.7, top_p=0.9, do_sample=True,
                    return_audio=self.generation_settings.return_audio,
                    voice=self.generation_settings.voice, num_candidates=1,
                )
                new_out = self.backbone.generate(sample, rewrite_prompt, repair_settings)
                new_candidate = self._parse_layout_if_needed(sample, new_out)
                new_cg = self.fact_extractor.extract_claim_graph(
                    new_candidate,
                    default_modality="layout" if new_candidate.layout is not None else "text",
                )
                new_a, new_r = self._assess(new_cg, observation_graph)
                state.history.append({"step": step, "old_risk": best_risk, "new_risk": new_r})
                state.step_idx += 1
                history.append((new_candidate, new_cg, new_a, new_r))
                if new_r < best_risk:
                    best_candidate, best_claim_graph, best_assessments, best_risk = new_candidate, new_cg, new_a, new_r
            best_entry = min(history, key=lambda h: h[3])
            best_candidate, best_claim_graph, best_assessments, best_risk = best_entry
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- B6: DeGF-style — single self-correction (no evidence graph) ----
        if mode == "degf":
            original_text = best_candidate.display_text()
            original_word_count = max(1, len(original_text.split()))
            # Anti-shrinkage version: still a single self-correction pass, but
            # the prompt explicitly forbids shortening as a way to mitigate
            # hallucinations (otherwise the model trivially wins on Risk by
            # producing very short responses).
            self_correct_prompt = (
                f"You previously generated the following output:\n\n"
                f"{original_text}\n\n"
                f"Review your output for factual errors, hallucinations, or "
                f"inconsistencies with the input, and produce a corrected "
                f"version. Keep approximately the same length and level of "
                f"detail as the original response (roughly {original_word_count} "
                f"words); do NOT shorten the response just to reduce uncertainty. "
                f"If a claim cannot be verified, rephrase it cautiously instead "
                f"of deleting it. Preserve the original tone and style."
            )
            repair_settings = GenerationSettings(
                max_new_tokens=self.generation_settings.max_new_tokens,
                temperature=0.0, top_p=0.9, do_sample=False,
                return_audio=self.generation_settings.return_audio,
                voice=self.generation_settings.voice, num_candidates=1,
            )
            corrected = self.backbone.generate(sample, self_correct_prompt, repair_settings)
            corrected = self._parse_layout_if_needed(sample, corrected)
            corrected_cg = self.fact_extractor.extract_claim_graph(
                corrected,
                default_modality="layout" if corrected.layout is not None else "text",
            )
            corrected_a, corrected_r = self._assess(corrected_cg, observation_graph)
            # Take whichever is better
            if corrected_r < best_risk:
                best_candidate, best_claim_graph, best_assessments, best_risk = corrected, corrected_cg, corrected_a, corrected_r
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- Mode: Volcano — self-feedback guided revision (NAACL 2024) ----
        # Standard self-refine: critique → revise → decide, 3 rounds
        # No structured G_x feedback — LLM generates free-form feedback itself
        if mode == "volcano":
            max_rounds = self.planner.max_steps
            history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]

            feedback_settings = GenerationSettings(
                max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True,
                return_audio=False, voice=self.generation_settings.voice, num_candidates=1,
            )
            revise_settings = GenerationSettings(
                max_new_tokens=self.generation_settings.max_new_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                return_audio=self.generation_settings.return_audio,
                voice=self.generation_settings.voice, num_candidates=1,
            )

            for step in range(max_rounds):
                current_text = best_candidate.text or ""
                original_word_count = max(1, len(current_text.split()))

                # Phase 1: CRITIQUE — LLM generates free-form feedback
                critique_prompt = (
                    f"Question: {sample.prompt}\n\n"
                    f"Response: {current_text}\n\n"
                    f"Analyze this response for factual errors, hallucinations, "
                    f"or inconsistencies with the image/audio. "
                    f"List specific issues that need correction."
                )
                feedback_out = self.backbone.generate(sample, critique_prompt, feedback_settings)
                feedback = (feedback_out.text or "").strip()

                # Phase 2: REVISE — LLM revises based on its own feedback
                # Anti-shrinkage: explicitly require similar length so that the
                # revision does not collapse content as a way to mitigate risk.
                revise_prompt = (
                    f"Question: {sample.prompt}\n\n"
                    f"Your previous response: {current_text}\n\n"
                    f"Issues found: {feedback}\n\n"
                    f"Provide a corrected response that fixes these issues. "
                    f"Keep approximately the same length and level of detail as "
                    f"the original (roughly {original_word_count} words); do "
                    f"NOT shorten the response just to avoid uncertain claims. "
                    f"Preserve the same style and format."
                )
                revised_out = self.backbone.generate(sample, revise_prompt, revise_settings)
                revised_text = (revised_out.text or "").strip()

                if not revised_text or revised_text == current_text:
                    state.history.append({"step": step, "old_risk": best_risk, "new_risk": best_risk})
                    state.step_idx += 1
                    history.append((best_candidate, best_claim_graph, best_assessments, best_risk))
                    continue

                # Phase 3: DECIDE — evaluate revised version
                revised_cand = copy.deepcopy(best_candidate)
                revised_cand.text = revised_text
                revised_cand = self._parse_layout_if_needed(sample, revised_cand)
                revised_cg = self.fact_extractor.extract_claim_graph(
                    revised_cand, default_modality="layout" if revised_cand.layout else "text")

                if not revised_cg.facts:
                    state.history.append({"step": step, "old_risk": best_risk, "new_risk": best_risk})
                    state.step_idx += 1
                    history.append((best_candidate, best_claim_graph, best_assessments, best_risk))
                    continue

                revised_a, revised_r = self._assess(revised_cg, observation_graph)
                state.history.append({"step": step, "old_risk": best_risk, "new_risk": revised_r})
                state.step_idx += 1
                history.append((revised_cand, revised_cg, revised_a, revised_r))

                if revised_r < best_risk:
                    best_candidate, best_claim_graph = revised_cand, revised_cg
                    best_assessments, best_risk = revised_a, revised_r

            best_entry = min(history, key=lambda h: h[3])
            best_candidate, best_claim_graph, best_assessments, best_risk = best_entry
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- Mode: tiger_feedback_raw / tiger_feedback_text ----
        # Ablations on the FEEDBACK function F^(t):
        #   raw : F^(t) = M(p, x, y^(t))                  -- LLM sees image/audio
        #   text: F^(t) = M(p, T(x), T(y^(t)))            -- LLM sees only text
        # Repair y^(t+1) = M(x, y^(t), F^(t)) is shared with TIGER (re-uses
        # backbone, sees original x via sample). Best-tracker acceptance.
        if mode in ("tiger_feedback_raw", "tiger_feedback_text"):
            from .schemas import UnifiedSample as _US

            max_steps = self.planner.max_steps
            history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]

            feedback_settings = GenerationSettings(
                max_new_tokens=256, temperature=0.7, top_p=0.9, do_sample=True,
                return_audio=False, voice=self.generation_settings.voice, num_candidates=1,
            )
            repair_settings = GenerationSettings(
                max_new_tokens=self.generation_settings.max_new_tokens,
                temperature=0.7, top_p=0.9, do_sample=True,
                return_audio=self.generation_settings.return_audio,
                voice=self.generation_settings.voice, num_candidates=1,
            )

            # --- Build T(x) once for the text-only variant ---
            t_x_cached: Optional[str] = None
            if mode == "tiger_feedback_text":
                # Caption / transcribe x into text. Use backbone with x as input.
                if sample.image_paths or sample.audio_paths:
                    caption_prompt = (
                        "Describe everything visible in this input in plain text. "
                        "List objects, attributes, actions, and spatial relations. "
                        "Be concise and factual."
                    )
                    caption_out = self.backbone.generate(sample, caption_prompt, feedback_settings)
                    t_x_cached = (caption_out.text or "").strip()
                else:
                    # text-only input: T(x) == x (the prompt itself)
                    t_x_cached = (sample.prompt or "").strip()

            for step in range(max_steps):
                current_text = best_candidate.text or ""

                # ---- Phase 1: build F^(t) ----
                if mode == "tiger_feedback_raw":
                    # F^(t) = M(p, x, y^(t)) -- LLM sees x via `sample`
                    critique_prompt = (
                        f"Task prompt: {sample.prompt}\n\n"
                        f"Current response: \"{current_text}\"\n\n"
                        f"Compare the response against the input above. List the "
                        f"specific claims in the response that are inaccurate or "
                        f"unsupported by the input. For each, briefly state what "
                        f"the correct information is. Use a numbered list."
                    )
                    feedback_out = self.backbone.generate(sample, critique_prompt, feedback_settings)
                else:
                    # tiger_feedback_text: F^(t) = M(p, T(x), T(y^(t)))
                    # Make a text-only sample (no image/audio) so the backbone only
                    # sees the textual description of x and y.
                    t_y = current_text  # T(y^(t)) = y^(t) (already text)
                    critique_prompt = (
                        f"Input description T(x): \"{t_x_cached}\"\n\n"
                        f"Current response y^(t): \"{t_y}\"\n\n"
                        f"Compare y^(t) against T(x). List the specific claims in "
                        f"y^(t) that are inaccurate or unsupported by T(x). For each, "
                        f"briefly state what the correct information is. "
                        f"Use a numbered list."
                    )
                    text_only_sample = _US(
                        sample_id=sample.sample_id + "__fb",
                        task_type=sample.task_type,
                        prompt=critique_prompt,
                        image_paths=[],
                        audio_paths=[],
                    )
                    feedback_out = self.backbone.generate(
                        text_only_sample, critique_prompt, feedback_settings
                    )

                feedback_text = (feedback_out.text or "").strip()

                # ---- Phase 2: y^(t+1) = M(x, y^(t), F^(t)) ----
                # Repair step is the SAME across raw/text ablations: backbone
                # always sees the original x (via `sample`).
                if not feedback_text:
                    state.history.append({"step": step, "old_risk": best_risk, "new_risk": best_risk})
                    state.step_idx += 1
                    history.append((best_candidate, best_claim_graph, best_assessments, best_risk))
                    continue

                original_word_count = max(1, len(current_text.split()))
                repair_prompt = (
                    f"Your previous response:\n\"{current_text}\"\n\n"
                    f"Feedback identifying inaccurate claims:\n{feedback_text}\n\n"
                    f"Regenerate your response, fixing only the inaccurate claims "
                    f"based on the feedback. Keep the rest of the response "
                    f"unchanged. Maintain approximately the same length as the "
                    f"original (roughly {original_word_count} words); do NOT "
                    f"shorten the response just to reduce uncertainty.\n\n"
                    f"Corrected response:"
                )
                repaired_out = self.backbone.generate(sample, repair_prompt, repair_settings)
                repaired_text = (repaired_out.text or "").strip()
                if not repaired_text or repaired_text == current_text:
                    state.history.append({"step": step, "old_risk": best_risk, "new_risk": best_risk})
                    state.step_idx += 1
                    history.append((best_candidate, best_claim_graph, best_assessments, best_risk))
                    continue

                # ---- Phase 3: extract G_y, assess, best-tracker ----
                repaired_cand = copy.deepcopy(best_candidate)
                repaired_cand.text = repaired_text
                repaired_cand = self._parse_layout_if_needed(sample, repaired_cand)
                repaired_cg = self.fact_extractor.extract_claim_graph(
                    repaired_cand,
                    default_modality="layout" if repaired_cand.layout else "text",
                )
                if not repaired_cg.facts:
                    state.history.append({"step": step, "old_risk": best_risk, "new_risk": best_risk})
                    state.step_idx += 1
                    history.append((best_candidate, best_claim_graph, best_assessments, best_risk))
                    continue
                repaired_a, repaired_r = self._assess(repaired_cg, observation_graph)
                state.history.append({
                    "step": step,
                    "old_risk": best_risk,
                    "new_risk": repaired_r,
                    "feedback_text": feedback_text,   # RQ4 Sub-Q1: record F_t for FER analysis
                })
                state.step_idx += 1
                history.append((repaired_cand, repaired_cg, repaired_a, repaired_r))

                if repaired_r < best_risk:
                    best_candidate, best_claim_graph = repaired_cand, repaired_cg
                    best_assessments, best_risk = repaired_a, repaired_r

            best_entry = min(history, key=lambda h: h[3])
            best_candidate, best_claim_graph, best_assessments, best_risk = best_entry
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- Mode: Best-of-N baselines (B2/B3/B4) ----
        if mode in ("baseline_scalar", "visualprm", "cycle_reward"):
            candidates_with_scores = [
                (best_candidate, best_claim_graph, best_assessments, best_risk, 0.0)
            ]

            # Generate N-1 additional candidates (with sampling)
            sample_settings = GenerationSettings(
                max_new_tokens=self.generation_settings.max_new_tokens,
                temperature=0.7,
                top_p=self.generation_settings.top_p,
                do_sample=True,
                return_audio=self.generation_settings.return_audio,
                voice=self.generation_settings.voice,
                num_candidates=1,
            )
            for _ in range(1, self.generation_settings.num_candidates):
                cand = self._parse_layout_if_needed(
                    sample, self.backbone.generate(sample, sample.prompt, sample_settings)
                )
                cg = self.fact_extractor.extract_claim_graph(
                    cand, default_modality="layout" if cand.layout is not None else "text"
                )
                a, r = self._assess(cg, observation_graph)
                candidates_with_scores.append((cand, cg, a, r, 0.0))

            if mode == "baseline_scalar":
                # B2: Best-of-N + CLIP/ASR reranking (scalar verifier).
                # Uses CLIP for image tasks, text overlap for audio/text tasks.
                if self._clip_verifier is None:
                    self._clip_verifier = CLIPVerifier()
                scored = [self._clip_verifier.score(sample, c[0])
                          for c in candidates_with_scores]
                best_idx = max(range(len(scored)), key=lambda i: scored[i])
            elif mode == "visualprm":
                if self._clip_verifier is None:
                    self._clip_verifier = CLIPVerifier()
                scored = [self._clip_verifier.score(sample, c[0])
                          for c in candidates_with_scores]
                best_idx = max(range(len(scored)), key=lambda i: scored[i])
            elif mode == "cycle_reward":
                if self._cycle_verifier is None:
                    self._cycle_verifier = CycleVerifier(
                        self.backbone, self.generation_settings
                    )
                scored = [self._cycle_verifier.score(sample, c[0])
                          for c in candidates_with_scores]
                best_idx = max(range(len(scored)), key=lambda i: scored[i])
            else:
                best_idx = 0

            best_candidate, best_claim_graph, best_assessments, best_risk, _ = (
                candidates_with_scores[best_idx]
            )
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph,
                claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph,
                claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics, evidence_trace=[],
            )

        # ---- Ablation: tiger_no_gy — use G_x to rewrite y directly, no G_y ----
        # Instead of extracting G_y, computing entailment, and targeting f*,
        # this ablation gives the backbone ALL G_x evidence and asks it to
        # rewrite y to be consistent. No fact-level localization.
        if mode == "tiger_no_gy":
            max_steps = self.planner.max_steps
            history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]
            evidence_str = "\n".join(
                f"{f.text()}" for f in observation_graph.facts[:15]
            )
            for step in range(max_steps):
                original_text = best_candidate.display_text()
                rewrite_prompt = (
                    f"The following text should be consistent with these evidence facts:\n"
                    f"{evidence_str}\n\n"
                    f"Current text: {original_text[:400]}\n\n"
                    f"Rewrite the text to make it fully consistent with the evidence. "
                    f"Keep the same style and length. Only change parts that contradict "
                    f"or are not supported by the evidence."
                )
                repair_settings = GenerationSettings(
                    max_new_tokens=self.generation_settings.max_new_tokens,
                    temperature=0.7, top_p=0.9, do_sample=True,
                    return_audio=self.generation_settings.return_audio,
                    voice=self.generation_settings.voice, num_candidates=1,
                )
                new_out = self.backbone.generate(sample, rewrite_prompt, repair_settings)
                new_candidate = self._parse_layout_if_needed(sample, new_out)
                # Still compute G_y and risk for EVALUATION (not for repair decisions)
                new_cg = self.fact_extractor.extract_claim_graph(
                    new_candidate,
                    default_modality="layout" if new_candidate.layout is not None else "text",
                )
                new_a, new_r = self._assess(new_cg, observation_graph)
                state.history.append({"step": step, "old_risk": best_risk, "new_risk": new_r})
                state.step_idx += 1
                history.append((new_candidate, new_cg, new_a, new_r))
                if new_r < best_risk:
                    best_candidate, best_claim_graph = new_candidate, new_cg
                    best_assessments, best_risk = new_a, new_r
            best_entry = min(history, key=lambda h: h[3])
            best_candidate, best_claim_graph, best_assessments, best_risk = best_entry
            metrics = task_metrics(sample, best_candidate)
            metrics.update(compute_evidence_metrics(InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics={}, evidence_trace=[],
            )))
            metrics["risk"] = best_risk
            return InferenceOutput(
                sample=sample, prediction=best_candidate,
                observation_graph=observation_graph, claim_graph=best_claim_graph,
                assessments=best_assessments, risk=best_risk,
                planner_state=state, metrics=metrics,
                evidence_trace=[a.to_dict() for a in best_assessments],
            )

        # ---- Mode: TIGER / tiger_random_batch ----
        # tiger: greedy batch — repair top-B highest-risk facts per round
        # tiger_random_batch: random batch — repair B randomly sampled facts per round
        random_batch = (mode == "tiger_random_batch")
        import random as _random

        max_steps = self.planner.max_steps
        n_gx = len(observation_graph.facts)
        if self.alpha_batch > 0:
            batch_size = max(1, int(self.alpha_batch * n_gx))
        else:
            batch_size = 1
        history = [(best_candidate, best_claim_graph, best_assessments, best_risk)]

        for step in range(max_steps):
            if not best_assessments:
                break

            # Optimization C: if all remaining facts are already low-risk,
            # there is nothing meaningful to repair — skip the round to save compute.
            max_risk_now = max((a.risk for a in best_assessments), default=0.0)
            if max_risk_now < 0.3:
                break

            if random_batch:
                # Random baseline: sample B facts uniformly
                targets = _random.sample(best_assessments, min(batch_size, len(best_assessments)))
            else:
                # Greedy: sort by risk descending, take top-B
                sorted_a = sorted(best_assessments, key=lambda a: a.risk, reverse=True)
                targets = sorted_a[:batch_size]

            # Repair ALL target facts in a single LLM call
            # Returns (candidate, claim_graph, assessments, risk) — no duplicate extraction
            new_candidate, new_claim_graph, new_assessments, new_risk = self._repair_batch(
                sample, best_candidate, targets, observation_graph, best_risk
            )

            state.history.append({
                "step": step,
                "old_risk": best_risk,
                "new_risk": new_risk,
                "batch_size": batch_size,
                # RQ4 Sub-Q1: record Ψ_α's top-α high-risk facts for FER analysis
                "selected_fact_ids": [a.fact.fact_id for a in targets],
                "selected_facts_text": [
                    f"{a.fact.subject}|{a.fact.predicate}|{a.fact.obj}"
                    for a in targets
                ],
            })
            state.step_idx += 1

            history.append((new_candidate, new_claim_graph, new_assessments, new_risk))

            if new_risk < best_risk:
                best_candidate = new_candidate
                best_claim_graph = new_claim_graph
                best_assessments = new_assessments
                best_risk = new_risk

        # Return the best from history (argmin risk)
        best_entry = min(history, key=lambda h: h[3])
        best_candidate, best_claim_graph, best_assessments, best_risk = best_entry

        metrics = task_metrics(sample, best_candidate)
        metrics.update(compute_evidence_metrics(InferenceOutput(
            sample=sample, prediction=best_candidate,
            observation_graph=observation_graph,
            claim_graph=best_claim_graph,
            assessments=best_assessments, risk=best_risk,
            planner_state=state, metrics={}, evidence_trace=[],
        )))
        metrics["risk"] = best_risk

        return InferenceOutput(
            sample=sample, prediction=best_candidate,
            observation_graph=observation_graph,
            claim_graph=best_claim_graph,
            assessments=best_assessments, risk=best_risk,
            planner_state=state, metrics=metrics,
            evidence_trace=[a.to_dict() for a in best_assessments],
        )
