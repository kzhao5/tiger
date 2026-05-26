from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import soundfile as sf

from .schemas import CandidateOutput, UnifiedSample
from .utils import ensure_dir, normalize_text

DEFAULT_AUDIO_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


@dataclass
class GenerationSettings:
    max_new_tokens: int = 256
    temperature: float = 0.2
    top_p: float = 0.9
    do_sample: bool = False
    return_audio: bool = False
    voice: str = "Chelsie"
    num_candidates: int = 1


class BaseBackbone:
    def generate(self, sample: UnifiedSample, prompt: str, settings: GenerationSettings) -> CandidateOutput:
        raise NotImplementedError

    def edit_text(self, sample: UnifiedSample, current_text: str, repair_instruction: str, settings: GenerationSettings) -> str:
        raise NotImplementedError

    def extract_facts_prompt(self, prompt: str, sample: Optional[UnifiedSample], settings: GenerationSettings) -> str:
        raise NotImplementedError


class DummyBackbone(BaseBackbone):
    def generate(self, sample: UnifiedSample, prompt: str, settings: GenerationSettings) -> CandidateOutput:
        if sample.task_type.value.endswith("mcq"):
            text = "A"
        elif sample.target_text:
            text = sample.target_text
        elif sample.target_layout is not None:
            text = json.dumps(sample.target_layout, ensure_ascii=False)
        else:
            text = f"DUMMY_RESPONSE: {prompt[:128]}"
        return CandidateOutput(text=text, raw={"dummy": True})

    def edit_text(self, sample: UnifiedSample, current_text: str, repair_instruction: str, settings: GenerationSettings) -> str:
        return current_text

    def extract_facts_prompt(self, prompt: str, sample: Optional[UnifiedSample], settings: GenerationSettings) -> str:
        return json.dumps(
            [
                {
                    "fact_id": "f0",
                    "subject": "dummy",
                    "predicate": "says",
                    "object": normalize_text(prompt)[:64],
                    "modality": "text",
                }
            ]
        )


class QwenOmniBackbone(BaseBackbone):
    def __init__(self, model_name: str, enable_audio_output: bool = True, device_map: str = "auto", dtype: str = "auto"):
        try:
            from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
        except Exception as exc:
            raise ImportError(
                "Qwen2.5-Omni classes are unavailable. Install a recent transformers build that includes Qwen2.5-Omni."
            ) from exc

        self.model_name = model_name
        self.enable_audio_output = enable_audio_output
        self.processor = Qwen2_5OmniProcessor.from_pretrained(model_name)
        self.model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
            model_name,
            dtype=dtype,
            device_map=device_map,
            enable_audio_output=enable_audio_output,
            attn_implementation="sdpa",
        )

    def _extract_video_frames(self, video_path: str, n_frames: int = 4, max_side: int = 320):
        """Decode a video and return n_frames uniformly-sampled PIL frames
        resized so the longest edge is <= max_side. Bypasses the Qwen video
        processor's uncapped sampling which can produce 100k+ vision tokens
        on long clips."""
        import decord
        from PIL import Image
        try:
            vr = decord.VideoReader(video_path, ctx=decord.cpu(0))
            total = len(vr)
            if total <= 0:
                return []
            idxs = np.linspace(0, total - 1, num=min(n_frames, total), dtype=int)
            frames_np = vr.get_batch(idxs.tolist()).asnumpy()
        except Exception:
            return []
        out = []
        for arr in frames_np:
            img = Image.fromarray(arr)
            w, h = img.size
            longest = max(w, h)
            if longest > max_side:
                scale = max_side / float(longest)
                img = img.resize((int(w * scale), int(h * scale)))
            out.append(img)
        return out

    def _build_conversation(self, sample: UnifiedSample, prompt: str, want_audio_output: bool) -> List[Dict[str, Any]]:
        # Always use Qwen's default system prompt to avoid degraded
        # audio/image perception when using custom prompts.
        system_text = DEFAULT_AUDIO_SYSTEM_PROMPT
        user_content: List[Dict[str, Any]] = []
        for img in sample.image_paths:
            user_content.append({"type": "image", "path": img})
        for audio in sample.audio_paths:
            user_content.append({"type": "audio", "path": audio})
        if sample.video_paths:
            # Pre-extract a small fixed number of frames per video to keep
            # vision tokens under the 32k context window. Pass them as
            # individual images so the video processor's uncapped frame
            # sampling is bypassed entirely.
            for video in sample.video_paths:
                frames = self._extract_video_frames(video, n_frames=4, max_side=320)
                for frame_img in frames:
                    user_content.append({"type": "image", "image": frame_img})
        user_content.append({"type": "text", "text": prompt})
        return [
            {"role": "system", "content": [{"type": "text", "text": system_text}]},
            {"role": "user", "content": user_content},
        ]

    def _generate_raw(self, conversations: List[Dict[str, Any]], settings: GenerationSettings):
        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        ).to(self.model.device)

        # Record input length to strip prompt from output
        input_len = inputs["input_ids"].shape[-1]

        if settings.return_audio:
            text_ids, audio = self.model.generate(
                **inputs,
                thinker_do_sample=settings.do_sample,
                thinker_temperature=settings.temperature,
                thinker_top_p=settings.top_p,
                thinker_max_new_tokens=settings.max_new_tokens,
                return_audio=True,
            )
            # Only decode newly generated tokens (strip input prompt)
            new_ids = text_ids[:, input_len:]
            text = self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            return text, audio
        else:
            text_ids = self.model.generate(
                **inputs,
                thinker_do_sample=settings.do_sample,
                thinker_temperature=settings.temperature,
                thinker_top_p=settings.top_p,
                thinker_max_new_tokens=settings.max_new_tokens,
                return_audio=False,
            )
            # Only decode newly generated tokens (strip input prompt)
            new_ids = text_ids[:, input_len:]
            text = self.processor.batch_decode(
                new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()
            return text, None

    def generate(self, sample: UnifiedSample, prompt: str, settings: GenerationSettings) -> CandidateOutput:
        conversations = self._build_conversation(sample, prompt, want_audio_output=settings.return_audio)
        text, audio = self._generate_raw(conversations, settings)
        out = CandidateOutput(text=text, raw={"model_name": self.model_name})
        if audio is not None:
            out_dir = ensure_dir("results/generated_audio")
            path = Path(out_dir) / f"{sample.sample_id}.wav"
            sf.write(path, audio.reshape(-1).detach().cpu().numpy(), samplerate=24000)
            out.audio_path = str(path)
        return out

    def edit_text(self, sample: UnifiedSample, current_text: str, repair_instruction: str, settings: GenerationSettings) -> str:
        repair_prompt = (
            "You are repairing a partially-correct answer using grounded evidence.\n"
            f"Original output:\n{current_text}\n\n"
            f"Repair instruction:\n{repair_instruction}\n\n"
            "Return only the corrected output."
        )
        out = self.generate(sample, repair_prompt, settings)
        return out.text or current_text

    def extract_facts_prompt(self, prompt: str, sample: Optional[UnifiedSample], settings: GenerationSettings) -> str:
        if sample is None:
            sample = UnifiedSample(sample_id="fact", task_type=None, prompt=prompt)  # type: ignore[arg-type]
        out = self.generate(sample, prompt, settings)
        return out.text or "[]"


def build_backbone(config: Dict[str, Any]) -> BaseBackbone:
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "dummy")
    if model_type == "dummy":
        return DummyBackbone()
    if model_type == "qwen2_5_omni":
        return QwenOmniBackbone(
            model_name=model_cfg["name"],
            enable_audio_output=bool(model_cfg.get("enable_audio_output", True)),
            device_map=model_cfg.get("device_map", "auto"),
            dtype=model_cfg.get("dtype", "auto"),
        )
    raise ValueError(
        f"Unsupported model type: {model_type}. "
        f"This release supports 'qwen2_5_omni' (Qwen/Qwen2.5-Omni-7B) and "
        f"'dummy' (testing only). To add a new backbone, subclass "
        f"BaseBackbone and extend build_backbone()."
    )
