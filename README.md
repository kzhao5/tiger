# TIGER

**TIGER (Traceable Inference with Graph-based Evidence Routing)** is a
multimodal hallucination-repair pipeline. Given a multimodal sample
(image / audio / video + prompt) and a base vision-language backbone,
TIGER iteratively rewrites the model's response so that each emitted
claim is grounded in the input.

---

## Install

```bash
git clone <this repo>
cd tiger
pip install -r requirements.txt
pip install -e .
python -m spacy download en_core_web_sm
```

Hardware: a single GPU with ≥ 24 GB VRAM is enough to run
Qwen2.5-Omni-7B at the default settings (the model weights are about
16 GB).

## Quickstart

```bash
python examples/demo.py
```

This loads `examples/sample_image.jpg`, runs the full TIGER pipeline
(extract observation graph $G_{\mathbf{x}}$ → generate initial response
$\mathbf{y}_0$ → extract claim graph $G_{\mathbf{y}}$ → score risk →
repair $T$ rounds), and prints the repaired response plus the final
risk.

## Programmatic use

```python
from tiger import (
    GenerationSettings, PlannerConfig, QwenOmniBackbone,
    TigerInferenceEngine, UnifiedSample,
)
from tiger.schemas import TaskType

backbone = QwenOmniBackbone(model_name="Qwen/Qwen2.5-Omni-7B")
engine = TigerInferenceEngine(
    backbone=backbone,
    planner_config=PlannerConfig(max_steps=5),
    generation_settings=GenerationSettings(max_new_tokens=256),
)

sample = UnifiedSample(
    sample_id="my_sample",
    task_type=TaskType.IMAGE_TEXT_TO_TEXT_FREEFORM,
    prompt="Describe the image.",
    image_paths=["path/to/image.jpg"],
)

out = engine.run(sample, mode="tiger")
print(out.prediction.text)
```

## Layout

```
tiger/                 # The package
  __init__.py
  pipeline.py          # TigerInferenceEngine — main loop
  evidence.py          # G_x / G_y extraction, support s(f), conflict c(f)
  schemas.py           # UnifiedSample, Fact, ObservationGraph, ClaimGraph
  models.py            # BaseBackbone + Qwen2.5-Omni implementation
  tools.py             # Deterministic graph-extraction tools
  metrics.py           # Per-sample metric helpers
  planner.py           # PlannerConfig (max_steps, etc.)
  verifiers.py         # CLIP / cycle-consistency verifiers (optional)
  utils.py             # I/O, normalization

examples/
  demo.py              # End-to-end example
  sample_image.jpg     # Sample COCO image

requirements.txt
setup.py
README.md
```

## Adding a backbone

Subclass `tiger.models.BaseBackbone` and implement `generate(...)`.
Then register the new branch in `build_backbone()` in `tiger/models.py`.

```python
class MyBackbone(BaseBackbone):
    def generate(self, sample, prompt, settings):
        ...
        return CandidateOutput(text=...)
```

## Citation

```bibtex
@article{tiger,
  title  = {TIGER: Traceable Inference with Graph-based Evidence Routing for Multimodal Hallucination Repair},
  author = {...},
  year   = {2025},
}
```

## License

MIT.
