"""Minimal TIGER demo — run a single image through the repair pipeline.

Usage:
    cd tiger_release
    python examples/demo.py
"""
from pathlib import Path

from tiger import (
    GenerationSettings,
    PlannerConfig,
    QwenOmniBackbone,
    TigerInferenceEngine,
    UnifiedSample,
)
from tiger.schemas import TaskType


HERE = Path(__file__).resolve().parent
IMAGE_PATH = HERE / "sample_image.jpg"


def main() -> None:
    # 1. Load Qwen2.5-Omni-7B (requires a recent transformers build with
    #    Qwen2_5OmniForConditionalGeneration; download is automatic on
    #    first run, ~16 GB).
    backbone = QwenOmniBackbone(
        model_name="Qwen/Qwen2.5-Omni-7B",
        enable_audio_output=False,
    )

    # 2. Build the TIGER inference engine. Default hyperparameters match
    #    the paper; tweak if you want a different operating point.
    engine = TigerInferenceEngine(
        backbone=backbone,
        planner_config=PlannerConfig(max_steps=5),    # T (repair rounds)
        generation_settings=GenerationSettings(
            max_new_tokens=256, temperature=0.2, top_p=0.9, do_sample=False,
        ),
        k_repair=1,         # candidates per round
        lambda_conflict=1.5,
        nu_inconsistency=0.1,
        alpha_batch=0.2,    # fraction of facts repaired per round
    )

    # 3. Build a sample. A "sample" is whatever the model should answer
    #    about. Here: an image + a free-form description prompt.
    sample = UnifiedSample(
        sample_id="demo_0",
        task_type=TaskType.IMAGE_TEXT_TO_TEXT_FREEFORM,
        prompt="Describe what you see in this image in 3-4 sentences.",
        image_paths=[str(IMAGE_PATH)],
    )

    # 4. Run TIGER. Mode "tiger" = full pipeline (extraction + scoring +
    #    iterative repair). Other modes exist in pipeline.py for
    #    research / ablation purposes but are not exposed here.
    out = engine.run(sample, mode="tiger")

    # 5. Inspect output.
    print("=== TIGER prediction ===")
    print(out.prediction.text)
    print()
    print(f"Final risk r(y_T) = {out.risk:.3f}")
    print(f"|G_x| (observation facts) = {len(out.observation_graph.facts)}")
    print(f"|G_y| (claim facts)       = {len(out.claim_graph.facts)}")


if __name__ == "__main__":
    main()
