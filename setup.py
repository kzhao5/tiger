from pathlib import Path
from setuptools import find_packages, setup


HERE = Path(__file__).resolve().parent
LONG_DESCRIPTION = (HERE / "README.md").read_text(encoding="utf-8")


setup(
    name="tiger",
    version="0.1.0",
    description=(
        "TIGER — Traceable Inference with Graph-based Evidence Routing. "
        "A multimodal hallucination-repair pipeline."
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["tiger", "tiger.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.1",
        "transformers>=4.46",
        "accelerate>=0.30",
        "sentence-transformers>=2.7",
        "spacy>=3.7",
        "Pillow>=10.0",
        "numpy>=1.24",
        "soundfile>=0.12",
        "decord>=0.6",
    ],
    extras_require={
        "clip": ["clip @ git+https://github.com/openai/CLIP.git"],
    },
)
