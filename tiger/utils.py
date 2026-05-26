from __future__ import annotations

import json
import os
import random
import re
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import soundfile as sf
import yaml
from PIL import Image


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def load_yaml(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def normalize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9\s\.\,\-\_:/]", "", text)
    return text


def exact_match(a: Optional[str], b: Optional[str]) -> float:
    return float(normalize_text(a) == normalize_text(b))


def option_normalize(text: Optional[str]) -> str:
    if text is None:
        return ""
    text = normalize_text(text)
    text = text.replace("option", "").replace("answer", "").strip()
    match = re.search(r"\b([a-dj])\b", text)
    if match:
        return match.group(1).upper()
    return text[:1].upper() if text else ""


def try_extract_json(text: str) -> Any:
    text = text.strip()
    if not text:
        raise ValueError("Empty text.")
    # First: direct parse
    try:
        return json.loads(text)
    except Exception:
        pass
    # Second: fenced block
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.S)
    if fence:
        try:
            return json.loads(fence.group(1))
        except Exception:
            pass
    # Third: longest bracketed object/array
    candidates = re.findall(r"(\{.*\}|\[.*\])", text, flags=re.S)
    for candidate in sorted(candidates, key=len, reverse=True):
        try:
            return json.loads(candidate)
        except Exception:
            continue
    raise ValueError("Could not extract JSON from text.")


def save_temp_image(image: Any, out_dir: str | os.PathLike[str], stem: str) -> str:
    ensure_dir(out_dir)
    path = Path(out_dir) / f"{stem}.png"
    if isinstance(image, Image.Image):
        image.save(path)
        return str(path)
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(path)
        return str(path)
    raise TypeError(f"Unsupported image type: {type(image)}")


def save_temp_audio(audio_obj: Any, out_dir: str | os.PathLike[str], stem: str) -> str:
    ensure_dir(out_dir)
    path = Path(out_dir) / f"{stem}.wav"
    if isinstance(audio_obj, dict) and "array" in audio_obj:
        array = np.asarray(audio_obj["array"])
        sr = int(audio_obj.get("sampling_rate", 16000))
        sf.write(path, array, sr)
        return str(path)
    if isinstance(audio_obj, (list, tuple, np.ndarray)):
        array = np.asarray(audio_obj)
        sf.write(path, array, 16000)
        return str(path)
    if isinstance(audio_obj, str) and os.path.exists(audio_obj):
        return audio_obj
    raise TypeError(f"Unsupported audio type: {type(audio_obj)}")


def first_existing(row: Dict[str, Any], candidates: Sequence[str]) -> Any:
    for key in candidates:
        if key in row and row[key] is not None:
            return row[key]
    return None


def flatten_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        out: List[Any] = []
        for item in x:
            if isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out
    return [x]


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def chunked(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]
