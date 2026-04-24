from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import os
import sys


DEFAULT_REPO_PATH = (
    Path(__file__).resolve().parents[2]
    / "external"
    / "AdvancedLiterateMachinery"
    / "DocumentUnderstanding"
    / "GeoLayoutLM"
)

DEFAULT_CONFIG_JSON = DEFAULT_REPO_PATH / "configs" / "GeoLayoutLM" / "GeoLayoutLM_large_model_config.json"
DEFAULT_CLASS_NAMES = ("O", "HEADER", "QUESTION", "ANSWER")


@dataclass(frozen=True)
class OCRWord:
    text: str
    token_ids: list[int]
    bbox_4p: list[list[int]]
    block_key: tuple[int, int, int]
    line_key: tuple[int, int, int]
    line_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class GeoLayoutPrediction:
    bbox: tuple[float, float, float, float]
    kind: str
    score: float | None = None
    label: str | None = None


def _repo_path() -> Path:
    configured = os.environ.get("ESGREVEAL_GEOLAYOUTLM_PATH", "").strip()
    if configured:
        return Path(configured)
    return DEFAULT_REPO_PATH


def _ensure_repo_path() -> Path:
    repo_path = _repo_path()
    repo_str = str(repo_path)
    if repo_path.exists() and repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    return repo_path


def _checkpoint_path() -> Path | None:
    configured = os.environ.get("ESGREVEAL_GEOLAYOUTLM_CHECKPOINT", "").strip()
    if not configured:
        return None
    path = Path(configured)
    return path if path.exists() else None


def _device() -> str:
    preferred = os.environ.get("ESGREVEAL_GEOLAYOUTLM_DEVICE", "").strip().lower()
    if preferred:
        return preferred
    try:
        import torch  # type: ignore
    except Exception:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _ocr_lang() -> str:
    return os.environ.get("ESGREVEAL_GEOLAYOUTLM_OCR_LANG", "kor+eng").strip() or "kor+eng"


def _max_seq_length() -> int:
    raw = os.environ.get("ESGREVEAL_GEOLAYOUTLM_MAX_SEQ_LENGTH", "512").strip()
    try:
        return max(32, int(raw))
    except Exception:
        return 512


def _max_block_num() -> int:
    raw = os.environ.get("ESGREVEAL_GEOLAYOUTLM_MAX_BLOCK_NUM", "256").strip()
    try:
        return max(16, int(raw))
    except Exception:
        return 256


def _img_size() -> tuple[int, int]:
    raw_h = os.environ.get("ESGREVEAL_GEOLAYOUTLM_IMG_H", "768").strip()
    raw_w = os.environ.get("ESGREVEAL_GEOLAYOUTLM_IMG_W", "768").strip()
    try:
        return max(64, int(raw_h)), max(64, int(raw_w))
    except Exception:
        return 768, 768


@lru_cache(maxsize=1)
def _repo_modules() -> dict[str, Any] | None:
    repo_path = _ensure_repo_path()
    if not repo_path.exists():
        return None
    try:
        from omegaconf import OmegaConf  # type: ignore
        from model import get_model  # type: ignore
        from bros import BrosTokenizer  # type: ignore
    except Exception:
        return None
    return {
        "OmegaConf": OmegaConf,
        "get_model": get_model,
        "BrosTokenizer": BrosTokenizer,
    }


def _load_model_weight(net: Any, pretrained_model_file: str) -> None:
    import torch  # type: ignore

    pretrained_model_state_dict = torch.load(pretrained_model_file, map_location="cpu")
    if isinstance(pretrained_model_state_dict, dict) and "state_dict" in pretrained_model_state_dict:
        pretrained_model_state_dict = pretrained_model_state_dict["state_dict"]

    valid_keys = net.state_dict().keys()
    new_state_dict: dict[str, Any] = {}
    for key, value in pretrained_model_state_dict.items():
        new_key = key[4:] if key.startswith("net.") else key
        if new_key in valid_keys:
            new_state_dict[new_key] = value
    net.load_state_dict(new_state_dict, strict=False)


@lru_cache(maxsize=1)
def _bundle() -> dict[str, Any] | None:
    modules = _repo_modules()
    checkpoint = _checkpoint_path()
    if modules is None or checkpoint is None:
        return None

    try:
        import torch  # type: ignore
    except Exception:
        return None

    OmegaConf = modules["OmegaConf"]
    get_model = modules["get_model"]
    BrosTokenizer = modules["BrosTokenizer"]
    config_json = Path(os.environ.get("ESGREVEAL_GEOLAYOUTLM_CONFIG_JSON", str(DEFAULT_CONFIG_JSON)))

    cfg = OmegaConf.create(
        {
            "pretrained_model_file": str(checkpoint),
            "model": {
                "backbone": os.environ.get(
                    "ESGREVEAL_GEOLAYOUTLM_BACKBONE",
                    "alibaba-damo/geolayoutlm-large-uncased",
                ),
                "config_json": str(config_json),
                "model_ckpt": str(checkpoint),
                "head": "vie",
                "n_classes": 7,
                "use_inner_id": True,
                "max_prob_as_father": True,
                "max_prob_as_father_upperbound": False,
            },
        }
    )
    try:
        model = get_model(cfg)
        _load_model_weight(model, str(checkpoint))
        model.to(_device())
        model.eval()
        tokenizer = BrosTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    except Exception:
        return None

    return {
        "torch": torch,
        "model": model,
        "tokenizer": tokenizer,
        "device": _device(),
    }


def _normalize_box_4p(box: tuple[int, int, int, int]) -> list[list[int]]:
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def _ocr_words(image: Any, tokenizer: Any) -> list[OCRWord]:
    try:
        import pytesseract  # type: ignore
    except Exception:
        return []

    data = pytesseract.image_to_data(image, lang=_ocr_lang(), output_type=pytesseract.Output.DICT)
    words: list[OCRWord] = []
    for idx, text in enumerate(data.get("text", [])):
        token = str(text or "").strip()
        if not token:
            continue
        left = int(data["left"][idx])
        top = int(data["top"][idx])
        width = int(data["width"][idx])
        height = int(data["height"][idx])
        if width <= 0 or height <= 0:
            continue
        box = (left, top, left + width, top + height)
        token_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(token))
        words.append(
            OCRWord(
                text=token,
                token_ids=token_ids or [tokenizer.unk_token_id],
                bbox_4p=_normalize_box_4p(box),
                block_key=(
                    int(data.get("block_num", [0])[idx]),
                    int(data.get("par_num", [0])[idx]),
                    int(data.get("line_num", [0])[idx]),
                ),
                line_key=(
                    int(data.get("block_num", [0])[idx]),
                    int(data.get("par_num", [0])[idx]),
                    int(data.get("line_num", [0])[idx]),
                ),
                line_box=box,
            )
        )
    return words


def _merge_box(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _line_boxes(words: list[OCRWord]) -> dict[tuple[int, int, int], tuple[int, int, int, int]]:
    merged: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
    for word in words:
        current = merged.get(word.line_key)
        merged[word.line_key] = word.line_box if current is None else _merge_box(current, word.line_box)
    return merged


def _block_boxes(words: list[OCRWord]) -> tuple[list[int], list[list[int]], list[tuple[int, int, int]]]:
    first_token_idx_list: list[int] = []
    boxes: list[list[int]] = []
    ordered_keys: list[tuple[int, int, int]] = []
    token_index = 0
    seen: set[tuple[int, int, int]] = set()
    line_box_map = _line_boxes(words)
    for word in words:
        key = word.line_key
        if key not in seen:
            seen.add(key)
            ordered_keys.append(key)
            first_token_idx_list.append(token_index + 1)
            box = line_box_map[key]
            boxes.append([box[0], box[1], box[2], box[3]])
        token_index += len(word.token_ids)
    return first_token_idx_list, boxes, ordered_keys


def _empty_batch_arrays(max_seq_length: int, max_block_num: int) -> dict[str, Any]:
    import numpy as np  # type: ignore

    return {
        "input_ids": np.zeros(max_seq_length, dtype=int),
        "bbox_4p_normalized": np.zeros((max_seq_length, 8), dtype=np.float32),
        "attention_mask": np.zeros(max_seq_length, dtype=int),
        "first_token_idxes": np.zeros(max_block_num, dtype=int),
        "block_mask": np.zeros(max_block_num, dtype=int),
        "bbox": np.zeros((max_seq_length, 4), dtype=np.float32),
        "line_rank_id": np.zeros(max_seq_length, dtype="int32"),
        "line_rank_inner_id": np.ones(max_seq_length, dtype="int32"),
        "are_box_first_tokens": np.zeros(max_seq_length, dtype=np.bool_),
        "bio_labels": np.ones(max_seq_length, dtype=int) * -100,
        "el_labels_seq": np.zeros((max_seq_length, max_seq_length), dtype=np.float32),
        "el_label_seq_mask": np.zeros((max_seq_length, max_seq_length), dtype=np.float32),
        "el_labels_blk": np.zeros((max_block_num, max_block_num), dtype=np.float32),
        "el_label_blk_mask": np.zeros((max_block_num, max_block_num), dtype=np.float32),
    }


def _build_batch(image: Any, tokenizer: Any) -> tuple[dict[str, Any], list[tuple[int, int, int]], list[OCRWord]]:
    import cv2  # type: ignore
    import numpy as np  # type: ignore
    import torch  # type: ignore

    words = _ocr_words(image, tokenizer)
    if not words:
        return {}, [], []

    width, height = image.size
    img_h, img_w = _img_size()
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image_np = cv2.resize(image_np, (img_w, img_h)).astype("float32").transpose(2, 0, 1)

    max_seq_length = _max_seq_length()
    max_block_num = _max_block_num()
    arrays = _empty_batch_arrays(max_seq_length, max_block_num)
    arrays["image"] = image_np
    arrays["size_raw"] = np.array([width, height])

    first_token_idx_list, block_boxes, ordered_keys = _block_boxes(words)
    cls_token_id = tokenizer.cls_token_id
    sep_token_id = tokenizer.sep_token_id
    pad_token_id = tokenizer.pad_token_id or 0

    list_tokens: list[int] = []
    list_bbs: list[list[int]] = []
    word_first_token_positions: list[int] = []
    word_block_boxes: list[list[int]] = []

    line_key_to_box = {key: box for key, box in zip(ordered_keys, block_boxes, strict=False)}
    for word in words:
        token_ids = word.token_ids
        if len(list_tokens) + len(token_ids) > max_seq_length - 2:
            break
        word_first_token_positions.append(len(list_tokens) + 1)
        list_tokens.extend(token_ids)
        flat_box = [coord for point in word.bbox_4p for coord in point]
        list_bbs.extend([flat_box for _ in token_ids])
        block_box = line_key_to_box[word.line_key]
        word_block_boxes.extend([block_box for _ in token_ids])

    if not list_tokens:
        return {}, [], []

    valid_block_count = 0
    truncated_first_tokens: list[int] = []
    truncated_boxes: list[list[int]] = []
    for idx, first_token_idx in enumerate(first_token_idx_list):
        if idx >= max_block_num or first_token_idx > len(list_tokens):
            break
        truncated_first_tokens.append(first_token_idx)
        truncated_boxes.append(block_boxes[idx])
        valid_block_count += 1

    cls_bbs = [0.0] * 8
    sep_bbs = [width, height] * 4
    cls_bbs_blk = [0, 0, 0, 0]
    sep_bbs_blk = [width, height, width, height]

    list_tokens = [cls_token_id] + list_tokens + [sep_token_id]
    list_bbs = [cls_bbs] + list_bbs + [sep_bbs]
    word_block_boxes = [cls_bbs_blk] + word_block_boxes + [sep_bbs_blk]
    length = len(list_tokens)

    arrays["input_ids"][:] = pad_token_id
    arrays["input_ids"][:length] = list_tokens
    arrays["attention_mask"][:length] = 1
    arrays["first_token_idxes"][:valid_block_count] = truncated_first_tokens
    arrays["block_mask"][:valid_block_count] = 1
    arrays["are_box_first_tokens"][word_first_token_positions] = True

    first_token_ext = truncated_first_tokens + [len(list_tokens)]
    line_id = 1
    for block_index in range(len(first_token_ext) - 1):
        start = first_token_ext[block_index]
        end = first_token_ext[block_index + 1]
        token_span = end - start
        arrays["line_rank_id"][start:end] = line_id
        line_id += 1
        if token_span > 1:
            arrays["line_rank_inner_id"][start:end] = [1] + [2] * (token_span - 2) + [3]

    bbox_4p_normalized = arrays["bbox_4p_normalized"]
    bbox_4p_normalized[:length, :] = list_bbs
    bbox_4p_normalized[:, [0, 2, 4, 6]] = bbox_4p_normalized[:, [0, 2, 4, 6]] / max(width, 1)
    bbox_4p_normalized[:, [1, 3, 5, 7]] = bbox_4p_normalized[:, [1, 3, 5, 7]] / max(height, 1)
    arrays["bbox_4p_normalized"] = bbox_4p_normalized

    bbox = arrays["bbox"]
    bbox[:length, :] = word_block_boxes
    bbox[:, [0, 2]] = bbox[:, [0, 2]] / max(width, 1) * 1000
    bbox[:, [1, 3]] = bbox[:, [1, 3]] / max(height, 1) * 1000
    arrays["bbox"] = bbox.astype(int)
    arrays["line_rank_inner_id"] = arrays["line_rank_inner_id"] * arrays["attention_mask"]

    batch = {}
    for key, value in arrays.items():
        if isinstance(value, np.ndarray):
            tensor = torch.from_numpy(value).unsqueeze(0)
            batch[key] = tensor
        else:
            batch[key] = torch.tensor(value).unsqueeze(0)
    return batch, ordered_keys[:valid_block_count], words


def _bio_class_names() -> list[str]:
    class_names = list(DEFAULT_CLASS_NAMES)
    bio = ["O"]
    for class_name in class_names:
        if not class_name.startswith("O"):
            bio.extend([f"B-{class_name}", f"I-{class_name}"])
    return bio


def _label_to_kind(label: str) -> str:
    normalized = label.upper()
    if "HEADER" in normalized:
        return "heading"
    return "paragraph"


def _aggregate_predictions(
    logits: Any,
    attention_mask: Any,
    line_keys: list[tuple[int, int, int]],
    words: list[OCRWord],
) -> list[GeoLayoutPrediction]:
    import torch  # type: ignore

    line_box_map = _line_boxes(words)
    probabilities = torch.softmax(logits[0], dim=-1)
    predictions = probabilities.argmax(-1).tolist()
    scores = probabilities.max(-1).values.tolist()
    attention = attention_mask[0].tolist()
    first_positions: list[int] = []
    running = 1
    for word in words:
        if running >= len(predictions) - 1:
            break
        first_positions.append(running)
        running += len(word.token_ids)

    bio_names = _bio_class_names()
    per_line_labels: dict[tuple[int, int, int], list[tuple[str, float]]] = {}
    for word, token_pos in zip(words, first_positions, strict=False):
        if token_pos >= len(attention) or attention[token_pos] == 0:
            continue
        label = bio_names[predictions[token_pos]]
        score = float(scores[token_pos])
        per_line_labels.setdefault(word.line_key, []).append((label, score))

    blocks: list[GeoLayoutPrediction] = []
    for key in line_keys:
        labels = per_line_labels.get(key, [])
        if not labels:
            continue
        header_hits = [score for label, score in labels if "HEADER" in label]
        if header_hits:
            kind = "heading"
            score = sum(header_hits) / len(header_hits)
            label = "HEADER"
        else:
            kind = "paragraph"
            score = sum(score for _, score in labels) / len(labels)
            label = max(labels, key=lambda item: item[1])[0]
        box = line_box_map[key]
        blocks.append(
            GeoLayoutPrediction(
                bbox=(float(box[0]), float(box[1]), float(box[2]), float(box[3])),
                kind=kind,
                score=score,
                label=label,
            )
        )
    return blocks


def predict_layout(*, image: Any, page_num: int, page: Any | None = None) -> list[GeoLayoutPrediction]:
    bundle = _bundle()
    if bundle is None:
        return []

    batch, line_keys, words = _build_batch(image, bundle["tokenizer"])
    if not batch or not line_keys or not words:
        return []

    torch = bundle["torch"]
    model = bundle["model"]
    device = bundle["device"]
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)

    try:
        with torch.no_grad():
            head_outputs, _ = model(batch)
    except Exception:
        return []

    logits = head_outputs.get("logits4labeling")
    attention_mask = batch.get("attention_mask")
    if logits is None or attention_mask is None:
        return []
    return _aggregate_predictions(logits, attention_mask, line_keys, words)
