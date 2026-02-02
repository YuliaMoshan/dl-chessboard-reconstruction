#!/usr/bin/env python3
"""Run single-image inference and save visualization outputs."""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import cairosvg
import chess
import chess.svg
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch import nn
from torchvision import models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "square_classifier.pt"
NORMALIZATION_PATH = MODELS_DIR / "normalization.json"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"

BOARD_MARGIN = 20  # pixels

NUM_TO_LABEL = {
    0: "P",
    1: "R",
    2: "N",
    3: "B",
    4: "Q",
    5: "K",
    6: "p",
    7: "r",
    8: "n",
    9: "b",
    10: "q",
    11: "k",
    12: "empty",
    13: "X",
}

IMAGE_SIZE: Tuple[int, int] = (96, 96)
GRID_SIZE = 8
CONTEXT_SCALE = 1.5
THRESHOLD = 0.75
OOD_CLASS_IDX = 13

LABEL_TO_OUTPUT = {
    "P": 0,  # White Pawn
    "R": 1,  # White Rook
    "N": 2,  # White Knight
    "B": 3,  # White Bishop
    "Q": 4,  # White Queen
    "K": 5,  # White King
    "p": 6,  # Black Pawn
    "r": 7,  # Black Rook
    "n": 8,  # Black Knight
    "b": 9,  # Black Bishop
    "q": 10, # Black Queen
    "k": 11, # Black King
    "empty": 12,
}

_MODEL = None
_TRANSFORM = None
_IDX_TO_LABEL: Dict[int, str] | None = None
_LABEL_TO_IDX: Dict[str, int] | None = None

SQUARE_ORDER = [
    f"{file}{rank}"
    for rank in range(8, 0, -1)
    for file in "abcdefgh"
]

MAX_COUNTS = {
    "P": 8, "p": 8,
    "R": 2, "r": 2,
    "N": 2, "n": 2,
    "B": 2, "b": 2,
    "Q": 1, "q": 1,
    "K": 1, "k": 1,
}


def _load_assets() -> None:
    global _MODEL, _TRANSFORM, _IDX_TO_LABEL, _LABEL_TO_IDX

    if (
        _MODEL is not None
        and _TRANSFORM is not None
        and _IDX_TO_LABEL is not None
        and _LABEL_TO_IDX is not None
    ):
        return

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")
    if not NORMALIZATION_PATH.exists():
        raise FileNotFoundError(f"Missing normalization stats: {NORMALIZATION_PATH}")
    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(f"Missing label map: {LABEL_MAP_PATH}")

    with open(LABEL_MAP_PATH) as f:
        idx_to_label = json.load(f)
    _IDX_TO_LABEL = {int(k): v for k, v in idx_to_label.items()}
    _LABEL_TO_IDX = {v: k for k, v in _IDX_TO_LABEL.items()}

    with open(NORMALIZATION_PATH) as f:
        norm = json.load(f)

    _TRANSFORM = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])

    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        len(_IDX_TO_LABEL),
    )
    state = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    _MODEL = model


def _extract_square_crops(image: Image.Image) -> List[Image.Image]:
    w, h = image.size
    square_w = w / GRID_SIZE
    square_h = h / GRID_SIZE

    crop_w = square_w * CONTEXT_SCALE
    crop_h = square_h * CONTEXT_SCALE

    crops: List[Image.Image] = []
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            cx = (col + 0.5) * square_w
            cy = (row + 0.5) * square_h

            left = int(round(cx - crop_w / 2))
            top = int(round(cy - crop_h / 2))
            right = int(round(cx + crop_w / 2))
            bottom = int(round(cy + crop_h / 2))

            left = max(0, left)
            top = max(0, top)
            right = min(w, right)
            bottom = min(h, bottom)

            crops.append(image.crop((left, top, right, bottom)))
    return crops


def _repair_board(board_preds, board_probs):
    """
    Rule-based board repair using standard FEN labels.
    Operates only on existing model probabilities.
    """
    repaired = board_preds.copy()

    for label, max_count in MAX_COUNTS.items():
        idx = _LABEL_TO_IDX[label]
        squares = [i for i, p in enumerate(repaired) if p == idx]

        if len(squares) > max_count:
            scored = [(i, board_probs[i][idx]) for i in squares]
            scored.sort(key=lambda x: x[1])  # weakest first

            for i, _prob in scored[:-max_count]:
                repaired[i] = OOD_CLASS_IDX

    return repaired


def predict_board(image: np.ndarray) -> torch.Tensor:
    """
    Predict the chessboard state from a single RGB image.

    Args:
        image: np.ndarray of shape (H, W, 3), dtype uint8, RGB order.

    Returns:
        torch.Tensor of shape (8, 8), dtype torch.int64, on CPU.
    """
    _load_assets()

    if not isinstance(image, np.ndarray):
        raise TypeError("image must be a numpy.ndarray")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must have shape (H, W, 3)")

    if image.dtype != np.uint8:
        image = image.astype(np.uint8)

    pil_img = Image.fromarray(image, mode="RGB")

    crops = _extract_square_crops(pil_img)
    tensors = [_TRANSFORM(c) for c in crops]
    batch = torch.stack(tensors, dim=0)

    with torch.no_grad():
        logits = _MODEL(batch)
        probs = torch.softmax(logits, dim=1)
        max_probs, preds = probs.max(dim=1)

    pred_numeric: List[int] = []
    for prob, idx in zip(max_probs, preds):
        if prob.item() < THRESHOLD:
            pred_numeric.append(OOD_CLASS_IDX)
        else:
            pred_numeric.append(int(idx.item()))

    pred_numeric = _repair_board(pred_numeric, probs.cpu().tolist())

    outputs: List[int] = []
    for idx in pred_numeric:
        if idx == OOD_CLASS_IDX:
            outputs.append(OOD_CLASS_IDX)
        else:
            label = _IDX_TO_LABEL[idx]
            outputs.append(LABEL_TO_OUTPUT[label])

    out_tensor = torch.tensor(outputs, dtype=torch.int64).reshape(GRID_SIZE, GRID_SIZE)
    return out_tensor.cpu()


def labels_to_board_fen(labels_64):
    """
    Convert 64 labels (a8 -> h1) to board-only FEN.
    'X' is treated as empty.
    """
    fen = ""
    empty = 0
    for i, lbl in enumerate(labels_64):
        if lbl in ("empty", "X"):
            empty += 1
        else:
            if empty:
                fen += str(empty)
                empty = 0
            fen += lbl
        if (i + 1) % 8 == 0:
            if empty:
                fen += str(empty)
                empty = 0
            if i != 63:
                fen += "/"
    return fen


def render_board_png(board_fen: str, out_path: Path, size: int = 520):
    full_fen = f"{board_fen} w - - 0 1"
    board = chess.Board(full_fen)
    svg = chess.svg.board(board=board, size=size)
    cairosvg.svg2png(bytestring=svg.encode("utf-8"), write_to=str(out_path))


def draw_x_on_board_png(board_png_path: Path, x_squares, out_path: Path):
    img = Image.open(board_png_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    w, h = img.size
    board_w = w - 2 * BOARD_MARGIN
    board_h = h - 2 * BOARD_MARGIN

    sq_w = board_w / 8
    sq_h = board_h / 8

    for sq in x_squares:
        r, c = sq
        x1 = BOARD_MARGIN + c * sq_w
        y1 = BOARD_MARGIN + r * sq_h
        x2 = BOARD_MARGIN + (c + 1) * sq_w
        y2 = BOARD_MARGIN + (r + 1) * sq_h
        draw.line((x1, y1, x2, y2), fill="red", width=6)
        draw.line((x2, y1, x1, y2), fill="red", width=6)

    img.save(out_path)


def make_side_by_side(img_left, img_right, gap=20, bg_color=(255, 255, 255)):
    h = max(img_left.height, img_right.height)
    w = img_left.width + img_right.width + gap

    canvas = Image.new("RGB", (w, h), bg_color)
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (img_left.width + gap, 0))
    return canvas


def pad_image(img, pad=20, color=(0, 0, 0)):
    w, h = img.size
    padded = Image.new("RGB", (w + 2 * pad, h + 2 * pad), color)
    padded.paste(img, (pad, pad))
    return padded


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-path", required=True, help="Path to an RGB chessboard image")
    parser.add_argument("--out-dir", default=str(RESULTS_DIR), help="Output directory")
    args = parser.parse_args()

    image_path = Path(args.image_path)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img_np = np.asarray(img)

    board_tensor = predict_board(img_np)
    board_array = board_tensor.cpu().numpy()

    labels = [NUM_TO_LABEL[int(v)] for v in board_array.reshape(-1)]
    board_fen = labels_to_board_fen(labels)

    stem = image_path.stem
    board_png = out_dir / f"{stem}_pred.png"
    render_board_png(board_fen, board_png)

    x_squares = [(r, c) for r in range(8) for c in range(8) if board_array[r, c] == 13]
    if x_squares:
        draw_x_on_board_png(board_png, x_squares, board_png)

    raw_img = pad_image(img, pad=20, color=(0, 0, 0))
    board_img = Image.open(board_png).convert("RGB")
    side_by_side = make_side_by_side(raw_img, board_img)
    side_by_side_path = out_dir / f"{stem}_side_by_side.png"
    side_by_side.save(side_by_side_path)

    print(f"[INFO] Saved board render: {board_png}")
    print(f"[INFO] Saved side-by-side: {side_by_side_path}")


if __name__ == "__main__":
    main()
