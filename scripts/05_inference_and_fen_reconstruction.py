#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import pandas as pd
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image, ImageDraw
from tqdm import tqdm
import chess
import chess.svg
import cairosvg


# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

CROPS_INDEX_PATH = CACHE_DIR / "square_crops_index.csv"
MODEL_PATH = MODELS_DIR / "square_classifier.pt"
NORMALIZATION_PATH = MODELS_DIR / "normalization.json"
LABEL_MAP_PATH = MODELS_DIR / "label_map.json"


# --- Constants ---
IMAGE_SIZE = (96, 96)
BOARD_MARGIN = 20  # pixels

THRESHOLD = 0.6
print(f"[INFO] Using confidence threshold: {THRESHOLD}")

X_CLASS_IDX = 13  # unknown / OOD class index
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SQUARE_ORDER = [
    f"{file}{rank}"
    for rank in range(8, 0, -1)
    for file in "abcdefgh"
]

FILE_TO_COL = {c: i for i, c in enumerate("abcdefgh")}
RANK_TO_ROW = {str(r): 8 - r for r in range(1, 9)}  # "8"->0 ... "1"->7

MAX_COUNTS = {
    "P": 8, "p": 8,
    "R": 2, "r": 2,
    "N": 2, "n": 2,
    "B": 2, "b": 2,
    "Q": 1, "q": 1,
    "K": 1, "k": 1,
}


# --- Helpers ---
def square_name_to_rc(sq: str):
    return RANK_TO_ROW[sq[1]], FILE_TO_COL[sq[0]]


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
        r, c = square_name_to_rc(sq)
        x1 = BOARD_MARGIN + c * sq_w
        y1 = BOARD_MARGIN + r * sq_h
        x2 = BOARD_MARGIN + (c + 1) * sq_w
        y2 = BOARD_MARGIN + (r + 1) * sq_h
        draw.line((x1, y1, x2, y2), fill="red", width=6)
        draw.line((x2, y1, x1, y2), fill="red", width=6)

    img.save(out_path)


def repair_board(board_preds, board_probs, label_to_idx, x_idx, square_order):
    """
    Rule-based board repair using standard FEN labels.
    Operates only on existing model probabilities.
    """
    repaired = board_preds.copy()

    repair_log = []

    for label, max_count in MAX_COUNTS.items():
        idx = label_to_idx[label]
        squares = [i for i, p in enumerate(repaired) if p == idx]

        if len(squares) > max_count:
            scored = [(i, board_probs[i][idx]) for i in squares]
            scored.sort(key=lambda x: x[1])  # weakest first

            for i, prob in scored[:-max_count]:
                repaired[i] = x_idx
                repair_log.append(
                    f"{square_order[i]}: {label} → X (p={prob:.3f})"
                )

    return repaired, repair_log


# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", nargs="+", default=["test"], choices=["train", "valid", "test", "all"],
        help="Which dataset splits to run on (default: test)"
    )
    args = parser.parse_args()


    # Load model + label map
    with open(LABEL_MAP_PATH) as f:
        idx_to_label = json.load(f)
    idx_to_label = {int(k): v for k, v in idx_to_label.items()}
    label_to_idx = {v: k for k, v in idx_to_label.items()}

    model = models.convnext_tiny(weights=None)
    model.classifier[2] = nn.Linear(
        model.classifier[2].in_features,
        len(idx_to_label)
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()

    with open(NORMALIZATION_PATH) as f:
        norm = json.load(f)

    transform = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=norm["mean"], std=norm["std"]),
    ])


    # Load data
    df = pd.read_csv(CROPS_INDEX_PATH)

    if "all" not in args.split:
        df = df[df["split"].isin(args.split)]

    print(f"[INFO] Running inference on splits: {args.split}")

    grouped = df.groupby(["game_id", "frame_id"])


    # Metrics + debug
    square_correct = 0
    square_total = 0
    square_correct_no_empty = 0
    square_total_no_empty = 0
    board_exact_match = 0
    board_total = 0

    debug_rows = []


    # Inference loop
    with torch.no_grad():
        for (game_id, frame_id), g in tqdm(grouped, desc="Reconstructing boards"):
            g = g.set_index("square_name").loc[SQUARE_ORDER]

            imgs = []
            for p in g["crop_path"]:
                img = Image.open(p).convert("RGB")
                imgs.append(transform(img))

            imgs = torch.stack(imgs).to(DEVICE)

            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            max_probs, preds = probs.max(dim=1)

            pred_labels = []
            pred_numeric = []

            for prob, idx in zip(max_probs, preds):
                if prob.item() < THRESHOLD:
                    pred_labels.append("X")
                    pred_numeric.append(X_CLASS_IDX)
                else:
                    lbl = idx_to_label[idx.item()]
                    pred_labels.append(lbl)
                    pred_numeric.append(idx.item())

            # --- Board-level rule-based repair ---
            pred_numeric, repair_log  = repair_board(
                board_preds=pred_numeric,
                board_probs=probs.cpu().tolist(),
                label_to_idx=label_to_idx,
                x_idx=X_CLASS_IDX,
                square_order=SQUARE_ORDER,
            )

            # rebuild labels after repair
            pred_labels = [
                "X" if idx == X_CLASS_IDX else idx_to_label[idx]
                for idx in pred_numeric
            ]

            x_squares = [
                sq for sq, cls in zip(SQUARE_ORDER, pred_numeric)
                if cls == X_CLASS_IDX
            ]

            board_fen = labels_to_board_fen(pred_labels)

            out_base = RESULTS_DIR / f"{game_id}_{frame_id}"
            board_png = out_base.with_name(out_base.name + ".png")

            render_board_png(board_fen, board_png)
            draw_x_on_board_png(board_png, x_squares, board_png)


            gt_labels = g["label"].tolist()

            # square metrics
            for gt, pr in zip(gt_labels, pred_labels):
                square_total += 1
                if gt == pr:
                    square_correct += 1
                if gt != "empty":
                    square_total_no_empty += 1
                    if gt == pr:
                        square_correct_no_empty += 1

            board_correct = all(
                (p == g or p == "X")
                for p, g in zip(pred_labels, gt_labels)
            )

            if board_correct:
                board_exact_match += 1
            board_total += 1

            wrong_squares = [
                sq for sq, p, g in zip(SQUARE_ORDER, pred_labels, gt_labels)
                if (p != g and p != "X")
            ]

            x_squares_list = [
                sq for sq, cls in zip(SQUARE_ORDER, pred_numeric)
                if cls == X_CLASS_IDX
            ]

            debug_rows.append({
                "game_id": game_id,
                "frame_id": frame_id,
                "is_occluded": g["is_occluded"].iloc[0],
                "split": g["split"].iloc[0],
                "board_correct": board_correct,
                "x_squares": ",".join(x_squares_list),
                "wrong_squares": ",".join(wrong_squares),
                "num_square_errors": len(wrong_squares),
                "repair_applied": len(repair_log) > 0,
                "repair_log": " | ".join(repair_log),
            })


    # Results
    print("\n[BOARD RECONSTRUCTION RESULTS]")
    print(f"Square acc (all):        {square_correct / square_total:.4f}")
    print(f"Square acc (no empty):   {square_correct_no_empty / square_total_no_empty:.4f}")
    print(f"Exact board match rate:  {board_exact_match / board_total:.4f}")
    print(f"Total boards evaluated:  {board_total}")

    debug_df = pd.DataFrame(debug_rows)
    debug_csv = RESULTS_DIR / "inference_debug.csv"
    debug_df.to_csv(debug_csv, index=False)

    print(f"[INFO] Saved debug CSV to {debug_csv}")


if __name__ == "__main__":
    main()