#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
from PIL import Image


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
RESULTS_DIR = PROJECT_ROOT / "results"
OUT_DIR = RESULTS_DIR / "side_by_side"
OUT_DIR.mkdir(exist_ok=True)

DEBUG_CSV = RESULTS_DIR / "inference_debug.csv"
FRAMES_INDEX_CSV = CACHE_DIR / "frames_index.csv"


# Helpers
def make_side_by_side(img_left, img_right, gap=20, bg_color=(255, 255, 255)):
    h = max(img_left.height, img_right.height)
    w = img_left.width + img_right.width + gap

    canvas = Image.new("RGB", (w, h), bg_color)
    canvas.paste(img_left, (0, 0))
    canvas.paste(img_right, (img_left.width + gap, 0))
    return canvas


def pad_image(img, pad=20, color=(0, 0, 0)):
    w, h = img.size
    padded = Image.new("RGB", (w + 2*pad, h + 2*pad), color)
    padded.paste(img, (pad, pad))
    return padded


# Main
def main():
    print("[INFO] Loading CSV files")

    debug_df = pd.read_csv(DEBUG_CSV)
    frames_df = pd.read_csv(FRAMES_INDEX_CSV)

    # Build lookup: (game_id, frame_id) -> img_path
    frame_path_map = {
        (row.game_id, row.frame_id): row.img_path
        for _, row in frames_df.iterrows()
    }

    print(f"[INFO] Found {len(debug_df)} inference results")

    success = 0
    skipped = 0

    for _, row in debug_df.iterrows():
        game_id = row.game_id
        frame_id = row.frame_id

        key = (game_id, frame_id)
        if key not in frame_path_map:
            print(f"[WARN] Missing raw frame for {game_id}_{frame_id}")
            skipped += 1
            continue

        raw_frame_path = Path(frame_path_map[key])
        board_png_path = RESULTS_DIR / f"{game_id}_{frame_id}.png"

        if not raw_frame_path.exists():
            print(f"[WARN] Raw frame not found: {raw_frame_path}")
            skipped += 1
            continue

        if not board_png_path.exists():
            print(f"[WARN] Board image not found: {board_png_path}")
            skipped += 1
            continue

        try:
            raw_img = Image.open(raw_frame_path).convert("RGB")
            raw_img = pad_image(raw_img, pad=20, color=(0, 0, 0))
            board_img = Image.open(board_png_path).convert("RGB")

            combined = make_side_by_side(raw_img, board_img)

            out_path = OUT_DIR / f"{game_id}_{frame_id}_side_by_side.png"
            combined.save(out_path)

            success += 1

        except Exception as e:
            print(f"[WARN] Failed {game_id}_{frame_id}: {e}")
            skipped += 1

    print("\n[SUMMARY]")
    print(f"  Saved:   {success}")
    print(f"  Skipped: {skipped}")
    print(f"[INFO] Output directory: {OUT_DIR}")


if __name__ == "__main__":
    main()
