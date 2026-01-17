#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm


# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"
CROPS_DIR = CACHE_DIR / "square_crops"
CROPS_DIR.mkdir(exist_ok=True)

INPUT_CSV = CACHE_DIR / "frames_with_square_labels.csv"
OUTPUT_CSV = CACHE_DIR / "square_crops_index.csv"


# --- Constants ---
GRID_SIZE = 8
CONTEXT_SCALE = 1.5   # 1.5x square size (0.25x padding around)

# Square names: a8 -> h1
FILES = "abcdefgh"
RANKS = "87654321"
SQUARE_NAMES = [f + r for r in RANKS for f in FILES]
SQUARE_COLUMNS = [f"sq_{i:02d}" for i in range(64)]


# --- Core logic ---
def extract_square_crops(df: pd.DataFrame):
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Extracting square crops"):
        img = Image.open(r["img_path"]).convert("RGB")
        w, h = img.size

        square_w = w / GRID_SIZE
        square_h = h / GRID_SIZE

        crop_w = square_w * CONTEXT_SCALE
        crop_h = square_h * CONTEXT_SCALE

        for idx in range(64):
            row_idx = idx // GRID_SIZE
            col_idx = idx % GRID_SIZE

            # Square center
            cx = (col_idx + 0.5) * square_w
            cy = (row_idx + 0.5) * square_h

            left = int(cx - crop_w / 2)
            top = int(cy - crop_h / 2)
            right = int(cx + crop_w / 2)
            bottom = int(cy + crop_h / 2)

            # Clamp to image bounds
            left = max(left, 0)
            top = max(top, 0)
            right = min(right, w)
            bottom = min(bottom, h)

            crop = img.crop((left, top, right, bottom))

            crop_name = (
                f"{r['game_id']}_"
                f"frame{int(r['frame_id']):06d}_"
                f"sq_{idx:02d}.jpg"
            )
            crop_path = CROPS_DIR / crop_name
            crop.save(crop_path)

            rows.append({
                "game_id": r["game_id"],
                "frame_id": int(r["frame_id"]),
                "square_idx": idx,
                "square_name": SQUARE_NAMES[idx],
                "label": r[SQUARE_COLUMNS[idx]],
                "crop_path": str(crop_path),
                "is_occluded": r["is_occluded"],
                "split": r["split"],
            })

    return pd.DataFrame(rows)


# --- Main ---
def main():
    print(f"[INFO] Loading labeled frames: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

    print("[INFO] Extracting square crops...")
    df_crops = extract_square_crops(df)

    df_crops.to_csv(OUTPUT_CSV, index=False)

    print(f"[INFO] Saved crop index: {OUTPUT_CSV}")
    print(f"[INFO] Total square crops: {len(df_crops)}")


if __name__ == "__main__":
    main()
