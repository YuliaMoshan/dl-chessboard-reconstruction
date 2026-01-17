#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import numpy as np


# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CSV_GAMES_DIR = DATA_DIR / "csv_games"
OCCLUDED_FRAMES_DIR = DATA_DIR / "occluded_frames"
CACHE_DIR = PROJECT_ROOT / "cache"

FRAMES_INDEX_PATH = CACHE_DIR / "frames_index.csv"
CONFLICTING_LABELS_PATH = CACHE_DIR / "conflicting_labels.csv"
MISSING_FRAMES_PATH = CACHE_DIR / "missing_frames.csv"

CACHE_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
TRAIN_RATIO = 0.70
VALID_RATIO = 0.15
TEST_RATIO  = 0.15


# --- Core logic ---
def collect_occluded_relative_paths(occluded_root: Path) -> set[str]:
    """
    Collect occluded frame paths as normalized relative paths:
    <game_id>/tagged_images/frame_xxxxxx.jpg
    """
    occluded = set()
    if not occluded_root.exists():
        return occluded

    for img_path in occluded_root.rglob("frame_*.jpg"):
        rel = img_path.relative_to(occluded_root)

        # Expect: <game_id>/tagged_images/frame_xxxxxx.jpg
        if len(rel.parts) != 3:
            continue

        game_id, tagged_images, filename = rel.parts
        if tagged_images != "tagged_images":
            continue

        occluded.add(f"{game_id}/tagged_images/{filename}")

    return occluded


def build_csv_index(csv_games_dir: Path, occluded_paths: set[str]):
    """
    Build a DataFrame index from CSV game directories.
    Detect conflicting labels and missing images.
    """
    rows = []
    conflict_rows = []
    missing_rows = []

    for game_dir in sorted(p for p in csv_games_dir.iterdir() if p.is_dir()):
        csv_files = list(game_dir.glob("*.csv"))
        if len(csv_files) != 1:
            raise RuntimeError(
                f"Expected exactly 1 CSV in {game_dir}, found {len(csv_files)}"
            )
        csv_path = csv_files[0]
        df = pd.read_csv(csv_path)

        required_cols = {"from_frame", "to_frame", "fen"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"CSV schema mismatch in {csv_path}, missing: {missing}"
            )

        # Detect conflicting labels
        fen_counts = df.groupby("from_frame")["fen"].nunique()
        conflicting_frame_ids = set(fen_counts[fen_counts > 1].index.tolist())

        for frame_id in conflicting_frame_ids:
            fens = (
                df[df["from_frame"] == frame_id]["fen"]
                .drop_duplicates()
                .tolist()
            )
            conflict_rows.append({
                "game_id": game_dir.name,
                "frame_id": int(frame_id),
                "fen_1": fens[0],
                "fen_2": fens[1],
                "csv_path": str(csv_path),
            })

        images_dir = game_dir / "tagged_images"
        if not images_dir.exists():
            raise FileNotFoundError(f"Missing images folder: {images_dir}")

        for _, r in df.iterrows():
            frame_id = int(r["from_frame"])
            if frame_id in conflicting_frame_ids:
                continue

            img_path = images_dir / f"frame_{frame_id:06d}.jpg"
            if not img_path.exists():
                missing_rows.append({
                    "game_id": game_dir.name,
                    "frame_id": frame_id,
                    "expected_img_path": str(img_path),
                    "csv_path": str(csv_path),
                })
                continue

            rel_key = f"{game_dir.name}/tagged_images/{img_path.name}"
            is_occluded = rel_key in occluded_paths

            rows.append({
                "game_id": game_dir.name,
                "frame_id": frame_id,
                "img_path": str(img_path),
                "fen": r["fen"],
                "is_occluded": is_occluded,
            })

    df_index = (
        pd.DataFrame(rows)
        .sort_values(["game_id", "frame_id"])
        .reset_index(drop=True)
    )

    df_conflicts = pd.DataFrame(conflict_rows)
    df_missing = pd.DataFrame(missing_rows)

    return df_index, df_conflicts, df_missing


def assign_splits(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign dataset splits.
    Occluded frames -> test
    Remaining frames -> 70/15/15 split
    """
    df = df.copy()
    df["split"] = None

    # Occluded frames
    df.loc[df.is_occluded, "split"] = "test"

    # Split remaining
    clean_idx = df[df.is_occluded == False].index.to_numpy()
    rng = np.random.default_rng(RANDOM_SEED)
    rng.shuffle(clean_idx)

    n = len(clean_idx)
    n_train = int(TRAIN_RATIO * n)
    n_valid = int(VALID_RATIO * n)

    train_idx = clean_idx[:n_train]
    valid_idx = clean_idx[n_train:n_train + n_valid]
    test_idx  = clean_idx[n_train + n_valid:]

    df.loc[train_idx, "split"] = "train"
    df.loc[valid_idx, "split"] = "valid"
    df.loc[test_idx,  "split"] = "test"

    return df


# --- Main ---
def main():
    print("[INFO] Collecting occluded frame paths...")
    occluded_paths = collect_occluded_relative_paths(OCCLUDED_FRAMES_DIR)
    print(f"[INFO] Found {len(occluded_paths)} occluded frames")

    print("[INFO] Building CSV frame index...")
    df_index, df_conflicts, df_missing = build_csv_index(CSV_GAMES_DIR, occluded_paths)

    print("[INFO] Assigning dataset splits...")
    df_index = assign_splits(df_index)

    df_index.to_csv(FRAMES_INDEX_PATH, index=False)
    df_conflicts.to_csv(CONFLICTING_LABELS_PATH, index=False)
    df_missing.to_csv(MISSING_FRAMES_PATH, index=False)

    print(f"[INFO] Saved index: {FRAMES_INDEX_PATH} ({len(df_index)} rows)")
    print(f"[INFO] Saved conflicting labels log: {CONFLICTING_LABELS_PATH} ({len(df_conflicts)} rows)")
    print(f"[INFO] Saved missing frames log: {MISSING_FRAMES_PATH} ({len(df_missing)} rows)")


if __name__ == "__main__":
    main()
