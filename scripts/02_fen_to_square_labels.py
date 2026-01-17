#!/usr/bin/env python3
from pathlib import Path
import pandas as pd
import chess


# --- Project paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

CACHE_DIR = PROJECT_ROOT / "cache"

FRAMES_INDEX_PATH = CACHE_DIR / "frames_index.csv"
OUTPUT_PATH = CACHE_DIR / "frames_with_square_labels.csv"


# --- Constants ---
# Square order: a8 -> h8, a7 -> h7, ..., a1 -> h1
# We store labels as sq_00 ... sq_63 following this order.
SQUARE_COLUMNS = [f"sq_{i:02d}" for i in range(64)]


# --- Core logic ---
def fen_to_square_labels(fen: str):
    """
    Convert a FEN string into a list of 64 square labels
    in a8 -> h1 order.

    Returns:
        List[str] of length 64, each element is:
        'empty' or one of {'P','N','B','R','Q','K','p','n','b','r','q','k'}
    """
    board = chess.Board(fen)
    labels = []

    # Rank 8 down to 1
    for rank in range(8, 0, -1):
        # File a (0) to h (7)
        for file in range(8):
            square = chess.square(file, rank - 1)
            piece = board.piece_at(square)
            labels.append("empty" if piece is None else piece.symbol())

    assert len(labels) == 64
    return labels


def build_square_label_table(df_index: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df_index.iterrows():
        labels = fen_to_square_labels(r["fen"])

        row = {
            "game_id": r["game_id"],
            "frame_id": int(r["frame_id"]),
            "img_path": r["img_path"],
            "fen": r["fen"],
            "is_occluded": r["is_occluded"],
            "split": r["split"],
        }

        for col, val in zip(SQUARE_COLUMNS, labels):
            row[col] = val

        rows.append(row)

    df_out = pd.DataFrame(rows)

    # Deterministic ordering
    df_out = (
        df_out
        .sort_values(["game_id", "frame_id"])
        .reset_index(drop=True)
    )

    return df_out


# --- Main ---

def main():
    print(f"[INFO] Loading frame index: {FRAMES_INDEX_PATH}")
    df_index = pd.read_csv(FRAMES_INDEX_PATH)

    print("[INFO] Converting FEN to square labels...")
    df_out = build_square_label_table(df_index)

    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"[INFO] Saved square-label table: {OUTPUT_PATH}")
    print(f"[INFO] Total frames: {len(df_out)}")


if __name__ == "__main__":
    main()
