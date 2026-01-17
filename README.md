# Chessboard Reconstruction from Video Frames

This project builds a pipeline to reconstruct chessboard states from tagged video
frames. It extracts square crops, trains a classifier, calibrates a threshold,
then reconstructs boards and visualizes predictions.

## Repository layout
- `scripts/`: pipeline scripts
- `data/`: input data (CSV + frame images)
- `cache/`: generated indexes and square crops
- `models/`: trained model artifacts and metadata
- `results/`: inference outputs and debug files

## Requirements
Python 3.9+ with the following packages:
- numpy, pandas, pillow, tqdm
- torch, torchvision
- matplotlib
- python-chess, cairosvg

Tip: use a virtual environment and install your CUDA-enabled PyTorch build
if you have a GPU available.

## Expected data layout
```
data/
  csv_games/
    <game_id>/
      <game>.csv
      tagged_images/
        frame_000001.jpg
        frame_000002.jpg
        ...
  occluded_frames/
    <game_id>/
      tagged_images/
        frame_000123.jpg
        ...
```

The CSV files must include columns: `from_frame`, `to_frame`, and `fen`.

## Pipeline overview
Run the scripts in this order:

1) Build frame index and dataset splits
```
python scripts/01_build_csv_index.py
```
Outputs:
- `cache/frames_index.csv`
- `cache/conflicting_labels.csv`
- `cache/missing_frames.csv`

2) Expand FEN labels to per-square labels
```
python scripts/02_fen_to_square_labels.py
```
Outputs:
- `cache/frames_with_square_labels.csv`

3) Extract square crops (1.5x context) and build crop index
```
python scripts/03_extract_square_crops.py
```
Outputs:
- `cache/square_crops/` images
- `cache/square_crops_index.csv`

4) Train square classifier
```
python scripts/04_train_square_classifier.py
```
Outputs:
- `models/square_classifier.pt`
- `models/normalization.json`
- `models/label_map.json`
- `models/train_valid_loss.png`

5) Run inference and reconstruct boards
```
python scripts/05_inference_and_fen_reconstruction.py --split test
```
Outputs:
- `results/<game_id>_<frame_id>.png`
- `results/inference_debug.csv`

6) Visualize side-by-side predictions
```
python scripts/06_visualize_frame_prediction.py
```
Outputs:
- `results/side_by_side/*_side_by_side.png`

