# Scrabble Player Rating

PRML project for predicting human Scrabble player ratings from game metadata and turn-level gameplay logs.

## Repository Structure

```text
Scrabble_Player_Rating/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
├── scripts/
├── src/
├── .gitignore
└── README.md
```

## Dataset Access

The raw dataset is not stored in GitHub. All team members should use the same shared Drive folder:

<https://drive.google.com/drive/folders/1SxC27uGBSb4679CgFDrC5ymUvmSfBN7a?usp=drive_link>

Expected raw archive filename:

`scrabble-player-rating.zip`

Place the downloaded zip file here:

`data/raw/scrabble-player-rating.zip`

After that, run:

```bash
python3 scripts/setup_data.py
```

This will extract the dataset into:

`data/raw/scrabble-player-rating/`

Expected extracted files:

- `games.csv`
- `sample_submission.csv`
- `test.csv`
- `train.csv`
- `turns.csv`

## Dataset Consistency

All team members should verify they are using the same dataset archive.

SHA256 for `scrabble-player-rating.zip`:

`1e2a033b846b723f44a7ea27ea2ca6877efc0008eb037d86614a267dc4284228`

You can verify it locally with:

```bash
sha256sum data/raw/scrabble-player-rating.zip
```

## Collaboration Rules

- Do not commit raw dataset files to GitHub.
- Do not modify files inside `data/raw/`.
- Put cleaned or transformed outputs in `data/processed/`.
- Keep all preprocessing and feature generation reproducible through scripts or notebooks.

## Package Requirements

This section will be finalized once the modeling stack is fixed. At minimum, the setup currently assumes:

- `python3`

## Run Instructions

1. Clone the repository.
2. Download `scrabble-player-rating.zip` from the shared Drive folder.
3. Place it in `data/raw/`.
4. Run `python3 scripts/setup_data.py`.
5. Confirm the extracted CSV files appear in `data/raw/scrabble-player-rating/`.

## Notes

The reference notebook currently in this repository is:

`full-walkthrough-eda-fe-model-tuning.ipynb`

It will later be reorganized into the `notebooks/` directory when we standardize the project workflow.
