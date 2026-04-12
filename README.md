# Scrabble Player Rating

This project predicts the rating of human Scrabble players using game metadata and turn-level gameplay logs.

## Project Structure

```text
Scrabble_Player_Rating/
├── data/
│   └── raw/
├── notebooks/
│   ├── baseline_model.ipynb
│   └── final_model.ipynb
├── reports/
│   ├── baseline_results.md
│   ├── final_comparison.md
│   ├── model_strategy.md
│   ├── turn_feature_notes.md
│   └── validation_notes.md
├── scripts/
│   └── setup_data.py
├── src/
│   ├── baseline_model.py
│   ├── model_candidates.py
│   ├── turn_features.py
│   └── validation.py
└── README.md
```

## Package Requirements

Recommended Python version:

- `Python 3.12.3`

Required packages used in the project:

- `numpy==2.4.4`
- `pandas==3.0.2`
- `scikit-learn`
- `matplotlib`
- `xgboost==3.2.0`
- `lightgbm==4.6.0`
- `nbformat==5.10.4`
- `nbclient==0.10.4`
- `ipykernel==7.2.0`

If exact versions are not available for every package on another system, use compatible recent versions of the same libraries.

## Dataset Setup

The raw dataset is not stored in the repository.

Download the dataset archive from the shared Drive folder before running any code:

- <https://drive.google.com/drive/folders/1SxC27uGBSb4679CgFDrC5ymUvmSfBN7a?usp=drive_link>

Expected archive:

- `scrabble-player-rating.zip`

Place the archive here:

- `data/raw/scrabble-player-rating.zip`

Then extract the dataset by running:

```bash
python3 scripts/setup_data.py
```

After extraction, the dataset should be available at:

- `data/raw/scrabble-player-rating/`

Expected extracted files:

- `games.csv`
- `sample_submission.csv`
- `test.csv`
- `train.csv`
- `turns.csv`

## Run Instructions (in the project base folder)

### 1. Clone the repository

```bash
git clone https://github.com/bitreaper42/Scrabble_Player_Rating.git
cd Scrabble_Player_Rating
```

### 2. Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install required packages

```bash
pip install numpy pandas scikit-learn matplotlib xgboost lightgbm nbformat nbclient ipykernel
```

### 4. Place the dataset archive in `data/raw/`

Download the file:

- `scrabble-player-rating.zip`

from the shared Drive folder and place it at:

- `data/raw/scrabble-player-rating.zip`

### 5. Extract the dataset

```bash
python3 scripts/setup_data.py
```

### 6. Run the baseline pipeline

```bash
python3 src/baseline_model.py
```

This evaluates the metadata-only baseline model.

### 7. Run the final model pipeline

```bash
python3 src/model_candidates.py
```

This evaluates the final candidate models using metadata and turn-level features, then selects the best-performing final model.

### 8. Open the notebooks if needed

Available notebooks:

- `notebooks/baseline_model.ipynb`
- `notebooks/final_model.ipynb`

These notebooks summarize the baseline and final-model workflows with visible outputs.

## Final Model Summary

Selected final model:

- `blend_ridge_hist`

This model combines:

- a tuned ridge regression pipeline with one-hot encoded categorical features
- a tuned histogram gradient boosting regressor

The final prediction is a weighted blend of the two component model outputs.

## Notes

- `src/validation.py` contains the grouped-human validation logic.
- `src/turn_features.py` creates the turn-level aggregate features used by the final model.
- `src/model_candidates.py` trains and compares the final candidate models.
