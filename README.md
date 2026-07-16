# Explainable Handwritten Character Recognition

This project recognizes handwritten characters from the UJI Pen Characters dataset and explains the model decision at stroke level. The final classifier is a point-sequence Transformer trained on resampled raw pen trajectories, not only on stroke-summary feature matrices.

## What The Project Does

- Classifies handwritten characters into 62 classes: digits, uppercase letters, and lowercase letters.
- Converts each handwritten character into a 96-point sequence.
- Uses point features `(x, y, dx, dy, pen_flag)` for each resampled point.
- Trains a Transformer over the full pen trajectory so the model keeps local curve and shape information.
- Explains predictions using Shapley-style stroke contribution analysis.
- Validates explanations through stroke-removal experiments.
- Provides an interactive Streamlit dashboard for prediction, visualization, and interpretation.

## Current Final Model

The final model is trained by:

```bash
python train_point_transformer.py
```

It saves the best checkpoint as:

```text
best_point_model.pt
```

Model setup:

- Input shape: `(96, 5)` per sample
- Point features: `x`, `y`, `dx`, `dy`, `pen_flag`
- Resampled points per character: 96
- Embedding size: 128
- Transformer encoder layers: 4
- Attention heads: 8
- Feed-forward dimension: 512
- Pooling/summary token: learned classification token
- Optimizer: AdamW
- Regularization: dropout, label smoothing, gradient clipping, early stopping
- Training augmentation: light rotation, scaling, shifting, and noise

This point-sequence model replaced the stroke-summary Transformer because the summary representation compressed each stroke into only 15 numbers. The point model keeps much more of the handwritten shape, including local movement and curve structure.

## Project Structure

```text
final/
|-- app.py                               # Final Streamlit dashboard
|-- best_point_model.pt                   # Final trained point-transformer weights
|-- train_point_transformer.py            # Final training script
|-- requirements.txt                      # Python dependencies
|-- README.md                             # Project documentation
|-- data/                                 # Raw UJI Pen Characters files
|-- processed/                            # Processed stroke-summary arrays for display/analysis
|-- files/                                # CSV and pickle outputs used by dashboard
`-- scripts/                              # Model, preprocessing, Shapley, and analysis scripts
```

Important files:

- `scripts/point_sequence_dataset.py`: converts raw strokes into 96-point model sequences
- `scripts/point_sequence_transformer.py`: final Transformer architecture
- `scripts/predictor.py`: loads `best_point_model.pt` for the dashboard
- `files/parsed_samples.pkl`: parsed raw samples used by the dashboard and point predictor
- `files/shapley_results_clean.csv`: per-sample stroke contributions
- `files/character_profiles.csv`: character-level explanation profiles
- `files/character_taxonomy.csv`: final character behavior categories

## Requirements

Use Python 3.10 or newer.

Install dependencies:

```bash
pip install -r requirements.txt
```

Main dependencies:

- numpy
- pandas
- scikit-learn
- torch
- tqdm
- matplotlib
- streamlit

## Run The Dashboard

From the `final` folder:

```bash
python -m streamlit run app.py
```

If dependencies are missing, run this first:

```bash
python -m pip install -r requirements.txt
```

## Train The Final Model

From the `final` folder:

```bash
python train_point_transformer.py
```

This trains the point-sequence Transformer and saves:

```text
best_point_model.pt
```

## Shapley And Analysis Pipeline

Because the final model is point-sequence based, the Shapley scripts now rebuild the same raw-sample test split used by `train_point_transformer.py`. Stroke coalitions are evaluated by keeping/removing raw strokes, then resampling the remaining handwriting to the 96-point model input.

Run these from inside the `scripts` folder:

```bash
cd scripts
python shapley_dataset.py
python analyse_shapley.py
python character_cooperation.py
python character_synergy.py
python character_profiles.py
python character_removal_analysis.py
python stroke_removal_validation.py
python character_taxonomy.py
```

Recommended order and outputs:

1. `shapley_dataset.py` creates `shapley_results_clean.csv`.
2. `analyse_shapley.py` creates `character_shapley_summary.csv`.
3. `character_cooperation.py` creates `character_cooperation.csv`.
4. `character_synergy.py` creates `character_synergy.csv` and `sample_synergy.csv`.
5. `character_profiles.py` creates `character_profiles.csv`.
6. `character_removal_analysis.py` creates `character_removal_summary.csv`.
7. `stroke_removal_validation.py` creates `stroke_removal_results.csv`.
8. `character_taxonomy.py` creates `character_taxonomy.csv`.

After running the pipeline, keep the dashboard-facing CSV files in the `files/` folder because `app.py` reads dashboard artifacts from there.

## Dashboard Pages

The Streamlit app includes:

- Project Overview
- Character Explorer
- Feature View
- Point Transformer Prediction
- Stroke Importance
- Stroke Removal Lab
- Research Interpretation
- Future Scope

The Feature View can still display the older 3 x 15 stroke-summary features for interpretability, but prediction is performed by the point-sequence Transformer using raw parsed strokes.

## Notes

- `point_sequence_transformer.py` and `point_sequence_dataset.py` are helper modules and should not be run directly.
- `app.py` is the final dashboard file.
- The app uses `best_point_model.pt`.
- The old masked small-FF model files are kept only as earlier experiment/baseline artifacts.
- The `files/` folder stores explanation CSV files and parsed samples used by the dashboard.

