# TransAttUNet++ 3D with 5-Fold Cross-Validation & Zip Extraction

## Quick Start (Colab)

### Prerequisites
1. Upload `PANTHER_NIfTI.zip` to your Google Drive's `MyDrive` folder
2. Script will automatically extract it during initialization

### In Colab Cell 1
```python
# Upload and run the script
!wget -O transattunet++.py 'https://your-github-raw-url/transattunet++.py'
!python transattunet++.py
```

## What's New

### 1. Automatic Zip Extraction
- **Colab**: Automatically extracts `PANTHER_NIfTI.zip` from Google Drive
- **Local**: Uses local `PANTHER_NIfTI` directory path
- **Smart**: Only extracts if not already present (saves time on reruns)

```python
# Automatically called in main():
extract_zip_if_needed(CONFIG)
```

### 2. 5-Fold Cross-Validation
Instead of single train/val split, the model trains 5 separate models:
- Each fold uses ~80% data for training, ~20% for validation
- All data is used for training across all folds
- Better generalization estimates and model robustness

#### Data Flow
```
All Patients (n=100, example)
├── Fold 1: Train[0-80] → Val[80-100]
├── Fold 2: Train[0-20,80-100] → Val[20-40]
├── Fold 3: Train[0-40,60-100] → Val[40-60]
├── Fold 4: Train[0-60,80-100] → Val[60-80]
└── Fold 5: Train[20-100] → Val[0-20]
```

### 3. Output Files Per Fold
For each fold, you get:
```
output_dir/
├── fold_1_best_model.pth          # Best trained model for fold 1
├── fold_1_epoch_5.pth              # Checkpoint at epoch 5
├── fold_1_history.csv              # Training history (loss, metrics)
├── fold_2_best_model.pth
├── fold_2_history.csv
├── ... (up to fold 5)
├── cv_summary.json                 # Summary of all folds
└── cv_results.png                  # Visualization
```

## Configuration

Edit these in the script before running:

```python
CONFIG = {
    'epochs': 50,                   # Per fold
    'batch_size': 4,                # For 3D (reduce if CUDA OOM)
    'learning_rate': 1e-4,
    'n_splits': 5,                  # Number of CV folds
    'save_frequency': 5,            # Save checkpoint every N epochs
}
```

## Colab Memory Tips

If you get CUDA out-of-memory errors:

1. **Reduce batch size**
   ```python
   CONFIG['batch_size'] = 2  # from 4
   ```

2. **Reduce input size**
   ```python
   CONFIG['input_size_3d'] = (16, 64, 64)  # from (32, 128, 128)
   ```

3. **Reduce epochs per fold**
   ```python
   CONFIG['epochs'] = 20  # from 50
   ```

## Monitoring Training

The script prints real-time metrics:
```
============================================================
Preparing Fold 1/5
============================================================
  Train samples: 80
  Val samples: 20

Training ████████████░░░░░░░░░░░░ [50%] Loss: 0.4231

============================================================
FOLD 1/5
============================================================
  Epoch 1/50
    Train Loss: 0.4503 | Train Dice: 0.5421
    Val Loss: 0.3892 | Val Dice: 0.6234
    Val IoU: 0.4521 | Val F1: 0.7654
    ✅ Best model saved
```

## Results Analysis

After training completes:

1. **Check fold performance**
   ```bash
   cat cv_summary.json
   ```
   Shows best epoch and loss for each fold

2. **Review fold-specific metrics**
   ```bash
   cat fold_1_history.csv  # Training history for fold 1
   ```

3. **View cross-validation plot**
   - Open `cv_results.png` to see:
     - Best epoch per fold
     - Best loss per fold

## Metrics Tracked

Per epoch, the script computes:
- **Dice Score**: Overlap between prediction and label (0-1)
- **IoU**: Intersection over Union (0-1)
- **Sensitivity**: True positive rate (0-1)
- **Specificity**: True negative rate (0-1)
- **Precision**: Positive predictive value (0-1)
- **Recall**: Same as sensitivity (0-1)
- **F1-Score**: Harmonic mean of precision/recall (0-1)

## Local Execution

For local training with existing `PANTHER_NIfTI` directory:

```bash
python transattunet++.py
```

Script automatically detects local path:
```python
CONFIG['data_dir'] = r'D:\GitHub\...\Dataset\PANTHER_NIfTI'
```

## Troubleshooting

### "PANTHER_NIfTI.zip not found in Google Drive"
- Solution: Upload `PANTHER_NIfTI.zip` to `MyDrive` in Google Drive

### "No Task1 or Task2 directories found"
- Check zip structure has `PANTHER_NIfTI/Task1/imagesTr/` and `labelsTr/`

### CUDA out of memory
- Reduce batch size, input size, or epochs
- See "Colab Memory Tips" section above

### Training is very slow
- Colab CPU training is slow; GPU should be available
- Check: Runtime → Change runtime type → GPU

## Expected Training Time (Colab GPU)

- Per fold: ~2-4 minutes (50 epochs)
- All 5 folds: ~15-20 minutes total
- With data extraction: +2-3 minutes

## Next Steps After Training

1. **Use best models for inference**
   ```python
   model.load_state_dict(torch.load('fold_1_best_model.pth'))
   predictions = model(test_image)
   ```

2. **Ensemble all 5 models** for better predictions
   ```python
   # Load all 5 fold models and average predictions
   ```

3. **Analyze per-fold performance** from CSV history files

4. **Adjust hyperparameters** if performance is suboptimal
   - Already tracked in `cv_summary.json`
   - Use results to inform next training run
