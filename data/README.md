# Datasets

This directory contains dataset placeholders and instructions.

## Dataset Requirements

### 1. Food41 Dataset
Required for training base ResNet models.

**Download from Kaggle:**
- https://www.kaggle.com/datasets/fooDB/food41

**Setup:**
```bash
# Extract to this directory
unzip food41.zip
mv food41/images ./
```

**Structure expected:**
```
food41/images/
├── class_001/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── class_002/
│   └── ...
└── ...
```

**Statistics:**
- Classes: 101 food categories
- Total Images: ~16,000
- Resolution: Variable
- Used for: Model training and pruning

### 2. CIFAR-10 Dataset
Required for fine-tuning experiments.

**Download from Kaggle:**
- https://www.kaggle.com/datasets/swaroopkumarpati/cifar10-python-files

**Setup:**
```bash
# Extract to this directory
unzip cifar10.zip
mv cifar10/train ./train
mv cifar10/test ./test
```

**Structure expected:**
```
cifar10/
├── train/
│   ├── airplane/
│   ├── automobile/
│   ├── bird/
│   ├── cat/
│   ├── deer/
│   ├── dog/
│   ├── frog/
│   ├── horse/
│   ├── ship/
│   └── truck/
└── test/
    ├── airplane/
    ├── automobile/
    └── ...
```

**Statistics:**
- Classes: 10 object categories
- Training Images: 50,000
- Test Images: 10,000
- Resolution: 32x32 (upsampled to 224x224)
- Used for: Fine-tuning pruned models

## Data Loading

Data is automatically loaded through utility functions:

```python
from src.data.dataset import get_food41_dataloaders, get_cifar10_dataloaders

# Food41
train_loader, val_loader, test_loader, num_classes = get_food41_dataloaders(
    data_dir='data/food41/images',
    batch_size=64
)

# CIFAR-10
train_loader, val_loader, test_loader = get_cifar10_dataloaders(
    data_dir='data/cifar10',
    batch_size=128
)
```

## Data Split

- **Food41**: 70% training, 15% validation, 15% test
- **CIFAR-10**: 90% training (from train set), 10% validation, 100% test

All splits use deterministic seed (42) for reproducibility.

## Carbon Intensity Data (Optional)

For grid carbon intensity traces:
- **CAISO Data**: California ISO grid
- **PJM Data**: PJM Interconnection grid

Download from WattTime or use the utility function in `src/evaluation/plotting.py`

## Notes

- Datasets are large (~10-50 GB) and not included in the repository
- Always keep `.gitignore` updated to prevent accidental commits
- Use symlinks if storing datasets elsewhere for space efficiency
