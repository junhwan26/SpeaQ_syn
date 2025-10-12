# SpeaQ: Multi-Dataset Scene Graph Generation

SpeaQ is a transformer-based scene graph generation model that supports both real and synthetic datasets with dynamic sampling and individual sample-level loss weighting.

## Features

- **Multi-Dataset Training**: Combines real (Visual Genome) and synthetic datasets with configurable ratios
- **Dynamic Sampling**: Real-time batch sampling maintaining specified dataset ratios
- **Individual Loss Weighting**: Sample-level loss weights for different dataset types
- **Robust NaN Prevention**: Enhanced safety checks preventing training instability
- **Transformer Architecture**: Based on DETR with iterative relation detection

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/SpeaQ.git
cd SpeaQ
```

2. Create conda environment:
```bash
conda create -n SpeaQ python=3.9
conda activate SpeaQ
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Single Dataset Training

#### Visual Genome Only:
```bash
python train_iterative_model.py \
    --config-file configs/visual_genome.yaml \
    --num-gpus 4
```

#### Synthetic Dataset Only:
```bash
python train_iterative_model.py \
    --config-file configs/synthetic_genome.yaml \
    --num-gpus 4
```

### Multi-Dataset Training

#### Real + Synthetic with Custom Ratios:
```bash
python train_iterative_model.py \
    --config-file configs/speaq_multi_dataset.yaml \
    --num-gpus 4 \
    OUTPUT_DIR outputs/multi_dataset_training \
    DATASETS.MULTI_DATASET.REAL_SAMPLING_RATIO 0.7 \
    DATASETS.MULTI_DATASET.SYNTHETIC_SAMPLING_RATIO 0.3 \
    DATASETS.MULTI_DATASET.REAL_LOSS_WEIGHT 1.0 \
    DATASETS.MULTI_DATASET.SYNTHETIC_LOSS_WEIGHT 0.5
```

## Configuration

### Multi-Dataset Settings

Key parameters in `configs/speaq_multi_dataset.yaml`:

```yaml
DATASETS:
  TYPE: "MULTI_DATASET"
  MULTI_DATASET:
    ENABLED: True
    REAL_SAMPLING_RATIO: 0.7      # 70% real data
    SYNTHETIC_SAMPLING_RATIO: 0.3 # 30% synthetic data
    REAL_LOSS_WEIGHT: 1.0         # Loss weight for real data
    SYNTHETIC_LOSS_WEIGHT: 0.5    # Loss weight for synthetic data
```

### Model Settings

```yaml
MODEL:
  DETR:
    ONE2MANY_SCHEME: "dynamic"
    MULTIPLY_QUERY: 2
    MATCH_INDEPENDENT: True
    USE_GROUP_MASK: True
    NUM_GROUPS: 4
    OVERSAMPLE_PARAM: 0.07
    UNDERSAMPLE_PARAM: 1.5
```

## Dataset Structure

### Visual Genome Format
```
dataset/
├── VG_100K/                    # Images
├── VG-SGG-dicts-with-attri.json # Class mappings
├── image_data.json             # Image metadata
└── VG-SGG-with-attri.h5        # Annotations
```

### Synthetic Dataset Format
Same structure as Visual Genome but with synthetic images.

## Key Features

### Dynamic Sampling
The `MultiDatasetDynamicSampler` ensures precise ratio control within each batch:

```python
# Example: batch_size=20, real_ratio=0.7
batch = [
    real_data[0-13],      # 14 real samples (70%)
    synthetic_data[14-19] # 6 synthetic samples (30%)
]
```

### Individual Loss Weighting
Each sample gets its own loss weight based on dataset type:

```python
# Real data samples: loss_weight = 1.0
# Synthetic data samples: loss_weight = 0.5
loss = individual_sample_loss * loss_weight
```

### Robust Training
Enhanced safety checks prevent NaN losses:
- Statistics validation
- Weight clamping
- Exception handling with fallbacks

## Evaluation

```bash
python train_iterative_model.py \
    --config-file configs/speaq_multi_dataset.yaml \
    --eval-only \
    MODEL.WEIGHTS path/to/checkpoint.pth
```

## Results

The model achieves improved performance by combining real and synthetic datasets with carefully tuned loss weighting and dynamic sampling strategies.

## Citation

If you use this code, please cite:

```bibtex
@article{speaq2024,
  title={SpeaQ: Multi-Dataset Scene Graph Generation with Dynamic Sampling},
  author={Your Name},
  journal={Your Journal},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built on Detectron2 framework
- Uses Visual Genome dataset
- Inspired by DETR architecture