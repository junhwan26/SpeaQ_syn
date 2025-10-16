# SpeaQ with Multi-Dataset Scene Graph Generation

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
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_iterative_model.py \
    --resume --num-gpus 4 \
    --config-file configs/visual_genome.yaml \
    --dist-url 29503 \
    OUTPUT_DIR outputs/visual_genome_training \
    SOLVER.IMS_PER_BATCH 20 \
    MODEL.WEIGHTS /home/junhwanheo/SpeaQ/vg_objectdetector_pretrained.pth
```

#### Synthetic Dataset Only:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_iterative_model.py \
    --resume --num-gpus 4 \
    --config-file configs/synthetic_genome.yaml \
    --dist-url 29503 \
    OUTPUT_DIR outputs/synthetic_training \
    SOLVER.IMS_PER_BATCH 20 \
    MODEL.WEIGHTS /home/junhwanheo/SpeaQ/vg_objectdetector_pretrained.pth
```

### Multi-Dataset Training

#### Real + Synthetic with Custom Ratios:
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_iterative_model.py \
    --resume --num-gpus 4 \
    --config-file configs/speaq_multi_dataset.yaml \
    --dist-url 29503 \
    OUTPUT_DIR outputs/multi_dataset_training \
    SOLVER.IMS_PER_BATCH 20 \
    MODEL.WEIGHTS /home/junhwanheo/SpeaQ/vg_objectdetector_pretrained.pth \
    MODEL.DETR.ONE2MANY_SCHEME dynamic \
    MODEL.DETR.MULTIPLY_QUERY 2 \
    MODEL.DETR.ONLY_PREDICATE_MULTIPLY True \
    MODEL.DETR.ONE2MANY_K 4 \
    MODEL.DETR.ONE2MANY_DYNAMIC_SCHEME max \
    MODEL.DETR.USE_GROUP_MASK True \
    MODEL.DETR.MATCH_INDEPENDENT True \
    MODEL.DETR.NUM_GROUPS 4 \
    MODEL.DETR.ONE2MANY_PREDICATE_SCORE True \
    MODEL.DETR.ONE2MANY_PREDICATE_WEIGHT -0.5 \
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

### Dataset Preparation

The synthetic dataset uses the same format as Visual Genome but contains synthetic images instead of real images.

#### Required Files

The following files need to be prepared:

1. **Image files**: `.jpg` files in the image directory
2. **Annotation file**: `augmented_from_refined.h5` - HDF5 format same as Visual Genome
3. **Metadata file**: `image_data_new.json` - Image metadata
4. **Mapping dictionary**: `VG-SGG-dicts-with-attri.json` - Class and relation mappings

#### Directory Structure

```
datasets/
├── generated/                  # Synthetic dataset files
│   ├── augmented_from_refined.h5    # Annotation data
│   └── image_data_new.json          # Image metadata
└── vg/                        # Real dataset files
    ├── VG_100K/               # Real images
    ├── VG-SGG-dicts-with-attri.json # Shared mapping dictionary
    ├── image_data.json        # Real image metadata
    └── VG-SGG-with-attri.h5   # Real annotation data
```

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
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_iterative_model.py \
    --resume --eval-only --num-gpus 4 \
    --config-file configs/speaq_multi_dataset.yaml \
    --dist-url 29503 \
    OUTPUT_DIR outputs/multi_dataset_training \
    SOLVER.IMS_PER_BATCH 20
```

## Dataset Analysis and Visualization

### Predicate Distribution Analysis

Analyze and visualize the distribution of predicates (relations) in both real and synthetic datasets:

#### Using Config File (Recommended)
```bash
# Multi-dataset analysis
python visualize_predicate_distribution.py \
    --config-file configs/speaq_multi_dataset.yaml \
    --output-dir predicate_analysis

# Synthetic-only analysis
python visualize_predicate_distribution.py \
    --config-file configs/synthetic_genome.yaml \
    --output-dir synthetic_predicate_analysis
```

#### Using Manual Paths
```bash
python visualize_predicate_distribution.py \
    --real-mapping /home/dataset/vg/VG-SGG-dicts-with-attri.json \
    --real-h5 /home/dataset/vg/VG-SGG-with-attri.h5 \
    --synthetic-mapping /home/dataset/vg/VG-SGG-dicts-with-attri.json \
    --synthetic-h5 /home/junhwanheo/DA-SGG/datasets/generated/augmented_from_refined.h5 \
    --output-dir predicate_analysis
```

This will generate the following visualizations:

- **`top_predicates_comparison.png`**: Top 20 predicates in each dataset
- **`predicate_distribution_comparison.png`**: Side-by-side comparison of all predicates
- **`cumulative_predicate_distribution.png`**: Cumulative distribution curves
- **`predicate_statistics_summary.png`**: Comprehensive statistics overview

### Training Progress Visualization

Visualize training progress and loss curves:

```bash
python vis_train_log.py --log-file outputs/multi_dataset_training/log.txt
```

This generates training plots including:
- Loss components over time
- Validation metrics
- Learning rate schedules
- Model performance trends

## Parameter Descriptions

### Dataset Parameters

- `DATASETS.TYPE`: Set to "SYNTHETIC GENOME" for synthetic-only or "MULTI_DATASET" for combined training
- `DATASETS.SYNTHETIC_GENOME.IMAGES`: Path to synthetic image directory
- `DATASETS.SYNTHETIC_GENOME.VG_ATTRIBUTE_H5`: Path to annotation HDF5 file
- `DATASETS.SYNTHETIC_GENOME.IMAGE_DATA`: Path to image metadata JSON file
- `DATASETS.SYNTHETIC_GENOME.MAPPING_DICTIONARY`: Path to class mapping JSON file

### SpeaQ Model Parameters

- `MODEL.DETR.ONE2MANY_SCHEME`: 'dynamic' - Dynamic query generation scheme
- `MODEL.DETR.MULTIPLY_QUERY`: 2 - Query amplification factor
- `MODEL.DETR.ONLY_PREDICATE_MULTIPLY`: True - Amplify only relations
- `MODEL.DETR.ONE2MANY_K`: 4 - Number of queries per relation
- `MODEL.DETR.USE_GROUP_MASK`: True - Use group masking
- `MODEL.DETR.MATCH_INDEPENDENT`: True - Independent matching
- `MODEL.DETR.NUM_GROUPS`: 4 - Number of groups

## Important Notes

1. **Data Format**: Synthetic dataset must use the same HDF5 format as Visual Genome
2. **File Paths**: Ensure correct data file paths in configuration files
3. **GPU Memory**: GPU memory usage may vary depending on synthetic image resolution
4. **Data Preprocessing**: Verify that data is correctly preprocessed

## Troubleshooting

### Common Issues

1. **File Not Found**: Check if data file paths are correct
2. **Out of Memory**: Reduce `SOLVER.IMS_PER_BATCH` value
3. **Data Loading Error**: Verify HDF5 file format is correct

### Debugging Tips

- Set `DATASETS.SYNTHETIC_GENOME.FILTER_EMPTY_RELATIONS: False` to include empty relations
- Test with smaller batch size first
- Check logs for data loading status

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