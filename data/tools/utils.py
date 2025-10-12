import os
import sys

from ..datasets import VisualGenomeTrainData
from ..datasets.synthetic_genome import SyntheticGenomeTrainData
from ..datasets.multi_dataset import MultiDatasetTrainData, register_multi_datasets
from detectron2.data.datasets import register_coco_instances

def register_datasets(cfg):
    if cfg.DATASETS.TYPE == 'VISUAL GENOME':
        for split in ['train', 'val', 'test']:
            dataset_instance = VisualGenomeTrainData(cfg, split=split)
    elif cfg.DATASETS.TYPE == 'SYNTHETIC GENOME':
        for split in ['train', 'val', 'test']:
            dataset_instance = SyntheticGenomeTrainData(cfg, split=split)
    elif cfg.DATASETS.TYPE == 'MULTI_DATASET':
        register_multi_datasets(cfg)
    else:
        raise ValueError(f"Unsupported dataset type: {cfg.DATASETS.TYPE}")
        