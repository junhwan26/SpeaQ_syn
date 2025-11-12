#!/usr/bin/env python3
"""
데이터셋 로딩 디버깅 및 분석 스크립트

사용법:
    python debug_dataset_loading.py --config-file configs/speaq_multi_dataset.yaml
"""

import argparse
import sys
import os
import json
import h5py
import pickle
import numpy as np
from pathlib import Path

# Detectron2 및 프로젝트 임포트
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.tools import register_datasets
from configs.defaults import add_dataset_config, add_scenegraph_config


def analyze_h5_structure(h5_path):
    """H5 파일 구조 분석"""
    print("\n" + "="*80)
    print(f"H5 파일 구조 분석: {h5_path}")
    print("="*80)
    
    if not os.path.exists(h5_path):
        print(f"⚠️  파일이 존재하지 않습니다: {h5_path}")
        return
    
    with h5py.File(h5_path, 'r') as f:
        print("\n📁 키 목록:")
        for key in f.keys():
            data = f[key]
            print(f"  - {key}: shape={data.shape}, dtype={data.dtype}")
        
        # Split 분석
        if 'split' in f:
            split = f['split'][:]
            print(f"\n📊 Split 분포:")
            print(f"  - Train (0): {np.sum(split == 0)}")
            print(f"  - Val (1): {np.sum(split == 1)}")
            print(f"  - Test (2): {np.sum(split == 2)}")
        
        # 박스 분석
        if 'img_to_first_box' in f:
            first_box = f['img_to_first_box'][:]
            valid_images = np.sum(first_box >= 0)
            print(f"\n📦 박스 정보:")
            print(f"  - 박스가 있는 이미지: {valid_images}")
            print(f"  - 박스가 없는 이미지: {len(first_box) - valid_images}")
        
        # 관계 분석
        if 'img_to_first_rel' in f:
            first_rel = f['img_to_first_rel'][:]
            valid_relations = np.sum(first_rel >= 0)
            print(f"\n🔗 관계 정보:")
            print(f"  - 관계가 있는 이미지: {valid_relations}")
            print(f"  - 관계가 없는 이미지: {len(first_rel) - valid_relations}")


def analyze_mapping_dictionary(mapping_path):
    """매핑 딕셔너리 분석"""
    print("\n" + "="*80)
    print(f"매핑 딕셔너리 분석: {mapping_path}")
    print("="*80)
    
    if not os.path.exists(mapping_path):
        print(f"⚠️  파일이 존재하지 않습니다: {mapping_path}")
        return
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    print("\n📚 클래스 정보:")
    if 'label_to_idx' in mapping:
        print(f"  - 객체 클래스 수: {len(mapping['label_to_idx'])}")
        print(f"  - 첫 10개 클래스:")
        for i, (label, idx) in enumerate(sorted(mapping['label_to_idx'].items(), key=lambda x: x[1])[:10]):
            print(f"    {idx}: {label}")
    
    if 'predicate_to_idx' in mapping:
        print(f"\n🔗 관계 클래스 수: {len(mapping['predicate_to_idx'])}")
        print(f"  - 첫 10개 관계:")
        for i, (pred, idx) in enumerate(sorted(mapping['predicate_to_idx'].items(), key=lambda x: x[1])[:10]):
            print(f"    {idx}: {pred}")
    
    if 'attribute_to_idx' in mapping:
        print(f"\n🏷️  속성 클래스 수: {len(mapping['attribute_to_idx'])}")


def analyze_dataset_dicts(dataset_name, sample_count=5):
    """데이터셋 dict 분석"""
    print("\n" + "="*80)
    print(f"데이터셋 분석: {dataset_name}")
    print("="*80)
    
    if dataset_name not in DatasetCatalog.list():
        print(f"⚠️  데이터셋이 등록되지 않았습니다: {dataset_name}")
        return
    
    dataset_dicts = DatasetCatalog.get(dataset_name)
    print(f"\n📊 기본 정보:")
    print(f"  - 총 샘플 수: {len(dataset_dicts)}")
    
    if len(dataset_dicts) == 0:
        print("  ⚠️  데이터셋이 비어있습니다!")
        return
    
    # 첫 번째 샘플 분석
    sample = dataset_dicts[0]
    print(f"\n📝 샘플 구조:")
    print(f"  - 키: {list(sample.keys())}")
    print(f"  - 이미지 ID: {sample.get('image_id', 'N/A')}")
    print(f"  - 파일명: {sample.get('file_name', 'N/A')}")
    print(f"  - 이미지 크기: {sample.get('width', 'N/A')} x {sample.get('height', 'N/A')}")
    print(f"  - 객체 수: {len(sample.get('annotations', []))}")
    print(f"  - 관계 수: {len(sample.get('relations', []))}")
    
    # 객체 통계
    if 'annotations' in sample and len(sample['annotations']) > 0:
        obj = sample['annotations'][0]
        print(f"\n📦 객체 구조:")
        print(f"  - 키: {list(obj.keys())}")
        print(f"  - bbox: {obj.get('bbox', 'N/A')}")
        print(f"  - category_id: {obj.get('category_id', 'N/A')}")
        print(f"  - attribute shape: {np.array(obj.get('attribute', [])).shape}")
    
    # 관계 통계
    if 'relations' in sample and len(sample['relations']) > 0:
        relations = np.array(sample['relations'])
        print(f"\n🔗 관계 구조:")
        print(f"  - shape: {relations.shape}")
        print(f"  - 첫 5개 관계:")
        for i, rel in enumerate(relations[:5]):
            print(f"    {i}: subject={rel[0]}, object={rel[1]}, predicate={rel[2]}")
    
    # 전체 통계
    num_objs = [len(d['annotations']) for d in dataset_dicts[:1000]]  # 처음 1000개만
    num_rels = [len(d['relations']) for d in dataset_dicts[:1000]]
    
    print(f"\n📈 통계 (처음 1000개 샘플):")
    print(f"  - 평균 객체 수: {np.mean(num_objs):.2f}")
    print(f"  - 최대 객체 수: {np.max(num_objs)}")
    print(f"  - 평균 관계 수: {np.mean(num_rels):.2f}")
    print(f"  - 최대 관계 수: {np.max(num_rels)}")


def analyze_cache_files():
    """캐시 파일 분석"""
    print("\n" + "="*80)
    print("캐시 파일 분석")
    print("="*80)
    
    cache_dir = Path("tmp")
    if not cache_dir.exists():
        print("  ⚠️  캐시 디렉토리가 없습니다.")
        return
    
    cache_files = list(cache_dir.glob("*.pkl"))
    print(f"\n📦 캐시 파일 수: {len(cache_files)}")
    
    for cache_file in cache_files[:10]:  # 처음 10개만
        file_size = cache_file.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {cache_file.name}: {file_size:.2f} MB")
        
        # 캐시 파일에서 샘플 정보 확인
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                if isinstance(data, list) and len(data) > 0:
                    print(f"    → 샘플 수: {len(data)}")
                    if isinstance(data[0], dict):
                        print(f"    → 키: {list(data[0].keys())}")
        except Exception as e:
            print(f"    ⚠️  로드 실패: {e}")


def analyze_metadata(dataset_name):
    """메타데이터 분석"""
    print("\n" + "="*80)
    print(f"메타데이터 분석: {dataset_name}")
    print("="*80)
    
    if dataset_name not in MetadataCatalog.list():
        print(f"⚠️  메타데이터가 없습니다: {dataset_name}")
        return
    
    metadata = MetadataCatalog.get(dataset_name)
    print(f"\n📚 클래스 정보:")
    
    if hasattr(metadata, 'thing_classes'):
        print(f"  - 객체 클래스 수: {len(metadata.thing_classes)}")
        print(f"  - 첫 10개: {metadata.thing_classes[:10]}")
    
    if hasattr(metadata, 'predicate_classes'):
        print(f"  - 관계 클래스 수: {len(metadata.predicate_classes)}")
        print(f"  - 첫 10개: {metadata.predicate_classes[:10]}")
    
    if hasattr(metadata, 'attribute_classes'):
        print(f"  - 속성 클래스 수: {len(metadata.attribute_classes)}")
    
    if hasattr(metadata, 'statistics'):
        stats = metadata.statistics
        print(f"\n📊 통계 정보:")
        print(f"  - fg_rel_count shape: {stats['fg_rel_count'].shape}")
        print(f"  - fg_rel_count sum: {stats['fg_rel_count'].sum():.0f}")
        print(f"  - fg_matrix shape: {stats['fg_matrix'].shape}")
        print(f"  - 가장 빈번한 관계 (상위 10개):")
        top_rels = torch.topk(stats['fg_rel_count'], 10)
        for i, (value, idx) in enumerate(zip(top_rels.values, top_rels.indices)):
            if idx < len(metadata.predicate_classes):
                print(f"    {i+1}. {metadata.predicate_classes[idx]}: {value:.0f}")


def analyze_multi_dataset(cfg):
    """멀티 데이터셋 분석"""
    print("\n" + "="*80)
    print("멀티 데이터셋 설정 분석")
    print("="*80)
    
    if cfg.DATASETS.TYPE != "MULTI_DATASET":
        print("  ⚠️  멀티 데이터셋이 아닙니다.")
        return
    
    multi_cfg = cfg.DATASETS.MULTI_DATASET
    print(f"\n⚙️  설정:")
    print(f"  - Enabled: {multi_cfg.ENABLED}")
    print(f"  - Real sampling ratio: {multi_cfg.REAL_SAMPLING_RATIO}")
    print(f"  - Synthetic sampling ratio: {multi_cfg.SYNTHETIC_SAMPLING_RATIO}")
    print(f"  - Real loss weight: {multi_cfg.REAL_LOSS_WEIGHT}")
    print(f"  - Synthetic loss weight: {multi_cfg.SYNTHETIC_LOSS_WEIGHT}")
    
    # Real 데이터셋 분석
    if 'VG_train' in DatasetCatalog.list():
        real_dicts = DatasetCatalog.get('VG_train')
        print(f"\n📊 Real 데이터셋:")
        print(f"  - 크기: {len(real_dicts)}")
    
    # Synthetic 데이터셋 분석 (있는 경우)
    # Multi-dataset에서는 별도로 등록되지 않을 수 있음
    if 'MULTI_train' in DatasetCatalog.list():
        multi_dicts = DatasetCatalog.get('MULTI_train')
        print(f"\n📊 Multi 데이터셋:")
        print(f"  - 크기: {len(multi_dicts)}")
        if hasattr(multi_dicts, 'real_size'):
            print(f"  - Real size: {multi_dicts.real_size}")
            print(f"  - Synthetic size: {multi_dicts.synthetic_size}")


def main():
    parser = argparse.ArgumentParser(description='데이터셋 로딩 디버깅 및 분석')
    parser.add_argument('--config-file', type=str, required=True,
                       help='설정 파일 경로')
    parser.add_argument('--analyze-h5', action='store_true',
                       help='H5 파일 구조 분석')
    parser.add_argument('--analyze-mapping', action='store_true',
                       help='매핑 딕셔너리 분석')
    parser.add_argument('--analyze-dataset', type=str, default=None,
                       help='분석할 데이터셋 이름 (예: VG_train, MULTI_train)')
    parser.add_argument('--analyze-cache', action='store_true',
                       help='캐시 파일 분석')
    parser.add_argument('--analyze-all', action='store_true',
                       help='모든 분석 수행')
    
    args = parser.parse_args()
    
    # 설정 로드
    print("="*80)
    print("데이터셋 로딩 디버깅 및 분석")
    print("="*80)
    print(f"\n설정 파일: {args.config_file}")
    
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.freeze()
    
    # 데이터셋 등록
    print("\n데이터셋 등록 중...")
    register_datasets(cfg)
    print(f"등록된 데이터셋: {DatasetCatalog.list()}")
    
    # 분석 수행
    if args.analyze_all:
        args.analyze_h5 = True
        args.analyze_mapping = True
        args.analyze_cache = True
        if cfg.DATASETS.TYPE == "MULTI_DATASET":
            args.analyze_dataset = "MULTI_train"
        elif cfg.DATASETS.TYPE == "VISUAL GENOME":
            args.analyze_dataset = "VG_train"
        elif cfg.DATASETS.TYPE == "SYNTHETIC GENOME":
            args.analyze_dataset = "SYNTHETIC_train"
    
    # H5 파일 분석
    if args.analyze_h5:
        if cfg.DATASETS.TYPE == "VISUAL GENOME":
            h5_path = cfg.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5
            analyze_h5_structure(h5_path)
        elif cfg.DATASETS.TYPE == "SYNTHETIC GENOME":
            h5_path = cfg.DATASETS.SYNTHETIC_GENOME.SYNTHETIC_ATTRIBUTE_H5
            analyze_h5_structure(h5_path)
        elif cfg.DATASETS.TYPE == "MULTI_DATASET":
            # Real 데이터셋
            h5_path = cfg.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5
            analyze_h5_structure(h5_path)
            # Synthetic 데이터셋
            h5_path_syn = cfg.DATASETS.VISUAL_GENOME_SYNTHETIC.VG_ATTRIBUTE_H5
            analyze_h5_structure(h5_path_syn)
    
    # 매핑 딕셔너리 분석
    if args.analyze_mapping:
        if cfg.DATASETS.TYPE == "VISUAL GENOME":
            mapping_path = cfg.DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY
            analyze_mapping_dictionary(mapping_path)
        elif cfg.DATASETS.TYPE == "SYNTHETIC GENOME":
            mapping_path = cfg.DATASETS.SYNTHETIC_GENOME.MAPPING_DICTIONARY
            analyze_mapping_dictionary(mapping_path)
        elif cfg.DATASETS.TYPE == "MULTI_DATASET":
            mapping_path = cfg.DATASETS.VISUAL_GENOME.MAPPING_DICTIONARY
            analyze_mapping_dictionary(mapping_path)
    
    # 캐시 파일 분석
    if args.analyze_cache:
        analyze_cache_files()
    
    # 데이터셋 분석
    if args.analyze_dataset:
        analyze_dataset_dicts(args.analyze_dataset)
        analyze_metadata(args.analyze_dataset)
    
    # 멀티 데이터셋 분석
    if cfg.DATASETS.TYPE == "MULTI_DATASET":
        analyze_multi_dataset(cfg)
    
    print("\n" + "="*80)
    print("분석 완료!")
    print("="*80)


if __name__ == "__main__":
    import torch
    main()

