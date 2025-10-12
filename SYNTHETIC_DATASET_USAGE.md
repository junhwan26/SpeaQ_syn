# Synthetic Genome Dataset Usage Guide

이 문서는 SpeaQ 코드베이스에서 Synthetic Genome 데이터셋을 사용하는 방법을 설명합니다.

## 데이터셋 준비

Synthetic Genome 데이터셋은 Visual Genome과 동일한 포맷을 사용하지만, 실제 이미지 대신 합성 이미지를 포함합니다.

### 필요한 파일들

다음 파일들이 준비되어야 합니다:

1. **이미지 파일들**: `./data/datasets/synthetic_genome/synthetic_images/` 디렉토리에 `.jpg` 파일들
2. **어노테이션 파일**: `synthetic-SGG-with-attri.h5` - Visual Genome과 동일한 HDF5 포맷
3. **메타데이터 파일**: `synthetic_image_data.json` - 이미지 메타데이터
4. **매핑 딕셔너리**: `synthetic-SGG-dicts-with-attri.json` - 클래스 및 관계 매핑

### 디렉토리 구조

```
data/datasets/synthetic_genome/
├── synthetic_images/           # 합성 이미지 파일들
│   ├── 1.jpg
│   ├── 2.jpg
│   └── ...
├── synthetic-SGG-with-attri.h5 # 어노테이션 데이터
├── synthetic_image_data.json   # 이미지 메타데이터
└── synthetic-SGG-dicts-with-attri.json # 클래스 매핑
```

## 설정 파일

### 1. 기본 설정 파일: `configs/synthetic_genome.yaml`

```yaml
DATASETS:
  TYPE: "SYNTHETIC GENOME"
  TRAIN: ('SYNTHETIC_train',)
  TEST: ('SYNTHETIC_val',)
  SYNTHETIC_GENOME:
    IMAGES: './data/datasets/synthetic_genome/synthetic_images/'
    MAPPING_DICTIONARY: './data/datasets/synthetic_genome/synthetic-SGG-dicts-with-attri.json'
    IMAGE_DATA: './data/datasets/synthetic_genome/synthetic_image_data.json'
    SYNTHETIC_ATTRIBUTE_H5: './data/datasets/synthetic_genome/synthetic-SGG-with-attri.h5'
```

### 2. 테스트 설정 파일: `configs/synthetic_genome_test.yaml`

SpeaQ 기능이 활성화된 테스트용 설정입니다.

## 실행 명령어

### 1. 기본 모델 학습 (Synthetic 데이터셋)

```bash
python train_iterative_model.py --resume --num-gpus <NUM_GPUS> \
--config-file configs/synthetic_genome.yaml --dist-url <PORT_NUM> \
OUTPUT_DIR <PATH_TO_CHECKPOINT_DIR> \
SOLVER.IMS_PER_BATCH 20 \
MODEL.WEIGHTS <PATH_TO_PRETRAINED_WEIGHTS>
```

### 2. SpeaQ 모델 학습 (Synthetic 데이터셋)

```bash
python train_iterative_model.py --resume --num-gpus <NUM_GPUS> \
--config-file configs/synthetic_genome.yaml --dist-url <PORT_NUM> \
OUTPUT_DIR <PATH_TO_CHECKPOINT_DIR> \
SOLVER.IMS_PER_BATCH 20 \
MODEL.WEIGHTS <PATH_TO_PRETRAINED_WEIGHTS> \
MODEL.DETR.ONE2MANY_SCHEME dynamic \
MODEL.DETR.MULTIPLY_QUERY 2 \
MODEL.DETR.ONLY_PREDICATE_MULTIPLY True \
MODEL.DETR.ONE2MANY_K 4 \
MODEL.DETR.ONE2MANY_DYNAMIC_SCHEME max \
MODEL.DETR.USE_GROUP_MASK True \
MODEL.DETR.MATCH_INDEPENDENT True \
MODEL.DETR.NUM_GROUPS 4 \
MODEL.DETR.ONE2MANY_PREDICATE_SCORE True \
MODEL.DETR.ONE2MANY_PREDICATE_WEIGHT -0.5
```

### 3. 모델 평가 (Synthetic 데이터셋)

```bash
python train_iterative_model.py --resume --eval-only --num-gpus <NUM_GPUS> \
--config-file configs/synthetic_genome_test.yaml --dist-url <PORT_NUM> \
OUTPUT_DIR <PATH_TO_TRAINED_CHECKPOINT> \
SOLVER.IMS_PER_BATCH 20
```

## 주요 파라미터 설명

### 데이터셋 관련 파라미터

- `DATASETS.TYPE`: "SYNTHETIC GENOME"으로 설정
- `DATASETS.SYNTHETIC_GENOME.IMAGES`: 합성 이미지 디렉토리 경로
- `DATASETS.SYNTHETIC_GENOME.SYNTHETIC_ATTRIBUTE_H5`: 어노테이션 HDF5 파일 경로
- `DATASETS.SYNTHETIC_GENOME.IMAGE_DATA`: 이미지 메타데이터 JSON 파일 경로
- `DATASETS.SYNTHETIC_GENOME.MAPPING_DICTIONARY`: 클래스 매핑 JSON 파일 경로

### SpeaQ 관련 파라미터

- `MODEL.DETR.ONE2MANY_SCHEME`: 'dynamic' - 동적 쿼리 생성 방식
- `MODEL.DETR.MULTIPLY_QUERY`: 2 - 쿼리 증폭 배수
- `MODEL.DETR.ONLY_PREDICATE_MULTIPLY`: True - 관계만 증폭
- `MODEL.DETR.ONE2MANY_K`: 4 - 각 관계에 대한 쿼리 수
- `MODEL.DETR.USE_GROUP_MASK`: True - 그룹 마스크 사용
- `MODEL.DETR.MATCH_INDEPENDENT`: True - 독립적 매칭
- `MODEL.DETR.NUM_GROUPS`: 4 - 그룹 수

## 예시 실행 명령어

### 4 GPU로 SpeaQ 모델 학습

```bash
python train_iterative_model.py --resume --num-gpus 4 \
--config-file configs/synthetic_genome.yaml --dist-url 12345 \
OUTPUT_DIR ./checkpoints/synthetic_speaq \
SOLVER.IMS_PER_BATCH 20 \
MODEL.WEIGHTS ./pretrained/vg_objectdetector_pretrained.pth \
MODEL.DETR.ONE2MANY_SCHEME dynamic \
MODEL.DETR.MULTIPLY_QUERY 2 \
MODEL.DETR.ONLY_PREDICATE_MULTIPLY True \
MODEL.DETR.ONE2MANY_K 4 \
MODEL.DETR.ONE2MANY_DYNAMIC_SCHEME max \
MODEL.DETR.USE_GROUP_MASK True \
MODEL.DETR.MATCH_INDEPENDENT True \
MODEL.DETR.NUM_GROUPS 4 \
MODEL.DETR.ONE2MANY_PREDICATE_SCORE True \
MODEL.DETR.ONE2MANY_PREDICATE_WEIGHT -0.5
```

### 모델 평가

```bash
python train_iterative_model.py --resume --eval-only --num-gpus 4 \
--config-file configs/synthetic_genome_test.yaml --dist-url 12345 \
OUTPUT_DIR ./checkpoints/synthetic_speaq \
SOLVER.IMS_PER_BATCH 20
```

## 주의사항

1. **데이터 포맷**: Synthetic 데이터셋은 Visual Genome과 동일한 HDF5 포맷을 사용해야 합니다.
2. **파일 경로**: 설정 파일에서 데이터 파일 경로를 올바르게 설정해야 합니다.
3. **GPU 메모리**: 합성 이미지의 해상도에 따라 GPU 메모리 사용량이 달라질 수 있습니다.
4. **전처리**: 데이터가 올바르게 전처리되었는지 확인하세요.

## 문제 해결

### 일반적인 오류들

1. **파일을 찾을 수 없음**: 데이터 파일 경로가 올바른지 확인
2. **메모리 부족**: `SOLVER.IMS_PER_BATCH` 값을 줄여보세요
3. **데이터 로딩 오류**: HDF5 파일이 올바른 포맷인지 확인

### 디버깅 팁

- `DATASETS.SYNTHETIC_GENOME.FILTER_EMPTY_RELATIONS: False`로 설정하여 빈 관계도 포함
- 작은 배치 크기로 먼저 테스트
- 로그를 확인하여 데이터 로딩 상태 점검