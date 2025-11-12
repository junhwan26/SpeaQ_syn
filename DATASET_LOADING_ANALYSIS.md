# 데이터셋 로딩 심층 분석 (Dataset Loading Deep Dive)

## 📋 목차
1. [전체 구조 개요](#전체-구조-개요)
2. [데이터셋 등록 플로우](#데이터셋-등록-플로우)
3. [데이터셋 타입별 상세 분석](#데이터셋-타입별-상세-분석)
4. [데이터 로딩 메커니즘](#데이터-로딩-메커니즘)
5. [캐싱 시스템](#캐싱-시스템)
6. [데이터 변환 과정](#데이터-변환-과정)
7. [멀티 데이터셋 처리](#멀티-데이터셋-처리)

---

## 전체 구조 개요

```
train_iterative_model.py (메인 엔트리)
    ↓
register_datasets(cfg) [data/tools/utils.py]
    ↓
┌─────────────────────────────────────────┐
│  DATASETS.TYPE에 따른 분기              │
├─────────────────────────────────────────┤
│  • VISUAL GENOME                        │
│  • SYNTHETIC GENOME                     │
│  • MULTI_DATASET                        │
└─────────────────────────────────────────┘
    ↓
각 데이터셋 클래스 초기화
    ↓
_fetch_data_dict() → pickle 캐시 확인
    ↓
_process_data() → H5 파일 읽기
    ↓
_load_graphs() → Detectron2 형식 변환
    ↓
DatasetCatalog.register() → Detectron2에 등록
    ↓
DetrDatasetMapper → 학습/추론 시 변환
```

---

## 데이터셋 등록 플로우

### 1. 초기화 지점

**파일**: `train_iterative_model.py`

```python
def setup(args):
    cfg = get_cfg()
    add_dataset_config(cfg)
    add_scenegraph_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    register_datasets(cfg)  # ← 여기서 데이터셋 등록
    default_setup(cfg, args)
    return cfg
```

### 2. 등록 함수

**파일**: `data/tools/utils.py`

```python
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
```

---

## 데이터셋 타입별 상세 분석

### 1. VisualGenomeTrainData

**파일**: `data/datasets/visual_genome.py`

#### 주요 구성 요소

```python
class VisualGenomeTrainData:
    def __init__(self, cfg, split='train'):
        # 1. 설정 로드
        self.cfg = cfg
        self.split = split
        
        # 2. 마스크 파일 경로 설정
        if split == 'train':
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.TRAIN_MASKS
        elif split == 'val':
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.VAL_MASKS
        else:
            self.mask_location = cfg.DATASETS.VISUAL_GENOME.TEST_MASKS
        
        # 3. 필터링 옵션
        self.filter_empty_relations = cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS
        self.filter_non_overlap = cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP
        
        # 4. 데이터 로드 (캐시 우선)
        self.dataset_dicts = self._fetch_data_dict()
        
        # 5. Detectron2에 등록
        self.register_dataset()
        
        # 6. 통계 정보 계산
        statistics = self.get_statistics()
```

#### 데이터 소스

1. **H5 파일**: `VG_ATTRIBUTE_H5`
   - 이미지별 박스, 라벨, 관계 정보
   - 구조:
     ```
     'split': [0=train, 1=val, 2=test]
     'img_to_first_box': 이미지별 첫 박스 인덱스
     'img_to_last_box': 이미지별 마지막 박스 인덱스
     'img_to_first_rel': 이미지별 첫 관계 인덱스
     'img_to_last_rel': 이미지별 마지막 관계 인덱스
     'boxes_1024': [N, 4] (cx, cy, w, h) 형식
     'labels': [N] 객체 클래스
     'attributes': [N, K] 객체 속성
     'relationships': [M, 2] (subject_idx, object_idx)
     'predicates': [M] 관계 타입
     ```

2. **JSON 파일**: 
   - `MAPPING_DICTIONARY`: 클래스/속성/관계 매핑
   - `IMAGE_DATA`: 이미지 메타데이터 (이미지 ID, 크기 등)

#### 데이터 구조 변환

**H5 형식** → **Detectron2 형식**

```python
record = {
    'file_name': '/path/to/image.jpg',
    'image_id': 12345,
    'height': 768,
    'width': 1024,
    'annotations': [
        {
            'bbox': [x1, y1, x2, y2],
            'bbox_mode': BoxMode.XYXY_ABS,
            'category_id': 0,  # 0-indexed
            'attribute': [1, 0, 1, ...],  # 속성 벡터
            'segmentation': [...]  # optional
        },
        ...
    ],
    'relations': np.array([
        [subject_idx, object_idx, predicate_idx],
        ...
    ])  # shape: [N, 3]
}
```

### 2. SyntheticGenomeTrainData

**파일**: `data/datasets/synthetic_genome.py`

VisualGenomeTrainData와 거의 동일하지만:
- 다른 H5 파일 경로 사용 (`SYNTHETIC_ATTRIBUTE_H5`)
- 다른 이미지 디렉토리 사용
- 별도의 이미지 제거 리스트 (`synthetic_images_to_remove.txt`)

### 3. MultiDatasetTrainData

**파일**: `data/datasets/multi_dataset.py`

#### 핵심 특징

1. **두 데이터셋 동시 로드**
   ```python
   # Real dataset
   real_cfg = copy.deepcopy(cfg)
   real_cfg.DATASETS.TYPE = "VISUAL GENOME"
   real_dataset = VisualGenomeTrainData(real_cfg, split=split)
   
   # Synthetic dataset
   synthetic_cfg = copy.deepcopy(cfg)
   synthetic_cfg.DATASETS.TYPE = "VISUAL GENOME"
   synthetic_cfg.DATASETS.VISUAL_GENOME = cfg.DATASETS.VISUAL_GENOME_SYNTHETIC
   synthetic_dataset = VisualGenomeTrainData(synthetic_cfg, split=split)
   ```

2. **동적 샘플링**
   - `MultiDatasetDynamicSampler` 사용
   - 배치 단위로 real/synthetic 비율 유지
   - 예: real 70%, synthetic 30%

3. **통계 정보 결합**
   ```python
   def get_statistics(self):
       real_stats = self.real_dataset.get_statistics()
       synthetic_stats = self.synthetic_dataset.get_statistics()
       
       # 통계 합산
       combined_fg_rel_count = real_stats['fg_rel_count'] + synthetic_stats['fg_rel_count']
       combined_fg_matrix = real_stats['fg_matrix'] + synthetic_stats['fg_matrix']
       combined_pred_dist = torch.log(combined_fg_matrix / ...)
       
       return combined_statistics
   ```

---

## 데이터 로딩 메커니즘

### 1. 캐시 기반 로딩

**파일**: `data/datasets/visual_genome.py::_fetch_data_dict()`

```python
def _fetch_data_dict(self):
    # 캐시 파일명 생성 (설정에 따라 다름)
    fileName = "tmp/visual_genome_{}_data_{}{}{}{}{}{}{}{}_{}.pkl".format(
        self.split, 
        'masks' if self.mask_exists else '', 
        '_oi' if 'oi' in self.mask_location else '', 
        "_clamped" if self.clamped else "", 
        "_precomp" if self.precompute else "", 
        "_clipped" if self.clipped else "", 
        '_overlapfalse' if not self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP else "", 
        '_emptyfalse' if not self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS else '', 
        "_perclass" if self.per_class_dataset else '',
        h5_path_hash  # H5 파일 경로 해시 (중요!)
    )
    
    if os.path.isfile(fileName):
        # 캐시에서 로드
        with open(fileName, 'rb') as inputFile:
            dataset_dicts = pickle.load(inputFile)
    else:
        # 처음 로드 - 처리 후 캐시 저장
        os.makedirs('tmp', exist_ok=True)
        dataset_dicts = self._process_data()
        with open(fileName, 'wb') as inputFile:
            pickle.dump(dataset_dicts, inputFile)
    
    return dataset_dicts
```

**캐시 파일명 예시**:
```
tmp/visual_genome_train_data__overlapfalse_a6814dad.pkl
```

### 2. H5 파일 처리

**파일**: `data/datasets/visual_genome.py::_process_data()`

```python
def _process_data(self):
    # 1. H5 파일 오픈
    self.VG_attribute_h5 = h5py.File(
        self.cfg.DATASETS.VISUAL_GENOME.VG_ATTRIBUTE_H5, 'r'
    )
    
    # 2. 이미지 메타데이터 로드
    image_data = json.load(open(self.cfg.DATASETS.VISUAL_GENOME.IMAGE_DATA, 'r'))
    
    # 3. 손상된 이미지 제거
    self.corrupted_ims = ['1592', '1722', '4616', '4617']
    for img in image_data:
        if str(img['image_id']) in self.corrupted_ims:
            continue
        self.image_data.append(img)
    
    # 4. 마스크 로드 (옵션)
    if self.mask_location != "":
        with open(self.mask_location, 'rb') as f:
            self.masks = pickle.load(f)
    
    # 5. 그래프 데이터 로드
    dataset_dicts = self._load_graphs()
    return dataset_dicts
```

### 3. 그래프 데이터 변환

**파일**: `data/datasets/visual_genome.py::_load_graphs()`

#### 단계별 처리

```python
def _load_graphs(self):
    # 1. Split 필터링
    data_split = self.VG_attribute_h5['split'][:]
    split_flag = 0 if self.split == 'train' else 1 if self.split == 'val' else 2
    split_mask = data_split == split_flag
    
    # 2. 박스가 없는 이미지 필터링
    split_mask &= self.VG_attribute_h5['img_to_first_box'][:] >= 0
    
    # 3. 관계가 없는 이미지 필터링 (옵션)
    if self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS:
        split_mask &= self.VG_attribute_h5['img_to_first_rel'][:] >= 0
    
    image_index = np.where(split_mask)[0]
    
    # 4. 모든 데이터 로드
    all_labels = self.VG_attribute_h5['labels'][:, 0]
    all_attributes = self.VG_attribute_h5['attributes'][:, :]
    all_boxes = self.VG_attribute_h5['boxes_1024'][:]  # cx, cy, w, h
    
    # 5. 박스 형식 변환: (cx, cy, w, h) → (x1, y1, x2, y2)
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]
    
    # 6. 이미지별 인덱스 가져오기
    first_box_index = self.VG_attribute_h5['img_to_first_box'][split_mask]
    last_box_index = self.VG_attribute_h5['img_to_last_box'][split_mask]
    first_relation_index = self.VG_attribute_h5['img_to_first_rel'][split_mask]
    last_relation_index = self.VG_attribute_h5['img_to_last_rel'][split_mask]
    
    # 7. 관계 데이터
    all_relations = self.VG_attribute_h5['relationships'][:]
    all_relation_predicates = self.VG_attribute_h5['predicates'][:, 0]
    
    # 8. 이미지별로 반복하여 record 생성
    dataset_dicts = []
    for idx, _ in enumerate(image_index):
        record = {}
        
        # 이미지 메타데이터
        image_data = self.image_data[image_indexer[idx]]
        record['file_name'] = os.path.join(
            self.cfg.DATASETS.VISUAL_GENOME.IMAGES, 
            '{}.jpg'.format(image_data['image_id'])
        )
        record['image_id'] = image_data['image_id']
        record['height'] = image_data['height']
        record['width'] = image_data['width']
        
        # 박스 및 라벨
        boxes = all_boxes[first_box_index[idx]:last_box_index[idx] + 1, :]
        gt_classes = all_labels[first_box_index[idx]:last_box_index[idx] + 1]
        gt_attributes = all_attributes[first_box_index[idx]:last_box_index[idx] + 1, :]
        
        # 관계
        if first_relation_index[idx] > -1:
            predicates = all_relation_predicates[
                first_relation_index[idx]:last_relation_index[idx] + 1
            ]
            objects = all_relations[
                first_relation_index[idx]:last_relation_index[idx] + 1
            ] - first_box_index[idx]  # 이미지 내 상대 인덱스로 변환
            predicates = predicates - 1  # 1-indexed → 0-indexed
            relations = np.column_stack((objects, predicates))
        else:
            relations = np.zeros((0, 3), dtype=np.int32)
        
        # 필터링: 겹치지 않는 관계 제거 (옵션)
        if self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP and self.split == 'train':
            boxes_list = Boxes(boxes)
            ious = pairwise_iou(boxes_list, boxes_list)
            relation_boxes_ious = ious[relations[:,0], relations[:,1]]
            iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
            if iou_indexes.size > 0:
                relations = relations[iou_indexes]
            else:
                continue  # 이미지 건너뛰기
        
        # 객체 어노테이션 생성
        objects = []
        for obj_idx in range(len(boxes)):
            # 박스 크기 조정 (BOX_SCALE 기준)
            resized_box = boxes[obj_idx] / self.cfg.DATASETS.VISUAL_GENOME.BOX_SCALE * max(
                record['height'], record['width']
            )
            obj = {
                "bbox": resized_box.tolist(),
                "bbox_mode": BoxMode.XYXY_ABS,
                "category_id": gt_classes[obj_idx] - 1,  # 1-indexed → 0-indexed
                "attribute": gt_attributes[obj_idx],
            }
            
            # 세그멘테이션 마스크 (옵션)
            if self.masks is not None:
                gt_masks = self.masks[image_data['image_id']]
                if gt_masks['empty_index'][obj_idx]:
                    refined_poly = []
                    for poly in gt_masks['polygons'][mask_idx]:
                        if len(poly) >= 6:  # 최소 포인트 수
                            refined_poly.append(poly)
                    obj["segmentation"] = refined_poly
                    mask_idx += 1
                else:
                    obj["segmentation"] = []
            
            objects.append(obj)
        
        record['annotations'] = objects
        record['relations'] = relations
        dataset_dicts.append(record)
    
    return dataset_dicts
```

---

## 캐싱 시스템

### 캐시 파일명 생성 규칙

캐시 파일명은 다음 파라미터들을 조합하여 생성됩니다:

1. **기본 정보**: `visual_genome_{split}_data_`
2. **마스크 존재 여부**: `masks` (있으면)
3. **마스크 타입**: `_oi` (OI 포함 시)
4. **Clamped 여부**: `_clamped`
5. **Precompute 여부**: `_precomp`
6. **Clipped 여부**: `_clipped`
7. **Overlap 필터**: `_overlapfalse` (필터링 안 함)
8. **Empty 관계 필터**: `_emptyfalse` (필터링 안 함)
9. **Per-class 샘플링**: `_perclass`
10. **H5 파일 경로 해시**: `_{h5_path_hash}` ⚠️ **중요**

### 캐시 무효화

H5 파일 경로가 변경되면 해시가 달라져 자동으로 새 캐시가 생성됩니다.

---

## 데이터 변환 과정

### 1. DatasetMapper

**파일**: `data/dataset_mapper.py::DetrDatasetMapper`

#### 역할

Detectron2 형식의 dict → 모델 입력 형식

#### 주요 변환

```python
def __call__(self, dataset_dict):
    # 1. 이미지 로드 및 변환
    image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
    
    # 2. 이미지 크기 조정
    image, transforms = T.apply_transform_gens(self.tfm_gens, image)
    
    # 3. 텐서 변환
    dataset_dict["image"] = torch.as_tensor(
        np.ascontiguousarray(image.transpose(2, 0, 1))
    )
    
    # 4. 중복 관계 필터링 (옵션)
    if self.filter_duplicate_relations and self.is_train:
        relation_dict = defaultdict(list)
        for object_0, object_1, relation in dataset_dict["relations"]:
            relation_dict[(object_0,object_1)].append(relation)
        dataset_dict["relations"] = [
            (k[0], k[1], np.random.choice(v)) 
            for k,v in relation_dict.items()
        ]
    
    # 5. 어노테이션 변환
    annos = [
        utils.transform_instance_annotations(obj, transforms, image_shape)
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ]
    instances = utils.annotations_to_instances(annos, image_shape)
    
    # 6. 속성 추가
    attributes = [obj['attribute'] for obj in annos]
    instances.gt_attributes = torch.from_numpy(np.array(attributes, dtype=np.int64))
    
    # 7. 빈 인스턴스 필터링
    dataset_dict["instances"], filter_mask = utils.filter_empty_instances(
        instances, return_mask=True
    )
    
    # 8. 관계 인덱스 재매핑 (필터링된 객체 반영)
    if not filter_mask.all():
        object_mapper = {
            int(old_idx): new_idx 
            for new_idx, old_idx in enumerate(torch.arange(filter_mask.size(0))[filter_mask])
        }
        new_relations = []
        for object_0, object_1, relation in dataset_dict['relations'].numpy():
            if (object_0 in object_mapper) and (object_1 in object_mapper):
                new_relations.append([
                    object_mapper[object_0], 
                    object_mapper[object_1], 
                    relation
                ])
        dataset_dict['relations'] = torch.tensor(new_relations) if new_relations else torch.zeros(0, 3).long()
    
    # 9. 최대 객체/관계 수 제한 (옵션)
    if len(dataset_dict['instances']) > self.max_num_objs:
        sample_idxs = np.random.permutation(
            np.arange(len(dataset_dict['instances']))
        )[:self.max_num_objs]
        dataset_dict['instances'] = dataset_dict['instances'][sample_idxs]
        # 관계 재매핑...
    
    return dataset_dict
```

---

## 멀티 데이터셋 처리

### MultiDatasetDynamicSampler

**파일**: `data/datasets/multi_dataset.py`

#### 동작 원리

```python
class MultiDatasetDynamicSampler(Dataset):
    def __init__(self, real_dicts, synthetic_dicts, 
                 real_ratio=0.7, synthetic_ratio=0.3,
                 real_loss_weight=1.0, synthetic_loss_weight=0.5,
                 batch_size=10):
        self.real_dicts = real_dicts
        self.synthetic_dicts = synthetic_dicts
        self.real_ratio = real_ratio
        self.synthetic_ratio = synthetic_ratio
        self.real_loss_weight = real_loss_weight
        self.synthetic_loss_weight = synthetic_loss_weight
        self.batch_size = batch_size
        
        # 배치당 샘플 수 계산
        self.real_samples_per_batch = int(batch_size * real_ratio)
        self.synthetic_samples_per_batch = batch_size - self.real_samples_per_batch
        
        # 전체 크기 계산
        min_size = min(len(real_dicts), len(synthetic_dicts))
        self.total_size = int(min_size / min(real_ratio, synthetic_ratio))
    
    def __getitem__(self, idx):
        # 배치 인덱스와 배치 내 위치 계산
        batch_idx = idx // self.batch_size
        item_idx = idx % self.batch_size
        
        # 배치 위치에 따라 real/synthetic 선택
        if item_idx < self.real_samples_per_batch:
            # Real 데이터셋에서 샘플링
            real_idx = np.random.randint(0, len(self.real_dicts))
            item = copy.deepcopy(self.real_dicts[real_idx])
            item['dataset_type'] = 'real'
            item['loss_weight'] = self.real_loss_weight
        else:
            # Synthetic 데이터셋에서 샘플링
            synthetic_idx = np.random.randint(0, len(self.synthetic_dicts))
            item = copy.deepcopy(self.synthetic_dicts[synthetic_idx])
            item['dataset_type'] = 'synthetic'
            item['loss_weight'] = self.synthetic_loss_weight
        
        return item
```

#### 특징

1. **배치 단위 비율 보장**: 각 배치에서 정확한 비율 유지
2. **랜덤 샘플링**: 매번 랜덤하게 선택하여 다양성 확보
3. **Loss 가중치**: 데이터셋 타입별로 다른 loss 가중치 적용
4. **메타데이터 추가**: `dataset_type`, `loss_weight` 필드 추가

---

## 주요 설정 파라미터

### Visual Genome 설정

```yaml
DATASETS:
  VISUAL_GENOME:
    IMAGES: '/path/to/images'
    MAPPING_DICTIONARY: '/path/to/mapping.json'
    IMAGE_DATA: '/path/to/image_data.json'
    VG_ATTRIBUTE_H5: '/path/to/data.h5'
    TRAIN_MASKS: ""  # 마스크 파일 경로 (옵션)
    FILTER_EMPTY_RELATIONS: True
    FILTER_NON_OVERLAP: False
    FILTER_DUPLICATE_RELATIONS: True
    BOX_SCALE: 1024
    MAX_NUM_RELATIONS: -1  # -1 = 무제한
    MAX_NUM_OBJECTS: -1
```

### Multi-Dataset 설정

```yaml
DATASETS:
  TYPE: "MULTI_DATASET"
  MULTI_DATASET:
    ENABLED: True
    REAL_SAMPLING_RATIO: 0.7
    SYNTHETIC_SAMPLING_RATIO: 0.3
    REAL_LOSS_WEIGHT: 1.0
    SYNTHETIC_LOSS_WEIGHT: 0.5
```

---

## 성능 최적화 포인트

### 1. 캐싱 활용

- 첫 로드 시 pickle로 저장
- 설정 변경 시에만 재처리 필요
- H5 파일 경로 해시로 자동 무효화

### 2. 메모리 효율성

- H5 파일은 메모리에 전체 로드
- 이미지 파일은 필요 시에만 로드 (DatasetMapper에서)
- 관계 데이터는 스파스 인덱싱으로 접근

### 3. 병렬 처리

- `DATALOADER.NUM_WORKERS` 설정으로 멀티프로세싱
- 각 워커가 독립적으로 데이터 로드

---

## 디버깅 팁

### 1. 캐시 확인

```python
import os
cache_files = [f for f in os.listdir('tmp/') if f.startswith('visual_genome')]
print(cache_files)
```

### 2. 데이터셋 크기 확인

```python
from detectron2.data import DatasetCatalog
dataset_dicts = DatasetCatalog.get('VG_train')
print(f"Dataset size: {len(dataset_dicts)}")
print(f"First sample keys: {dataset_dicts[0].keys()}")
```

### 3. 통계 정보 확인

```python
from detectron2.data import MetadataCatalog
metadata = MetadataCatalog.get('VG_train')
if hasattr(metadata, 'statistics'):
    print(f"Object classes: {len(metadata.thing_classes)}")
    print(f"Predicate classes: {len(metadata.predicate_classes)}")
    print(f"Relation count: {metadata.statistics['fg_rel_count'].sum()}")
```

### 4. H5 파일 구조 확인

```python
import h5py
with h5py.File('VG-SGG-with-attri.h5', 'r') as f:
    print("Keys:", list(f.keys()))
    print("Split shape:", f['split'].shape)
    print("Labels shape:", f['labels'].shape)
```

---

## 주의사항

1. **인덱싱 변환**: H5 파일은 1-indexed, Detectron2는 0-indexed
2. **박스 형식 변환**: (cx, cy, w, h) → (x1, y1, x2, y2)
3. **관계 인덱스**: 필터링 후 객체 인덱스 재매핑 필요
4. **캐시 무효화**: H5 파일 변경 시 캐시 파일 삭제 또는 자동 무효화
5. **메모리**: 대용량 데이터셋의 경우 메모리 부족 주의

---

## 관련 파일 목록

- `data/tools/utils.py`: 데이터셋 등록
- `data/datasets/visual_genome.py`: Visual Genome 데이터셋
- `data/datasets/synthetic_genome.py`: Synthetic 데이터셋
- `data/datasets/multi_dataset.py`: 멀티 데이터셋 처리
- `data/dataset_mapper.py`: 데이터 변환
- `configs/defaults.py`: 기본 설정
- `train_iterative_model.py`: 학습 엔트리 포인트

