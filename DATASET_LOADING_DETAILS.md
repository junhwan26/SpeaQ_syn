# 데이터셋 로딩 구현 세부사항

## 목차
1. [통계 정보 계산 (get_statistics)](#통계-정보-계산)
2. [관계 필터링 로직](#관계-필터링-로직)
3. [박스 좌표 변환](#박스-좌표-변환)
4. [샘플링 전략](#샘플링-전략)
5. [멀티 데이터셋 처리 세부사항](#멀티-데이터셋-처리-세부사항)

---

## 통계 정보 계산

### get_statistics() 메서드

**위치**: `data/datasets/visual_genome.py::get_statistics()`

#### 목적

모델 학습에 필요한 통계 정보를 계산합니다:
- **fg_matrix**: 객체-객체-관계 빈도 행렬 (Foreground matrix)
- **bg_matrix**: 객체-객체 빈도 행렬 (Background matrix, 관계 없음)
- **fg_rel_count**: 관계 타입별 빈도
- **pred_dist**: 관계 확률 분포 (로그 스케일)

#### 구현 상세

```python
def get_statistics(self, eps=1e-3, bbox_overlap=True):
    # 1. 클래스 수 가져오기
    num_object_classes = len(MetadataCatalog.get('VG_train').thing_classes) + 1  # +1 for background
    num_relation_classes = len(MetadataCatalog.get('VG_train').predicate_classes) + 1  # +1 for background
    
    # 2. 행렬 초기화
    fg_matrix = np.zeros((num_object_classes, num_object_classes, num_relation_classes))
    bg_matrix = np.zeros((num_object_classes, num_object_classes))
    fg_rel_count = np.zeros((num_relation_classes,))
    
    # 3. 각 이미지의 관계 데이터 수집
    for data in self.dataset_dicts:
        gt_relations = data['relations']  # [N, 3] (subject, object, predicate)
        gt_classes = np.array([x['category_id'] for x in data['annotations']])
        gt_boxes = np.array([x['bbox'] for x in data['annotations']])
        
        # 4. Foreground 관계 카운트
        for (o1, o2), rel in zip(gt_classes[gt_relations[:,:2]], gt_relations[:,2]):
            fg_matrix[o1, o2, rel] += 1  # 객체 o1과 o2 사이의 관계 rel 카운트
            fg_rel_count[rel] += 1  # 관계 타입 rel의 전체 카운트
        
        # 5. Background 관계 카운트 (관계가 없는 객체 쌍)
        if bbox_overlap and len(gt_boxes) > 1:
            # 겹치는 박스 쌍만 고려
            boxes = Boxes(gt_boxes)
            iou_matrix = pairwise_iou(boxes, boxes)
            
            overlap_pairs = []
            for i in range(len(gt_boxes)):
                for j in range(i+1, len(gt_boxes)):
                    if iou_matrix[i, j] > 0:  # IoU > 0
                        overlap_pairs.append((i, j))
            
            for (i, j) in overlap_pairs:
                bg_matrix[gt_classes[i], gt_classes[j]] += 1
        else:
            # 모든 객체 쌍 고려
            for i in range(len(gt_classes)):
                for j in range(i+1, len(gt_classes)):
                    bg_matrix[gt_classes[i], gt_classes[j]] += 1
    
    # 6. Background 관계를 마지막 관계 클래스로 설정
    bg_matrix += 1  # Laplace smoothing
    fg_matrix[:, :, -1] = bg_matrix  # 마지막 관계 = background (no relation)
    
    # 7. 확률 분포 계산 (로그 스케일)
    # fg_matrix.sum(2) = [num_objs, num_objs] 각 객체 쌍의 총 관계 수
    pred_dist = np.log(fg_matrix / (fg_matrix.sum(2)[:, :, None] + eps))
    
    return {
        'fg_matrix': torch.from_numpy(fg_matrix),
        'pred_dist': torch.from_numpy(pred_dist).float(),
        'fg_rel_count': torch.from_numpy(fg_rel_count).float(),
        'obj_classes': self.idx_to_classes + ['__background__'],
        'rel_classes': self.idx_to_predicates + ['__background__'],
        'att_classes': self.idx_to_attributes,
    }
```

#### 행렬 구조 설명

**fg_matrix**:
- Shape: `[num_object_classes, num_object_classes, num_relation_classes]`
- `fg_matrix[i, j, k]` = 객체 클래스 i와 j 사이에 관계 k가 나타난 횟수
- 마지막 차원의 마지막 인덱스는 background (관계 없음)

**pred_dist**:
- Shape: `[num_object_classes, num_object_classes, num_relation_classes]`
- `pred_dist[i, j, k]` = 객체 클래스 i와 j 사이에 관계 k가 나타날 로그 확률
- 계산: `log(fg_matrix[i, j, k] / sum(fg_matrix[i, j, :]))`

**fg_rel_count**:
- Shape: `[num_relation_classes]`
- `fg_rel_count[k]` = 관계 타입 k가 전체 데이터셋에서 나타난 총 횟수

#### 사용 목적

1. **Frequency Bias**: 모델이 객체 쌍에 대한 관계 예측 시 빈도를 고려
2. **Class Balancing**: 불균형한 관계 분포를 보정
3. **Loss Weighting**: 관계 타입별 loss 가중치 계산

---

## 관계 필터링 로직

### 1. FILTER_EMPTY_RELATIONS

**설정**: `cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS`

```python
# _load_graphs()에서
if self.cfg.DATASETS.VISUAL_GENOME.FILTER_EMPTY_RELATIONS:
    split_mask &= self.VG_attribute_h5['img_to_first_rel'][:] >= 0
```

**효과**: 관계가 하나도 없는 이미지를 제외합니다.

**이유**: Scene Graph Detection 태스크에서는 관계가 없는 이미지는 학습에 도움이 되지 않습니다.

### 2. FILTER_NON_OVERLAP

**설정**: `cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP`

```python
# _load_graphs()에서
if self.cfg.DATASETS.VISUAL_GENOME.FILTER_NON_OVERLAP and self.split == 'train':
    boxes_list = Boxes(boxes)
    ious = pairwise_iou(boxes_list, boxes_list)
    relation_boxes_ious = ious[relations[:,0], relations[:,1]]
    iou_indexes = np.where(relation_boxes_ious > 0.0)[0]
    if iou_indexes.size > 0:
        relations = relations[iou_indexes]
    else:
        continue  # 이미지 건너뛰기
```

**효과**: 겹치지 않는 박스 쌍의 관계를 제거합니다.

**이유**: 
- 공간적으로 가까운 객체들 사이의 관계가 더 의미있습니다.
- IoU = 0인 경우 두 객체가 전혀 겹치지 않으므로 관계가 모호할 수 있습니다.

**주의**: 
- `FILTER_NON_OVERLAP=False`일 경우 모든 관계를 유지합니다.
- 일부 데이터셋에서는 겹치지 않는 관계도 유효할 수 있습니다.

### 3. FILTER_DUPLICATE_RELATIONS

**설정**: `cfg.DATASETS.VISUAL_GENOME.FILTER_DUPLICATE_RELATIONS`

**위치**: `data/dataset_mapper.py::DetrDatasetMapper.__call__()`

```python
if self.filter_duplicate_relations and self.is_train:
    relation_dict = defaultdict(list)
    for object_0, object_1, relation in dataset_dict["relations"]:
        relation_dict[(object_0, object_1)].append(relation)
    
    # 같은 객체 쌍에 여러 관계가 있으면 랜덤하게 하나만 선택
    dataset_dict["relations"] = [
        (k[0], k[1], np.random.choice(v)) 
        for k, v in relation_dict.items()
    ]
```

**효과**: 같은 객체 쌍에 여러 관계가 있을 때 하나만 선택합니다.

**이유**: 
- 일부 모델은 객체 쌍당 하나의 관계만 예측합니다.
- 학습 시 하나의 관계만 사용하는 것이 더 안정적입니다.

---

## 박스 좌표 변환

### H5 파일 형식 → Detectron2 형식

#### 1. 좌표 형식 변환

**H5 파일**: 중심점 + 크기 형식 (cx, cy, w, h)
**Detectron2**: 좌상단 + 우하단 형식 (x1, y1, x2, y2)

```python
# _load_graphs()에서
all_boxes = self.VG_attribute_h5['boxes_1024'][:]  # [N, 4] (cx, cy, w, h)

# 변환
all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2  # cx, cy → x1, y1
all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]      # x1, y1 → x2, y2
```

**수식**:
```
x1 = cx - w/2
y1 = cy - h/2
x2 = x1 + w = cx + w/2
y2 = y1 + h = cy + h/2
```

#### 2. 스케일 변환

**BOX_SCALE**: H5 파일에서 박스 좌표가 정규화된 스케일 (일반적으로 1024)

```python
# _load_graphs()에서
resized_box = boxes[obj_idx] / self.cfg.DATASETS.VISUAL_GENOME.BOX_SCALE * max(
    record['height'], record['width']
)
```

**예시**:
- BOX_SCALE = 1024
- 이미지 크기: 768 x 1024
- H5 파일의 박스: [512, 384, 200, 150] (cx=512, cy=384, w=200, h=150)
- 변환 후: [512-100, 384-75, 512+100, 384+75] = [412, 309, 612, 459]
- 스케일 조정: [412*1024/1024, 309*1024/1024, 612*1024/1024, 459*1024/1024]
  = [412, 309, 612, 459] (이 경우 스케일 조정 없음)

**주의**: 실제 이미지 크기와 BOX_SCALE이 다를 경우 스케일 조정이 필요합니다.

---

## 샘플링 전략

### 1. BGNN (Bias-Guided Negative Sampling)

**설정**: `cfg.DATASETS.VISUAL_GENOME.BGNN = True`

#### 목적

불균형한 관계 분포를 보정하기 위해 오버샘플링과 언더샘플링을 결합합니다.

#### 구현

```python
# VisualGenomeTrainData.__init__()에서
if self.bgnn:
    # 1. 관계 빈도 계산
    statistics = self.get_statistics()
    freq = statistics['fg_rel_count'] / statistics['fg_rel_count'].sum()
    
    # 2. 오버샘플링 비율 계산
    oversample_param = cfg.DATASETS.VISUAL_GENOME.OVERSAMPLE_PARAM  # 0.07
    oversampling_ratio = np.maximum(
        np.sqrt((oversample_param / (freq + 1e-5))), 
        np.ones_like(freq)
    )[:-1]  # background 제외
    
    # 3. 각 이미지별 반복 횟수 계산
    sampled_dataset_dicts = []
    for record in self.dataset_dicts:
        relations = record['relations']
        if len(relations) > 0:
            unique_relations = np.unique(relations[:,2])
            repeat_num = int(np.ceil(np.max(oversampling_ratio[unique_relations])))
            
            # 이미지를 repeat_num만큼 반복
            for rep_idx in range(repeat_num):
                sampled_dataset_dicts.append(record)
        else:
            sampled_dataset_dicts.append(record)
    
    # 4. BGNNSampler 생성
    self.dataloader = BGNNSampler(
        sampled_dataset_dicts, 
        sampled_num, 
        oversampling_ratio, 
        undersample_param, 
        unique_relation_ratios, 
        unique_relations_dict
    )
```

#### BGNNSampler 동작

```python
class BGNNSampler(Dataset):
    def __getitem__(self, idx):
        record = self._lst[idx]
        relations = record['relations']
        new_record = copy.deepcopy(record)
        
        if len(relations) > 0:
            # 언더샘플링: 빈번한 관계를 일부 드롭
            unique_relations = self.unique_relations[idx]
            rc = self.unique_relation_ratios[idx]
            ri = self.sampled_num[idx]
            
            # 드롭아웃 확률 계산
            dropout = np.clip(((ri - rc)/ri) * self.undersample_param, 0.0, 1.0)
            
            # 랜덤하게 관계 드롭
            random_arr = np.random.uniform(size=len(relations))
            index_arr = np.array([unique_relations[rel] for rel in relations[:, 2]])
            rel_dropout = dropout[index_arr]
            to_keep = rel_dropout < random_arr
            dropped_relations = relations[to_keep]
            
            new_record['relations'] = dropped_relations
        
        return new_record
```

#### 효과

- **오버샘플링**: 드문 관계를 가진 이미지를 더 많이 사용
- **언더샘플링**: 빈번한 관계를 일부 드롭하여 균형 조정
- 결과적으로 관계 분포가 더 균형잡힙니다.

### 2. Per-Class Dataset

**설정**: `cfg.DATASETS.VISUAL_GENOME.PER_CLASS_DATASET = True`

#### 목적

관계 타입별로 데이터셋을 분할하여 클래스 균형을 유지합니다.

#### 구현

```python
# VisualGenomeTrainData.__init__()에서
if self.per_class_dataset:
    # 1. 관계별로 데이터셋 분할
    per_class_dataset = defaultdict(list)
    for record in self.dataset_dicts:
        relations = record['relations']
        if len(relations) > 0:
            unique_relations = np.unique(relations[:,2])
            for rel in unique_relations:
                per_class_dataset[rel].append(record)
    
    # 2. ClassBalancedSampler 생성
    self.dataloader = ClassBalancedSampler(
        per_class_dataset, 
        len(self.dataset_dicts), 
        sampled_num, 
        oversampling_ratio, 
        undersample_param, 
        unique_relation_ratios, 
        unique_relations_dict
    )
```

#### ClassBalancedSampler 동작

```python
class ClassBalancedSampler(Dataset):
    def __getitem__(self, idx):
        # 랜덤하게 관계 클래스 선택
        class_idx = np.random.randint(self._num_classes)
        
        # 해당 클래스에서 랜덤하게 샘플 선택
        random_example = np.random.randint(len(self._lst[class_idx]))
        record = self._lst[class_idx][random_example]
        
        # 언더샘플링 적용 (BGNN과 동일)
        # ...
        
        return new_record
```

#### 효과

- 각 관계 타입이 균등하게 샘플링됩니다.
- 드문 관계 타입도 충분히 학습됩니다.

---

## 멀티 데이터셋 처리 세부사항

### MultiDatasetDynamicSampler

#### 배치 구성 전략

```python
class MultiDatasetDynamicSampler(Dataset):
    def __init__(self, real_dicts, synthetic_dicts, 
                 real_ratio=0.7, synthetic_ratio=0.3,
                 real_loss_weight=1.0, synthetic_loss_weight=0.5,
                 batch_size=10):
        # 배치당 샘플 수 계산
        self.real_samples_per_batch = int(batch_size * real_ratio)  # 예: 7
        self.synthetic_samples_per_batch = batch_size - self.real_samples_per_batch  # 예: 3
        
        # 전체 크기 계산
        min_size = min(len(real_dicts), len(synthetic_dicts))
        self.total_size = int(min_size / min(real_ratio, synthetic_ratio))
```

#### 샘플링 동작

```python
def __getitem__(self, idx):
    batch_idx = idx // self.batch_size
    item_idx = idx % self.batch_size
    
    # 배치 내 위치에 따라 real/synthetic 선택
    if item_idx < self.real_samples_per_batch:
        # Real 데이터셋에서 샘플링
        real_idx = np.random.randint(0, len(self.real_dicts))
        item = copy.deepcopy(self.real_dicts[real_idx])
        item['dataset_type'] = 'real'
        item['loss_weight'] = self.real_loss_weight  # 1.0
    else:
        # Synthetic 데이터셋에서 샘플링
        synthetic_idx = np.random.randint(0, len(self.synthetic_dicts))
        item = copy.deepcopy(self.synthetic_dicts[synthetic_idx])
        item['dataset_type'] = 'synthetic'
        item['loss_weight'] = self.synthetic_loss_weight  # 0.5
    
    return item
```

#### 특징

1. **배치 단위 비율 보장**: 각 배치에서 정확한 비율 유지
   - 예: batch_size=20, real_ratio=0.7 → 각 배치에 real 14개, synthetic 6개

2. **랜덤 샘플링**: 매번 랜덤하게 선택하여 다양성 확보

3. **Loss 가중치**: 데이터셋 타입별로 다른 loss 가중치 적용
   - Real: 1.0
   - Synthetic: 0.5 (일반적으로 더 낮음)

4. **메타데이터**: 각 샘플에 `dataset_type`과 `loss_weight` 추가

### 통계 정보 결합

```python
def get_statistics(self):
    # 각 데이터셋의 통계 정보 가져오기
    real_stats = self.real_dataset.get_statistics()
    synthetic_stats = self.synthetic_dataset.get_statistics()
    
    # 통계 합산
    combined_fg_rel_count = real_stats['fg_rel_count'] + synthetic_stats['fg_rel_count']
    combined_fg_matrix = real_stats['fg_matrix'] + synthetic_stats['fg_matrix']
    
    # 확률 분포 재계산
    eps = 1e-3
    combined_pred_dist = torch.log(
        combined_fg_matrix / (combined_fg_matrix.sum(2)[:, :, None] + eps)
    )
    
    return {
        'fg_matrix': combined_fg_matrix,
        'pred_dist': combined_pred_dist,
        'fg_rel_count': combined_fg_rel_count,
        'obj_classes': real_stats['obj_classes'],
        'rel_classes': real_stats['rel_classes'],
        'att_classes': real_stats['att_classes'],
    }
```

**효과**: 두 데이터셋의 관계 분포를 결합하여 더 정확한 frequency bias 계산

---

## 주요 주의사항

### 1. 인덱싱 변환

- **H5 파일**: 1-indexed (1, 2, 3, ...)
- **Detectron2**: 0-indexed (0, 1, 2, ...)
- 변환 필요: `category_id = gt_classes[idx] - 1`

### 2. 관계 인덱스 재매핑

박스가 필터링되면 관계 인덱스도 재매핑 필요:

```python
# dataset_mapper.py에서
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
    dataset_dict['relations'] = torch.tensor(new_relations)
```

### 3. 캐시 무효화

- H5 파일 경로 변경 시 자동 무효화 (해시 변경)
- 설정 변경 시 캐시 파일 삭제 필요

### 4. 메모리 사용량

- 대용량 데이터셋의 경우 메모리 부족 주의
- 캐시 파일 크기 확인 필요

---

## 성능 최적화 팁

1. **캐시 활용**: 첫 로드 후 pickle 파일로 저장하여 재사용
2. **병렬 처리**: `DATALOADER.NUM_WORKERS` 설정으로 멀티프로세싱
3. **필터링 최소화**: 불필요한 필터링 옵션 비활성화
4. **배치 크기 조정**: 메모리에 맞게 배치 크기 조정

