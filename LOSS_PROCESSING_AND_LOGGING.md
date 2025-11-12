# 학습 시 Loss 처리 방식 및 로깅 시스템

## 목차
1. [Loss 구조 개요](#loss-구조-개요)
2. [Loss 계산 과정](#loss-계산-과정)
3. [Loss 가중치 시스템](#loss-가중치-시스템)
4. [멀티 데이터셋 Loss 가중치](#멀티-데이터셋-loss-가중치)
5. [로그 출력](#로그-출력)
6. [Loss 종류별 상세 설명](#loss-종류별-상세-설명)

---

## Loss 구조 개요

### 전체 Loss 구조

```
Total Loss = Σ (weight_dict[key] × loss_dict[key])
```

### Loss 구성 요소

1. **Subject/Object Losses** (객체 검출 관련)
   - `loss_ce_subject`: Subject 객체 분류 loss
   - `loss_bbox_subject`: Subject 객체 박스 회귀 loss (L1)
   - `loss_giou_subject`: Subject 객체 GIoU loss
   - `loss_ce_object`: Object 객체 분류 loss
   - `loss_bbox_object`: Object 객체 박스 회귀 loss (L1)
   - `loss_giou_object`: Object 객체 GIoU loss

2. **Relation Losses** (관계 검출 관련)
   - `loss_relation`: 관계 타입 분류 loss
   - `loss_bbox_relation`: 관계 박스 회귀 loss (L1)
   - `loss_giou_relation`: 관계 GIoU loss

3. **Auxiliary Losses** (Deep Supervision)
   - 각 decoder layer의 auxiliary output에 대한 loss
   - 형식: `{loss_name}_{layer_index}` (예: `loss_relation_0`, `loss_ce_subject_1`)

4. **기타 Losses**
   - `loss_nms`: NMS 관련 loss (옵션)
   - `loss_selection_subject/object`: 선택 loss (옵션)
   - `k_log_layer{N}`: One-to-Many matching 관련 로그

---

## Loss 계산 과정

### 1. Forward Pass

**파일**: `modeling/transformer/criterion.py::IterativeRelationCriterion.forward()`

```python
def forward(self, outputs, targets):
    """
    outputs: 모델 출력
        - relation_logits: 관계 분류 로짓
        - relation_boxes: 관계 박스
        - relation_subject_logits: Subject 분류 로짓
        - relation_subject_boxes: Subject 박스
        - relation_object_logits: Object 분류 로짓
        - relation_object_boxes: Object 박스
        - aux_outputs_r: Auxiliary relation outputs
        - aux_outputs_r_sub: Auxiliary subject outputs
        - aux_outputs_r_obj: Auxiliary object outputs
    
    targets: Ground truth
        - combined_boxes: 결합된 박스
        - combined_labels: 결합된 라벨
        - relation_boxes: 관계 박스
        - relation_labels: 관계 라벨
        - loss_weight: 샘플별 loss 가중치 (멀티 데이터셋)
    """
    losses = {}
    
    # 1. Loss weights 추출 (멀티 데이터셋용)
    loss_weights = []
    for target in targets:
        if 'loss_weight' in target:
            loss_weights.append(target['loss_weight'])
        else:
            loss_weights.append(1.0)  # 기본값
    loss_weights = torch.tensor(loss_weights, device=device, dtype=torch.float32)
    
    # 2. Relation Branch Losses
    relation_outputs = {...}  # Relation 관련 출력
    combined_indices, k_mean_log, augmented_targets = self.matcher.forward_relation(...)
    
    # k_log (One-to-Many matching 로그)
    losses.update({'k_log_layer{}'.format(l+1): k_mean_log})
    
    # 3. Entity 및 Relation Losses 계산
    entity_targets = [...]
    relation_targets = [...]
    kwargs = {'aux_loss': False, 'loss_weights': loss_weights}
    losses.update(self.get_relation_losses(...))
    
    # 4. Auxiliary Losses (Deep Supervision)
    if 'aux_outputs_r' in outputs:
        for i, (aux_outputs_r, ...) in enumerate(zip(...)):
            kwargs = {'log': False, 'aux_loss': True, 'loss_weights': loss_weights}
            aux_losses = self.get_relation_losses(...)
            aux_losses = {k + f'_{i}': v for k, v in aux_losses.items()}
            losses.update(aux_losses)
    
    return losses
```

### 2. get_relation_losses()

**위치**: `modeling/transformer/criterion.py::get_relation_losses()`

```python
def get_relation_losses(self, relation_outputs, entity_targets, relation_targets, combined_indices, **kwargs):
    losses = {}
    
    # 1. Subject Losses
    num_subject_boxes = sum(len(t[1]) for t in combined_indices['subject'])
    for loss in self.losses:  # ['labels', 'boxes']
        if loss == 'labels':
            subject_losses = self.get_loss(loss, ..., combined_indices['subject'], ...)
        if loss == 'boxes':
            subject_losses = self.get_loss(loss, ..., combined_indices['subject'], ...)
        if subject_losses is not None:
            subject_losses = {k + '_subject': v for k, v in subject_losses.items()}
            losses.update(subject_losses)
    
    # 2. Object Losses
    num_object_boxes = sum(len(t[1]) for t in combined_indices['object'])
    for loss in self.losses:
        if loss == 'labels':
            object_losses = self.get_loss(loss, ..., combined_indices['object'], ...)
        if loss == 'boxes':
            object_losses = self.get_loss(loss, ..., combined_indices['object'], ...)
        if object_losses is not None:
            object_losses = {k + '_object': v for k, v in object_losses.items()}
            losses.update(object_losses)
    
    # 3. Relation Losses
    num_relation_boxes = sum(len(t[1]) for t in combined_indices['relation'])
    losses.update(self.get_relation_loss(..., combined_indices['relation'], ...))
    
    return losses
```

### 3. get_relation_loss()

**위치**: `modeling/transformer/criterion.py::get_relation_loss()`

```python
def get_relation_loss(self, outputs, targets, indices, num_relation_boxes, **kwargs):
    # 1. 관계 분류 Loss
    src_logits = outputs['relation_logits']
    idx = self._get_src_permutation_idx_rel(indices)
    
    # Loss weights 적용 (멀티 데이터셋)
    loss_weights = kwargs.get('loss_weights', None)
    
    # 기본 Cross Entropy Loss
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_rel_weight)
    
    # 샘플별 가중치 적용
    if loss_weights is not None and len(idx[0]) > 0:
        batch_indices = idx[0]
        sample_weights = loss_weights[batch_indices]
        
        # Per-sample loss 계산
        losses_per_sample = F.cross_entropy(
            src_logits_matched, target_classes_matched, 
            self.empty_rel_weight, reduction='none'
        )
        
        # 가중치 적용
        weighted_losses = losses_per_sample * sample_weights
        loss_ce = weighted_losses.mean()
    
    losses = {'loss_relation': loss_ce}
    
    # 2. 관계 박스 회귀 Loss (L1)
    if len(idx[0]) > 0:
        src_boxes = outputs['relation_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        
        # Loss weights 적용
        if loss_weights is not None:
            sample_weights = loss_weights[batch_indices].unsqueeze(-1)
            loss_bbox = loss_bbox * sample_weights
        
        losses['loss_bbox_relation'] = loss_bbox.sum() / num_relation_boxes
    
    # 3. 관계 GIoU Loss
    loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(...))
    
    # Loss weights 적용
    if loss_weights is not None:
        sample_weights = loss_weights[batch_indices]
        loss_giou = loss_giou * sample_weights
    
    losses['loss_giou_relation'] = loss_giou.sum() / num_relation_boxes
    
    return losses
```

---

## Loss 가중치 시스템

### weight_dict 정의

**파일**: `modeling/meta_arch/detr.py::Detr.__init__()`

```python
weight_dict = {
    "loss_ce": cost_class,              # 객체 분류 loss 가중치
    "loss_bbox": l1_weight,             # 박스 회귀 loss 가중치 (L1)
    "loss_giou": giou_weight,           # GIoU loss 가중치
    
    "loss_ce_subject": cost_class,      # Subject 분류 loss 가중치
    "loss_ce_object": cost_class,       # Object 분류 loss 가중치
    "loss_bbox_subject": l1_weight,     # Subject 박스 loss 가중치
    "loss_bbox_object": l1_weight,      # Object 박스 loss 가중치
    "loss_giou_subject": giou_weight,   # Subject GIoU loss 가중치
    "loss_giou_object": giou_weight,    # Object GIoU loss 가중치
    
    "loss_relation": 1,                 # 관계 분류 loss 가중치
    "loss_bbox_relation": l1_weight,    # 관계 박스 loss 가중치
    "loss_giou_relation": giou_weight,  # 관계 GIoU loss 가중치
    
    "loss_nms": nms_weight,             # NMS loss 가중치
    "loss_selection_subject": cost_selection,
    "loss_selection_object": cost_selection,
}

# Deep Supervision을 위한 Auxiliary Loss 가중치
if deep_supervision:
    for i in range(max(dec_layers, obj_dec_layers) - 1):
        aux_weight_dict = {
            k + f"_{i}": v for k, v in weight_dict.items()
        }
        weight_dict.update(aux_weight_dict)
```

### 설정 파일에서의 가중치

**파일**: `configs/speaq_multi_dataset.yaml`

```yaml
MODEL:
  DETR:
    COST_CLASS: 2.0      # → cost_class
    L1_WEIGHT: 5.0       # → l1_weight
    GIOU_WEIGHT: 2.0     # → giou_weight
    RELATION_LOSS_WEIGHT: 1.0
    NO_OBJECT_WEIGHT: 0.1
    NO_REL_WEIGHT: 0.1
```

### Total Loss 계산

**파일**: `detectron2/engine/defaults.py` (Detectron2 내부)

```python
# Detectron2의 DefaultTrainer에서
loss_dict = self.model(batched_inputs)

# weight_dict를 사용하여 가중치 적용
total_loss = sum(
    weight_dict.get(k, 1.0) * v 
    for k, v in loss_dict.items() 
    if 'loss' in k
)
```

---

## 멀티 데이터셋 Loss 가중치

### 동작 방식

멀티 데이터셋 학습 시, 각 샘플에 대해 다른 loss 가중치를 적용합니다.

**데이터셋 타입별 가중치**:
- **Real 데이터셋**: `loss_weight = 1.0` (기본값)
- **Synthetic 데이터셋**: `loss_weight = 0.5` (설정 가능)

### 구현

**1. 데이터셋에서 가중치 추가**

**파일**: `data/datasets/multi_dataset.py::MultiDatasetDynamicSampler.__getitem__()`

```python
def __getitem__(self, idx):
    if item_idx < self.real_samples_per_batch:
        # Real 데이터셋
        item = copy.deepcopy(self.real_dicts[real_idx])
        item['dataset_type'] = 'real'
        item['loss_weight'] = self.real_loss_weight  # 1.0
    else:
        # Synthetic 데이터셋
        item = copy.deepcopy(self.synthetic_dicts[synthetic_idx])
        item['dataset_type'] = 'synthetic'
        item['loss_weight'] = self.synthetic_loss_weight  # 0.5
    
    return item
```

**2. Loss 계산 시 가중치 적용**

**파일**: `modeling/transformer/criterion.py::IterativeRelationCriterion.forward()`

```python
# 1. Targets에서 loss_weight 추출
loss_weights = []
for target in targets:
    if 'loss_weight' in target:
        loss_weights.append(target['loss_weight'])
    else:
        loss_weights.append(1.0)  # 기본값

loss_weights = torch.tensor(loss_weights, device=device, dtype=torch.float32)

# 2. 각 loss 함수에 loss_weights 전달
kwargs = {'aux_loss': False, 'loss_weights': loss_weights}
losses.update(self.get_relation_losses(..., **kwargs))
```

**3. 샘플별 가중치 적용**

**파일**: `modeling/transformer/criterion.py::get_relation_loss()`

```python
# 관계 분류 loss에 가중치 적용
if loss_weights is not None and len(idx[0]) > 0:
    batch_indices = idx[0]  # 매칭된 샘플의 배치 인덱스
    sample_weights = loss_weights[batch_indices]  # 해당 샘플들의 가중치
    
    # Per-sample loss 계산
    losses_per_sample = F.cross_entropy(
        src_logits_matched, target_classes_matched, 
        self.empty_rel_weight, reduction='none'  # reduction='none': 샘플별 loss
    )
    
    # 가중치 적용 후 평균
    weighted_losses = losses_per_sample * sample_weights
    loss_ce = weighted_losses.mean()
```

### 효과

- **Real 데이터셋**: 전체 loss에 1.0 가중치로 기여
- **Synthetic 데이터셋**: 전체 loss에 0.5 가중치로 기여
- 결과적으로 Real 데이터셋의 영향력이 더 큽니다.

#### Weighted Sum 계산 과정

**핵심 아이디어**: 각 샘플의 loss를 개별적으로 계산한 후, 해당 샘플의 `loss_weight`를 곱하여 가중치를 적용합니다.

**수식**:
```
L_weighted = (1/N) * Σᵢ (wᵢ * Lᵢ)

여기서:
- N: 매칭된 샘플 수
- Lᵢ: i번째 샘플의 loss
- wᵢ: i번째 샘플의 loss_weight
  - Real 샘플: wᵢ = 1.0
  - Synthetic 샘플: wᵢ = 0.5
```

**구현 단계**:
1. **Per-sample loss 계산**: `reduction='none'`을 사용하여 샘플별 loss 계산
2. **Weight 적용**: 각 샘플의 loss에 해당 샘플의 `loss_weight` 곱하기
3. **평균 계산**: 가중치가 적용된 loss들의 평균 계산

**예시**:
```python
# 배치: Real 2개, Synthetic 2개
loss_weights = [1.0, 1.0, 0.5, 0.5]

# Per-sample losses
losses_per_sample = [0.8, 1.2, 0.9, 0.7]

# Weighted losses
weighted_losses = [0.8*1.0, 1.2*1.0, 0.9*0.5, 0.7*0.5]
                = [0.8, 1.2, 0.45, 0.35]

# Final loss
loss = weighted_losses.mean() = (0.8 + 1.2 + 0.45 + 0.35) / 4 = 0.7
```

**자세한 설명**: `REAL_SYNTHETIC_WEIGHTED_SUM.md` 파일 참고

### Real/Synthetic Loss 분리 로깅

**기능**: Weighted sum 하기 전에 Real과 Synthetic 샘플의 loss를 각각 분리하여 로그로 출력합니다.

**출력되는 메트릭**:
- `loss_relation_real`: Real 샘플의 relation classification loss (weighted sum 전)
- `loss_relation_synthetic`: Synthetic 샘플의 relation classification loss (weighted sum 전)
- `loss_bbox_relation_real`: Real 샘플의 relation bbox regression loss
- `loss_bbox_relation_synthetic`: Synthetic 샘플의 relation bbox regression loss
- `loss_giou_relation_real`: Real 샘플의 relation GIoU loss
- `loss_giou_relation_synthetic`: Synthetic 샘플의 relation GIoU loss

**구현 위치**: `modeling/transformer/criterion.py::IterativeRelationCriterion.get_relation_loss()`

**자세한 설명**: `REAL_SYNTHETIC_LOSS_LOGGING.md` 파일 참고

---

## 로그 출력

### Loss 로그 출력 플로우

```
모델 Forward Pass
    ↓
Loss Dictionary 생성 (criterion.forward())
    ↓
DefaultTrainer.run_step()
    ↓
_write_metrics(loss_dict, data_time)
    ↓
SimpleTrainer.write_metrics()
    ├─→ Loss를 CPU로 이동 및 스칼라 변환
    ├─→ 분산 학습 시 모든 워커에서 수집 (gather)
    ├─→ 메인 프로세스에서 평균 계산
    └─→ EventStorage에 저장 (put_scalar, put_scalars)
    ↓
EventStorage
    ├─→ 메트릭 저장 및 스무딩
    └─→ Writer들에게 전달
    ↓
PeriodicWriter (20 iteration마다)
    ├─→ CommonMetricPrinter (콘솔 출력)
    ├─→ JSONWriter (JSON 파일)
    ├─→ TensorboardXWriter (TensorBoard)
    └─→ WandbWriter (Weights & Biases)
```

### 1. Loss Dictionary 생성

**파일**: `modeling/transformer/criterion.py::IterativeRelationCriterion.forward()`

```python
def forward(self, outputs, targets):
    losses = {}
    
    # Relation losses 계산
    losses.update(self.get_relation_losses(...))
    
    # Auxiliary losses 계산
    for i, aux_outputs in enumerate(outputs['aux_outputs_r']):
        aux_losses = self.get_relation_losses(...)
        aux_losses = {k + f'_{i}': v for k, v in aux_losses.items()}
        losses.update(aux_losses)
    
    return losses  # 예: {'loss_ce_subject': tensor(1.2), 'loss_bbox_subject': tensor(2.3), ...}
```

### 2. Loss Metrics 변환

**파일**: `detectron2/engine/train_loop.py::SimpleTrainer.write_metrics()`

```python
@staticmethod
def write_metrics(loss_dict: Mapping[str, torch.Tensor], data_time: float, prefix: str = ""):
    """
    Args:
        loss_dict: dict of scalar losses (예: {'loss_ce_subject': tensor(1.2), ...})
        data_time: 데이터 로딩 시간
        prefix: 로깅 키에 추가할 접두사
    """
    # 1. Tensor를 CPU로 이동하고 스칼라 값으로 변환
    metrics_dict = {
        k: v.detach().cpu().item() 
        for k, v in loss_dict.items()
    }
    metrics_dict["data_time"] = data_time
    
    # 2. 분산 학습 시 모든 워커에서 메트릭 수집
    all_metrics_dict = comm.gather(metrics_dict)
    # all_metrics_dict = [{'loss_ce_subject': 1.2, ...}, ...]  # 각 워커의 메트릭
    
    if comm.is_main_process():
        storage = get_event_storage()
        
        # 3. Data time은 워커 중 최대값 사용 (병목 지점)
        data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
        storage.put_scalar("data_time", data_time)
        
        # 4. 나머지 메트릭은 평균 계산
        metrics_dict = {
            k: np.mean([x[k] for x in all_metrics_dict]) 
            for k in all_metrics_dict[0].keys()
        }
        # metrics_dict = {'loss_ce_subject': 1.15, 'loss_bbox_subject': 2.2, ...}
        
        # 5. Total loss 계산 (모든 loss의 합)
        total_losses_reduced = sum(metrics_dict.values())
        
        # 6. NaN/Inf 체크
        if not np.isfinite(total_losses_reduced):
            raise FloatingPointError(
                f"Loss became infinite or NaN at iteration={storage.iter}!\n"
                f"loss_dict = {metrics_dict}"
            )
        
        # 7. EventStorage에 저장
        storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced)
        if len(metrics_dict) > 1:
            storage.put_scalars(**metrics_dict)  # 모든 개별 loss 저장
```

### 3. EventStorage에 저장

**EventStorage**는 Detectron2의 중앙 메트릭 저장소입니다.

```python
# EventStorage 내부 동작
storage.put_scalar("total_loss", 321.3)  # 단일 스칼라 저장
storage.put_scalars(
    loss_ce_subject=0.9114,
    loss_bbox_subject=1.591,
    loss_giou_subject=1.276,
    loss_ce_object=0.9101,
    loss_bbox_object=1.552,
    loss_giou_object=1.079,
    loss_relation=6.085,
    loss_bbox_relation=2.163,
    loss_giou_relation=2.302,
    loss_ce_subject_0=0.9265,  # Auxiliary loss
    loss_bbox_subject_0=1.75,
    # ... (모든 loss)
)
```

**특징**:
- **스무딩**: 최근 값들의 평균을 계산하여 노이즈 감소
- **이력 관리**: 각 메트릭의 최근 N개 값 저장
- **Writer 연동**: 등록된 모든 Writer에게 값 전달

### 4. Writer를 통한 로그 출력

**파일**: `engine/trainer.py::build_writers()`

```python
def build_writers(self):
    if self.cfg.WANDB.USE_WANDB:
        return [
            CommonMetricPrinter(max_iter),      # 콘솔 출력
            JSONWriter(os.path.join(output_dir, "metrics.json")),  # JSON 파일
            TensorboardXWriter(output_dir),      # TensorBoard
            WandbWriter(self.cfg, self.model)    # Weights & Biases
        ]
    else:
        return [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(output_dir, "metrics.json")),
            TensorboardXWriter(output_dir),
        ]
```

**로그 출력 주기**: `engine/trainer.py::build_hooks()`

```python
if comm.is_main_process():
    ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
```

- **주기**: 20 iteration마다 로그 출력
- **출력 대상**: 메인 프로세스만 (분산 학습 시)

#### 4.1 CommonMetricPrinter (콘솔 출력)

**파일**: Detectron2 내부 (`detectron2/utils/events.py`)

**출력 형식**:
```
[11/07 14:28:51] d2.utils.events INFO:  eta: 19:25:50  iter: 79  total_loss: 321.3  loss_ce_subject: 0.9114  loss_bbox_subject: 1.591  loss_giou_subject: 1.276  loss_ce_object: 0.9101  loss_bbox_object: 1.552  loss_giou_object: 1.079  loss_relation: 6.085  loss_bbox_relation: 2.163  loss_giou_relation: 2.302  loss_ce_subject_0: 0.9265  loss_bbox_subject_0: 1.75  ...  lr: 1.00e-04  data_time: 0.012  time: 0.234
```

**특징**:
- 실시간 콘솔 출력
- 모든 loss를 한 줄에 출력
- ETA (예상 완료 시간), iteration, learning rate, data_time, time 포함

#### 4.2 JSONWriter (JSON 파일)

**파일**: `outputs/{exp_name}/metrics.json`

**출력 형식**:
```json
{
  "iteration": 79,
  "total_loss": 321.3,
  "loss_ce_subject": 0.9114,
  "loss_bbox_subject": 1.591,
  "loss_giou_subject": 1.276,
  "loss_ce_object": 0.9101,
  "loss_bbox_object": 1.552,
  "loss_giou_object": 1.079,
  "loss_relation": 6.085,
  "loss_bbox_relation": 2.163,
  "loss_giou_relation": 2.302,
  "loss_ce_subject_0": 0.9265,
  "loss_bbox_subject_0": 1.75,
  "...": "...",
  "lr": 0.0001,
  "data_time": 0.012,
  "time": 0.234
}
```

**특징**:
- 모든 iteration의 메트릭이 JSON 형식으로 저장
- 나중에 파싱하여 분석 가능

#### 4.3 TensorboardXWriter (TensorBoard)

**파일**: `outputs/{exp_name}/events.out.tfevents.*`

**특징**:
- TensorBoard에서 시각화 가능
- 각 loss의 시간에 따른 변화 그래프
- 명령어: `tensorboard --logdir outputs/{exp_name}`

#### 4.4 WandbWriter (Weights & Biases)

**파일**: `engine/trainer.py::WandbWriter.write()`

```python
def write(self):
    storage = get_event_storage()
    log_dict = {}
    
    # EventStorage에서 최신 메트릭 가져오기 (스무딩 적용)
    for k, v in storage.latest_with_smoothing_hint().items():
        log_dict.update({k: float(v[0])}, step=v[1])
        # v[0]: 스무딩된 값, v[1]: iteration
    
    self._writer.log(log_dict)  # W&B에 로그
```

**특징**:
- 실시간 웹 대시보드
- 하이퍼파라미터 추적
- 실험 비교 및 공유


### 출력되는 Loss 목록

다음 loss들이 로그에 출력됩니다:

1. **total_loss**: 전체 loss 합 (가중치 적용 전 원본 loss들의 합)
2. **loss_ce_subject**: Subject 분류 loss
3. **loss_bbox_subject**: Subject 박스 회귀 loss (L1)
4. **loss_giou_subject**: Subject GIoU loss
5. **loss_ce_object**: Object 분류 loss
6. **loss_bbox_object**: Object 박스 회귀 loss (L1)
7. **loss_giou_object**: Object GIoU loss
8. **loss_relation**: 관계 분류 loss
9. **loss_bbox_relation**: 관계 박스 회귀 loss (L1)
10. **loss_giou_relation**: 관계 GIoU loss
11. **k_log_layer{N}**: One-to-Many matching 로그 (옵션)
12. **Auxiliary losses**: `loss_*_{layer_idx}` 형식 (Deep Supervision)
    - 예: `loss_ce_subject_0`, `loss_bbox_subject_0`, `loss_giou_subject_0`
    - 예: `loss_ce_subject_1`, `loss_bbox_subject_1`, `loss_giou_subject_1`
    - ... (각 decoder layer마다)

**참고**: 
- `total_loss`는 **가중치가 적용되기 전** 원본 loss들의 합입니다.
- 실제 학습에 사용되는 loss는 `weight_dict`를 통해 가중치가 적용된 값입니다.
- 로그에 출력되는 각 loss 값은 **가중치가 적용되기 전** 원본 값입니다.

### 실제 로그 출력 예시

**콘솔 출력** (20 iteration마다):
```
[11/07 14:28:51] d2.utils.events INFO:  eta: 19:25:50  iter: 79  total_loss: 321.3  loss_ce_subject: 0.9114  loss_bbox_subject: 1.591  loss_giou_subject: 1.276  loss_ce_object: 0.9101  loss_bbox_object: 1.552  loss_giou_object: 1.079  loss_relation: 6.085  loss_bbox_relation: 2.163  loss_giou_relation: 2.302  loss_ce_subject_0: 0.9265  loss_bbox_subject_0: 1.75  loss_giou_subject_0: 1.334  loss_ce_object_0: 0.8533  loss_bbox_object_0: 1.446  loss_giou_object_0: 1.022  loss_relation_0: 6.274  loss_bbox_relation_0: 2.127  loss_giou_relation_0: 2.36  ...  lr: 1.00e-04  data_time: 0.012  time: 0.234
```

**설명**:
- `iter: 79`: 현재 iteration
- `total_loss: 321.3`: 모든 loss의 합
- `loss_ce_subject: 0.9114`: Subject 분류 loss
- `loss_ce_subject_0: 0.9265`: 첫 번째 decoder layer의 Subject 분류 loss (Auxiliary)
- `lr: 1.00e-04`: Learning rate
- `data_time: 0.012`: 데이터 로딩 시간 (초)
- `time: 0.234`: 전체 iteration 시간 (초)
- `eta: 19:25:50`: 예상 완료 시간

### 로그 파싱 스크립트

**파일**: `vis_train_log.py`

학습 로그를 파싱하여 시각화하는 스크립트가 제공됩니다:

```bash
python vis_train_log.py --log-file outputs/exp_name/log.txt --output-dir training_plots
```

---

## Loss 종류별 상세 설명

### 1. Classification Loss (Cross Entropy)

**위치**: `modeling/transformer/criterion.py::loss_labels()`

```python
loss_ce = F.cross_entropy(
    src_logits.transpose(1, 2),  # [batch, num_queries, num_classes]
    target_classes,               # [batch, num_queries]
    self.empty_weight             # 클래스별 가중치 (불균형 보정)
)
```

**특징**:
- **empty_weight**: Background 클래스에 대한 가중치 (`eos_coef`)
- **Reweighting**: 관계 타입별 빈도 기반 가중치 적용 (옵션)

### 2. Box Regression Loss (L1)

**위치**: `modeling/transformer/criterion.py::loss_boxes()`

```python
loss_bbox = F.l1_loss(
    src_boxes,      # [num_matched, 4] (cx, cy, w, h)
    target_boxes,   # [num_matched, 4]
    reduction='none'
)
loss_bbox = loss_bbox.sum() / num_boxes  # 정규화
```

**특징**:
- 박스 좌표 (cx, cy, w, h)에 대한 L1 loss
- 매칭된 박스 쌍에 대해서만 계산

### 3. GIoU Loss

**위치**: `modeling/transformer/criterion.py::loss_boxes()`

```python
loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
    box_ops.box_cxcywh_to_xyxy(src_boxes),
    box_ops.box_cxcywh_to_xyxy(target_boxes)
))
loss_giou = loss_giou.sum() / num_boxes  # 정규화
```

**특징**:
- Generalized IoU 사용
- 박스 겹침 정도를 더 잘 측정

### 4. Relation Classification Loss

**위치**: `modeling/transformer/criterion.py::get_relation_loss()`

```python
loss_ce = F.cross_entropy(
    src_logits.transpose(1, 2),  # [batch, num_queries, num_rel_classes]
    target_classes,               # [batch, num_queries]
    self.empty_rel_weight         # 관계 타입별 가중치
)
```

**특징**:
- **empty_rel_weight**: 관계 타입별 빈도 기반 가중치
- **Reweighting**: `REWEIGHT_RELATIONS=True` 시 빈도 기반 재가중치

### 5. Relation Box Losses

관계 박스에 대한 L1 및 GIoU loss는 객체 박스 loss와 동일한 방식으로 계산됩니다.

---

## Loss 처리 플로우 요약

```
배치 데이터 입력
    ↓
모델 Forward Pass
    ↓
Criterion.forward()
    ├─→ Loss weights 추출 (멀티 데이터셋)
    ├─→ Matcher (Hungarian Matching)
    ├─→ get_relation_losses()
    │   ├─→ Subject Losses (CE, L1, GIoU)
    │   ├─→ Object Losses (CE, L1, GIoU)
    │   └─→ Relation Losses (CE, L1, GIoU)
    └─→ Auxiliary Losses (Deep Supervision)
    ↓
Loss Dictionary 반환
    ↓
Weight Dict 적용
    ↓
Total Loss = Σ(weight × loss)
    ↓
Backward Pass
    ↓
Optimizer Step
    ↓
로그 출력 (20 iteration마다)
```

---

## 주요 설정 파라미터

### Loss 가중치

```yaml
MODEL:
  DETR:
    COST_CLASS: 2.0           # Classification loss 가중치
    L1_WEIGHT: 5.0            # L1 loss 가중치
    GIOU_WEIGHT: 2.0          # GIoU loss 가중치
    RELATION_LOSS_WEIGHT: 1.0 # Relation loss 가중치
    NO_OBJECT_WEIGHT: 0.1     # Background 객체 가중치
    NO_REL_WEIGHT: 0.1        # Background 관계 가중치
```

### 멀티 데이터셋 Loss 가중치

```yaml
DATASETS:
  MULTI_DATASET:
    REAL_LOSS_WEIGHT: 1.0      # Real 데이터셋 loss 가중치
    SYNTHETIC_LOSS_WEIGHT: 0.5 # Synthetic 데이터셋 loss 가중치
```

### 관계 재가중치

```yaml
MODEL:
  DETR:
    REWEIGHT_RELATIONS: True   # 관계 타입별 재가중치 활성화
    REWEIGHT_REL_EOS_COEF: 0.1 # Background 관계 가중치
    OVERSAMPLE_PARAM: 0.07     # 오버샘플링 파라미터
    UNDERSAMPLE_PARAM: 1.5     # 언더샘플링 파라미터
```

---

## 디버깅 팁

### 1. Loss 값 확인

```python
# 모델 출력 확인
loss_dict = model(batched_inputs)
print("Loss dict:", loss_dict)

# Total loss 계산
total_loss = sum(
    weight_dict.get(k, 1.0) * v 
    for k, v in loss_dict.items() 
    if 'loss' in k
)
print("Total loss:", total_loss)
```

### 2. Loss 가중치 확인

```python
# Weight dict 확인
print("Weight dict:", criterion.weight_dict)

# 실제 적용된 가중치 확인
for k, v in loss_dict.items():
    weight = weight_dict.get(k, 1.0)
    weighted_loss = weight * v
    print(f"{k}: {v:.4f} × {weight} = {weighted_loss:.4f}")
```

### 3. 멀티 데이터셋 Loss 가중치 확인

```python
# 데이터셋 타입별 loss 확인
for i, target in enumerate(targets):
    dataset_type = target.get('dataset_type', 'unknown')
    loss_weight = target.get('loss_weight', 1.0)
    print(f"Sample {i}: {dataset_type}, loss_weight={loss_weight}")
```

---

## 참고 파일

- `modeling/transformer/criterion.py`: Loss 계산 로직
- `modeling/meta_arch/detr.py`: Weight dict 정의
- `engine/trainer.py`: 로깅 시스템
- `data/datasets/multi_dataset.py`: 멀티 데이터셋 loss 가중치
- `vis_train_log.py`: 로그 시각화 스크립트

