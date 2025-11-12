# Real/Synthetic Weighted Sum 상세 설명

## 목차
1. [개요](#개요)
2. [전체 플로우](#전체-플로우)
3. [단계별 상세 설명](#단계별-상세-설명)
4. [수학적 표현](#수학적-표현)
5. [구현 코드 분석](#구현-코드-분석)
6. [예시](#예시)

---

## 개요

멀티 데이터셋 학습 시, Real 데이터셋과 Synthetic 데이터셋의 샘플에 서로 다른 loss 가중치를 적용하여 weighted sum을 계산합니다.

**목적**:
- Real 데이터셋의 영향력을 더 크게 하기 위해 (일반적으로 `loss_weight = 1.0`)
- Synthetic 데이터셋의 영향력을 조절하기 위해 (일반적으로 `loss_weight = 0.5`)

**핵심 아이디어**:
- 각 샘플별로 loss를 계산한 후, 해당 샘플의 `loss_weight`를 곱하여 가중치 적용
- 가중치가 적용된 loss들의 평균을 최종 loss로 사용

---

## 전체 플로우

```
배치 구성 (MultiDatasetDynamicSampler)
    ↓
각 샘플에 loss_weight 추가
    ├─→ Real 샘플: loss_weight = 1.0
    └─→ Synthetic 샘플: loss_weight = 0.5
    ↓
모델 Forward Pass
    ↓
Criterion.forward()
    ├─→ Targets에서 loss_weight 추출
    ├─→ loss_weights = [1.0, 1.0, 0.5, 0.5, ...] (배치별)
    └─→ 각 loss 함수에 loss_weights 전달
    ↓
각 Loss 함수 (예: get_relation_loss)
    ├─→ 매칭된 샘플들의 batch_indices 추출
    ├─→ sample_weights = loss_weights[batch_indices]
    ├─→ Per-sample loss 계산 (reduction='none')
    ├─→ weighted_losses = losses_per_sample * sample_weights
    └─→ loss = weighted_losses.mean()
    ↓
최종 Loss Dictionary 반환
```

---

## 단계별 상세 설명

### 1. 배치 구성 및 loss_weight 추가

**파일**: `data/datasets/multi_dataset.py::MultiDatasetDynamicSampler.__getitem__()`

```python
def __getitem__(self, idx):
    batch_idx = idx // self.batch_size
    item_idx = idx % self.batch_size
    
    if item_idx < self.real_samples_per_batch:
        # Real 데이터셋 샘플
        real_idx = np.random.randint(0, self.real_size)
        item = copy.deepcopy(self.real_dicts[real_idx])
        item['dataset_type'] = 'real'
        item['loss_weight'] = self.real_loss_weight  # 1.0
    else:
        # Synthetic 데이터셋 샘플
        synthetic_idx = np.random.randint(0, self.synthetic_size)
        item = copy.deepcopy(self.synthetic_dicts[synthetic_idx])
        item['dataset_type'] = 'synthetic'
        item['loss_weight'] = self.synthetic_loss_weight  # 0.5
    
    return item
```

**예시**:
- 배치 크기: 20
- Real sampling ratio: 0.7 → Real 14개, Synthetic 6개
- 배치 구성: `[real, real, ..., real (14개), synthetic, synthetic, ..., synthetic (6개)]`
- loss_weight: `[1.0, 1.0, ..., 1.0 (14개), 0.5, 0.5, ..., 0.5 (6개)]`

### 2. Loss Weights 추출

**파일**: `modeling/transformer/criterion.py::IterativeRelationCriterion.forward()`

```python
def forward(self, outputs, targets):
    device = next(iter(outputs.values())).device
    losses = {}
    
    # 1. Targets에서 loss_weight 추출
    loss_weights = []
    for target in targets:
        if 'loss_weight' in target:
            loss_weights.append(target['loss_weight'])
        else:
            loss_weights.append(1.0)  # 기본값 (멀티 데이터셋이 아닌 경우)
    
    # 2. Tensor로 변환
    loss_weights = torch.tensor(loss_weights, device=device, dtype=torch.float32)
    # 예: loss_weights = tensor([1.0, 1.0, 1.0, ..., 0.5, 0.5, ...], device='cuda:0')
    
    # 3. 각 loss 함수에 loss_weights 전달
    kwargs = {'aux_loss': False, 'loss_weights': loss_weights}
    losses.update(self.get_relation_losses(..., **kwargs))
    
    return losses
```

**특징**:
- `loss_weights`는 배치 내 각 샘플의 가중치를 나타내는 1D tensor
- Shape: `[batch_size]`
- Real 샘플: 1.0, Synthetic 샘플: 0.5

### 3. Loss 계산 시 Weighted Sum 적용

#### 3.1 관계 분류 Loss (Cross Entropy)

**파일**: `modeling/transformer/criterion.py::get_relation_loss()`

```python
def get_relation_loss(self, outputs, targets, indices, num_relation_boxes, **kwargs):
    src_logits = outputs['relation_logits']  # [batch, num_queries, num_classes]
    idx = self._get_src_permutation_idx_rel(indices)
    # idx = (batch_indices, query_indices)
    # batch_indices: 매칭된 샘플들의 배치 인덱스 [0, 0, 1, 1, 2, ...]
    # query_indices: 매칭된 쿼리 인덱스
    
    # Loss weights 가져오기
    loss_weights = kwargs.get('loss_weights', None)
    
    # 기본 Cross Entropy Loss (가중치 없이)
    target_classes = torch.full(src_logits.shape[:2], self.num_rel_classes, ...)
    target_classes[idx] = target_classes_o
    loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_rel_weight)
    
    # Weighted Sum 적용
    if loss_weights is not None and len(idx[0]) > 0:
        # 1. 매칭된 샘플들의 배치 인덱스 추출
        batch_indices = idx[0]  # 예: [0, 0, 1, 1, 2, 2, 3, ...]
        
        # 2. 해당 샘플들의 loss_weight 가져오기
        sample_weights = loss_weights[batch_indices]
        # 예: loss_weights = [1.0, 1.0, 0.5, 0.5, ...]
        #     batch_indices = [0, 0, 1, 1, 2, 2, ...]
        #     sample_weights = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5, ...]
        
        # 3. Per-sample loss 계산 (reduction='none')
        src_logits_matched = src_logits[idx]  # [num_matched, num_classes]
        target_classes_matched = target_classes_o  # [num_matched]
        
        losses_per_sample = F.cross_entropy(
            src_logits_matched, 
            target_classes_matched, 
            self.empty_rel_weight, 
            reduction='none'  # 샘플별 loss 반환
        )  # Shape: [num_matched]
        # 예: losses_per_sample = [0.8, 1.2, 0.9, 1.1, 0.7, 0.6, ...]
        
        # 4. 각 샘플의 loss에 해당 샘플의 weight 곱하기
        weighted_losses = losses_per_sample * sample_weights
        # 예: weighted_losses = [0.8*1.0, 1.2*1.0, 0.9*1.0, 1.1*1.0, 0.7*0.5, 0.6*0.5, ...]
        #              = [0.8, 1.2, 0.9, 1.1, 0.35, 0.3, ...]
        
        # 5. 가중치가 적용된 loss들의 평균 계산
        loss_ce = weighted_losses.mean()
        # loss_ce = (0.8 + 1.2 + 0.9 + 1.1 + 0.35 + 0.3 + ...) / num_matched
    
    losses = {'loss_relation': loss_ce}
    return losses
```

#### 3.2 박스 회귀 Loss (L1)

```python
if len(idx[0]) > 0:
    src_boxes = outputs['relation_boxes'][idx]  # [num_matched, 4]
    target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
    
    # L1 Loss 계산 (reduction='none')
    loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
    # Shape: [num_matched, 4]
    
    # Weighted Sum 적용
    if loss_weights is not None and len(idx[0]) > 0:
        batch_indices = idx[0]
        sample_weights = loss_weights[batch_indices].unsqueeze(-1)  
        # [num_matched] → [num_matched, 1] (브로드캐스팅을 위해)
        
        # 각 샘플의 loss에 weight 곱하기
        loss_bbox = loss_bbox * sample_weights
        # [num_matched, 4] * [num_matched, 1] → [num_matched, 4]
        
        # 가중치가 적용된 loss들의 합을 정규화
        losses['loss_bbox_relation'] = loss_bbox.sum() / num_relation_boxes
```

#### 3.3 GIoU Loss

```python
loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(...))
# Shape: [num_matched]

# Weighted Sum 적용
if loss_weights is not None and len(idx[0]) > 0:
    batch_indices = idx[0]
    sample_weights = loss_weights[batch_indices]
    
    # 각 샘플의 loss에 weight 곱하기
    loss_giou = loss_giou * sample_weights
    # [num_matched] * [num_matched] → [num_matched]
    
    # 가중치가 적용된 loss들의 합을 정규화
    losses['loss_giou_relation'] = loss_giou.sum() / num_relation_boxes
```

---

## 수학적 표현

### 일반적인 Loss 계산 (가중치 없음)

```
L = (1/N) * Σᵢ Lᵢ

여기서:
- N: 매칭된 샘플 수
- Lᵢ: i번째 샘플의 loss
```

### Weighted Sum Loss 계산

```
L_weighted = (1/N) * Σᵢ (wᵢ * Lᵢ)

여기서:
- N: 매칭된 샘플 수
- Lᵢ: i번째 샘플의 loss
- wᵢ: i번째 샘플의 loss_weight
  - Real 샘플: wᵢ = 1.0
  - Synthetic 샘플: wᵢ = 0.5
```

### 예시 계산

**배치 구성**:
- Real 샘플 3개, Synthetic 샘플 2개
- 매칭된 샘플: Real 2개, Synthetic 2개

**Loss 값**:
- Real 샘플 1: L₁ = 1.0
- Real 샘플 2: L₂ = 1.2
- Synthetic 샘플 1: L₃ = 0.8
- Synthetic 샘플 2: L₄ = 0.9

**가중치**:
- w₁ = 1.0 (Real)
- w₂ = 1.0 (Real)
- w₃ = 0.5 (Synthetic)
- w₄ = 0.5 (Synthetic)

**Weighted Sum**:
```
L_weighted = (1/4) * (w₁*L₁ + w₂*L₂ + w₃*L₃ + w₄*L₄)
           = (1/4) * (1.0*1.0 + 1.0*1.2 + 0.5*0.8 + 0.5*0.9)
           = (1/4) * (1.0 + 1.2 + 0.4 + 0.45)
           = (1/4) * 3.05
           = 0.7625
```

**일반 평균 (가중치 없음)**:
```
L_avg = (1/4) * (1.0 + 1.2 + 0.8 + 0.9)
      = (1/4) * 3.9
      = 0.975
```

**비교**:
- Weighted Sum: 0.7625
- 일반 평균: 0.975
- Synthetic 샘플의 영향력이 0.5로 줄어들어 전체 loss가 감소

---

## 구현 코드 분석

### 핵심 코드: 관계 분류 Loss

**파일**: `modeling/transformer/criterion.py::get_relation_loss()`

```python
# 1. Loss weights 가져오기
loss_weights = kwargs.get('loss_weights', None)
# loss_weights: [batch_size] tensor, 예: [1.0, 1.0, 0.5, 0.5, ...]

# 2. 매칭된 샘플들의 배치 인덱스
idx = self._get_src_permutation_idx_rel(indices)
# idx = (batch_indices, query_indices)
# batch_indices: [0, 0, 1, 1, 2, 2, 3, ...] (매칭된 샘플이 속한 배치 인덱스)

# 3. 해당 샘플들의 loss_weight 추출
if loss_weights is not None and len(idx[0]) > 0:
    batch_indices = idx[0]  # [0, 0, 1, 1, 2, 2, 3, ...]
    sample_weights = loss_weights[batch_indices]
    # loss_weights[batch_indices]는 인덱싱을 통해 해당 샘플들의 weight를 가져옴
    # 예: loss_weights = [1.0, 1.0, 0.5, 0.5]
    #     batch_indices = [0, 0, 1, 1, 2, 2]
    #     sample_weights = [1.0, 1.0, 1.0, 1.0, 0.5, 0.5]
    
    # 4. Per-sample loss 계산
    losses_per_sample = F.cross_entropy(
        src_logits_matched, 
        target_classes_matched, 
        self.empty_rel_weight, 
        reduction='none'  # 중요: 샘플별 loss 반환
    )  # Shape: [num_matched]
    
    # 5. Weighted Sum
    weighted_losses = losses_per_sample * sample_weights
    # Element-wise multiplication
    # [num_matched] * [num_matched] → [num_matched]
    
    # 6. 평균 계산
    loss_ce = weighted_losses.mean()
    # Σ(weighted_losses) / num_matched
```

### 핵심 코드: 박스 회귀 Loss

```python
# L1 Loss (reduction='none')
loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
# Shape: [num_matched, 4]

# Weight 적용
if loss_weights is not None and len(idx[0]) > 0:
    batch_indices = idx[0]
    sample_weights = loss_weights[batch_indices].unsqueeze(-1)
    # [num_matched] → [num_matched, 1] (브로드캐스팅)
    
    # Element-wise multiplication with broadcasting
    loss_bbox = loss_bbox * sample_weights
    # [num_matched, 4] * [num_matched, 1] → [num_matched, 4]
    # 각 샘플의 4개 좌표 모두에 동일한 weight 적용
    
    # Sum and normalize
    losses['loss_bbox_relation'] = loss_bbox.sum() / num_relation_boxes
```

---

## 예시

### 시나리오

**배치 구성**:
- 배치 크기: 10
- Real 샘플: 7개 (인덱스 0-6)
- Synthetic 샘플: 3개 (인덱스 7-9)
- loss_weights: `[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]`

**매칭 결과**:
- 배치 인덱스 0: 2개 관계 매칭
- 배치 인덱스 1: 1개 관계 매칭
- 배치 인덱스 2: 3개 관계 매칭
- 배치 인덱스 7: 1개 관계 매칭 (Synthetic)
- 배치 인덱스 8: 2개 관계 매칭 (Synthetic)

**batch_indices**: `[0, 0, 1, 2, 2, 2, 7, 8, 8]`

**sample_weights 계산**:
```python
loss_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
batch_indices = [0, 0, 1, 2, 2, 2, 7, 8, 8]
sample_weights = loss_weights[batch_indices]
                = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.5, 0.5]
```

**Loss 계산**:
```python
# Per-sample losses (예시 값)
losses_per_sample = [0.8, 1.2, 0.9, 1.1, 0.7, 0.6, 0.5, 0.4, 0.3]

# Weighted losses
weighted_losses = losses_per_sample * sample_weights
                = [0.8*1.0, 1.2*1.0, 0.9*1.0, 1.1*1.0, 0.7*1.0, 0.6*1.0, 0.5*0.5, 0.4*0.5, 0.3*0.5]
                = [0.8, 1.2, 0.9, 1.1, 0.7, 0.6, 0.25, 0.2, 0.15]

# Final loss
loss_ce = weighted_losses.mean()
       = (0.8 + 1.2 + 0.9 + 1.1 + 0.7 + 0.6 + 0.25 + 0.2 + 0.15) / 9
       = 5.8 / 9
       = 0.644
```

**일반 평균과 비교**:
```python
# 일반 평균 (가중치 없음)
loss_avg = losses_per_sample.mean()
        = (0.8 + 1.2 + 0.9 + 1.1 + 0.7 + 0.6 + 0.5 + 0.4 + 0.3) / 9
        = 6.5 / 9
        = 0.722
```

**차이**:
- Weighted Sum: 0.644
- 일반 평균: 0.722
- Synthetic 샘플의 영향력이 0.5로 줄어들어 전체 loss가 약 10.8% 감소

---

## 중요한 특징

### 1. 샘플별 가중치 적용

- 각 샘플의 loss에 해당 샘플의 `loss_weight`를 곱함
- Real 샘플: `loss_weight = 1.0` → 영향력 100%
- Synthetic 샘플: `loss_weight = 0.5` → 영향력 50%

### 2. Per-Sample Loss 계산

- `reduction='none'`을 사용하여 샘플별 loss를 먼저 계산
- 그 후 각 샘플의 loss에 weight를 곱함
- 마지막으로 평균 또는 합을 계산

### 3. 모든 Loss 타입에 적용

- Classification Loss (Cross Entropy)
- Box Regression Loss (L1)
- GIoU Loss
- 모든 Auxiliary Losses

### 4. 안전성 체크

```python
# NaN/Inf 체크
if (not torch.isnan(sample_weights).any() and 
    not torch.isinf(sample_weights).any() and 
    len(sample_weights) > 0 and
    sample_weights.min() > 0):  # 양수인지 확인
    # Weighted Sum 적용
else:
    # 기본 loss 사용
```

---

## 설정

**파일**: `configs/speaq_multi_dataset.yaml`

```yaml
DATASETS:
  MULTI_DATASET:
    REAL_LOSS_WEIGHT: 1.0      # Real 데이터셋 loss 가중치
    SYNTHETIC_LOSS_WEIGHT: 0.5 # Synthetic 데이터셋 loss 가중치
```

**효과**:
- `REAL_LOSS_WEIGHT = 1.0`: Real 데이터셋의 loss가 그대로 반영
- `SYNTHETIC_LOSS_WEIGHT = 0.5`: Synthetic 데이터셋의 loss가 절반으로 반영
- 결과적으로 Real 데이터셋의 영향력이 Synthetic보다 2배 큼

---

## 요약

1. **배치 구성**: Real/Synthetic 샘플이 배치에 포함되고, 각 샘플에 `loss_weight` 추가
2. **Loss Weights 추출**: Targets에서 `loss_weight`를 추출하여 tensor로 변환
3. **Per-Sample Loss**: 각 샘플의 loss를 개별적으로 계산 (`reduction='none'`)
4. **Weighted Sum**: 각 샘플의 loss에 해당 샘플의 `loss_weight`를 곱함
5. **최종 Loss**: 가중치가 적용된 loss들의 평균 또는 합을 계산

이를 통해 Real 데이터셋의 영향력을 Synthetic 데이터셋보다 크게 하여, 더 신뢰할 수 있는 Real 데이터에 더 집중할 수 있습니다.

