# Real/Synthetic Loss 분리 로깅 기능

## 개요

Real 데이터와 Synthetic 데이터의 loss를 weighted sum 하기 전에 각각 분리하여 로그로 출력하는 기능이 추가되었습니다. 이를 통해 Real 데이터와 Synthetic 데이터 각각의 학습 진행 상황을 독립적으로 추적할 수 있습니다.

## 구현 내용

### 1. 추가된 Loss 메트릭

다음과 같은 loss 메트릭들이 로그에 추가로 출력됩니다:

- **Relation Loss**:
  - `loss_relation_real`: Real 샘플의 relation classification loss (weighted sum 전)
  - `loss_relation_synthetic`: Synthetic 샘플의 relation classification loss (weighted sum 전)

- **Bbox Relation Loss**:
  - `loss_bbox_relation_real`: Real 샘플의 relation bbox regression loss (weighted sum 전)
  - `loss_bbox_relation_synthetic`: Synthetic 샘플의 relation bbox regression loss (weighted sum 전)

- **GIoU Relation Loss**:
  - `loss_giou_relation_real`: Real 샘플의 relation GIoU loss (weighted sum 전)
  - `loss_giou_relation_synthetic`: Synthetic 샘플의 relation GIoU loss (weighted sum 전)

### 2. 구현 위치

**파일**: `modeling/transformer/criterion.py`

**함수**: `IterativeRelationCriterion.get_relation_loss()`

### 3. 동작 방식

#### 3.1 Real/Synthetic 샘플 구분

각 샘플의 `loss_weight`를 기반으로 Real과 Synthetic 샘플을 구분합니다:

```python
# Real 샘플: loss_weight >= 0.9 (typically 1.0)
# Synthetic 샘플: loss_weight < 0.9 (typically 0.5)
real_mask = sample_weights >= 0.9
synthetic_mask = sample_weights < 0.9
```

#### 3.2 Per-Sample Loss 계산

Weighted sum을 계산하기 전에, 각 샘플의 원본 loss를 개별적으로 계산합니다:

```python
# Per-sample loss 계산 (reduction='none')
losses_per_sample = F.cross_entropy(
    src_logits_matched, target_classes_matched, 
    self.empty_rel_weight, reduction='none'
)  # [num_matched]
```

#### 3.3 Real/Synthetic Loss 분리

계산된 per-sample loss를 Real과 Synthetic으로 분리하여 각각 평균을 계산합니다:

```python
# Real 샘플의 평균 loss
if real_mask.any():
    loss_ce_real_val = losses_per_sample[real_mask].mean()

# Synthetic 샘플의 평균 loss
if synthetic_mask.any():
    loss_ce_synthetic_val = losses_per_sample[synthetic_mask].mean()
```

#### 3.4 Weighted Sum 계산 (기존 로직 유지)

분리된 loss는 로깅 목적으로만 사용되며, 실제 학습에 사용되는 loss는 기존과 동일하게 weighted sum을 적용합니다:

```python
# Weighted sum (학습에 사용)
weighted_losses = losses_per_sample * sample_weights
loss_ce = weighted_losses.mean()  # 이것이 실제 학습에 사용되는 loss
```

#### 3.5 Loss Dictionary에 추가

분리된 loss를 losses dictionary에 추가하여 자동으로 로그에 출력되도록 합니다:

```python
losses = {'loss_relation': loss_ce}  # 기존 weighted loss

# 분리된 loss 추가 (로깅용)
if loss_ce_real_val is not None:
    losses['loss_relation_real'] = loss_ce_real_val
if loss_ce_synthetic_val is not None:
    losses['loss_relation_synthetic'] = loss_ce_synthetic_val
```

### 4. 로그 출력 예시

다음과 같은 형식으로 로그에 출력됩니다:

```
[11/07 14:28:51] d2.utils.events INFO:  iter: 79  
  total_loss: 321.3
  loss_relation: 6.085  # Weighted sum된 loss (학습에 사용)
  loss_relation_real: 6.2  # Real 샘플의 원본 loss (로깅용)
  loss_relation_synthetic: 5.8  # Synthetic 샘플의 원본 loss (로깅용)
  loss_bbox_relation: 2.163
  loss_bbox_relation_real: 2.2
  loss_bbox_relation_synthetic: 2.0
  loss_giou_relation: 2.302
  loss_giou_relation_real: 2.4
  loss_giou_relation_synthetic: 2.1
  ...
```

### 5. 사용 예시

#### 5.1 TensorBoard에서 확인

```bash
tensorboard --logdir outputs/{exp_name}
```

TensorBoard에서 `loss_relation_real`, `loss_relation_synthetic` 등의 메트릭을 확인할 수 있습니다.

#### 5.2 W&B에서 확인

W&B 대시보드에서도 자동으로 추적되며, Real과 Synthetic loss의 변화를 비교할 수 있습니다.

#### 5.3 JSON 로그 파일에서 확인

`outputs/{exp_name}/metrics.json` 파일에서도 각 iteration별로 분리된 loss 값을 확인할 수 있습니다.

### 6. 주의사항

1. **배치 구성**: 현재 배치에 Real 샘플만 있거나 Synthetic 샘플만 있는 경우, 해당하는 loss만 출력됩니다.

2. **Weighted Sum과의 차이**: 
   - `loss_relation_real`과 `loss_relation_synthetic`는 **weighted sum 전의 원본 loss**입니다.
   - 실제 학습에 사용되는 loss는 여전히 `loss_relation` (weighted sum된 값)입니다.

3. **NaN/Inf 처리**: 안전성을 위해 NaN/Inf 값이 있는 경우 분리된 loss는 출력되지 않습니다.

### 7. 코드 변경 사항 요약

**주요 변경 사항**:
- `get_relation_loss()` 함수에서 real/synthetic loss 분리 로직 추가
- `loss_relation_real`, `loss_relation_synthetic` 추가
- `loss_bbox_relation_real`, `loss_bbox_relation_synthetic` 추가
- `loss_giou_relation_real`, `loss_giou_relation_synthetic` 추가

**기존 기능 유지**:
- Weighted sum 계산 로직은 그대로 유지
- 실제 학습에 사용되는 loss는 변경 없음
- 기존 로깅 기능과 호환

### 8. 활용 방법

이 기능을 활용하여:

1. **Real/Synthetic 데이터의 학습 상태 비교**: 두 데이터셋의 loss 변화를 독립적으로 추적
2. **하이퍼파라미터 튜닝**: Real과 Synthetic loss의 균형을 조정하는데 도움
3. **데이터셋 품질 평가**: Synthetic 데이터가 모델 학습에 얼마나 기여하는지 평가
4. **디버깅**: Real/Synthetic 데이터 중 어느 쪽에서 문제가 발생하는지 파악

---

**참고**: 이 기능은 multi-dataset training (`MULTI_DATASET` 타입)에서만 동작합니다. Single dataset training에서는 `loss_weight`가 없으므로 분리된 loss가 출력되지 않습니다.

