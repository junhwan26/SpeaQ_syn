# Label Distribution Analysis for SpeaQ

이 폴더는 SpeaQ 프로젝트에서 Real data (Visual Genome)와 Synthetic data의 label 분포를 비교 분석하는 도구들을 포함합니다.

## 파일 설명

- `label_distribution_comparison.py`: 메인 분석 스크립트
- `run_label_analysis.sh`: 간단한 실행 스크립트 (conda 환경 자동 활성화)
- `README.md`: 이 파일

## 사용법

### 1. 간단한 실행 (권장)

```bash
# conda 환경 활성화
conda activate speaq_analysis

# 분석 실행
cd /home/junhwanheo/SpeaQ/analysis
./run_label_analysis.sh
```

### 2. 커스텀 설정으로 실행

```bash
# conda 환경 활성화
conda activate speaq_analysis

# 커스텀 설정으로 실행
cd /home/junhwanheo/SpeaQ/analysis
./run_label_analysis.sh --config-file /path/to/your/config.yaml --output-dir custom_output
```

### 3. Python 스크립트 직접 실행

```bash
# conda 환경 활성화
conda activate speaq_analysis

# 분석 실행
python label_distribution_comparison.py --config-file /home/junhwanheo/SpeaQ/configs/speaq_multi_dataset.yaml --output-dir label_distribution_analysis
```

## 출력 결과

분석이 완료되면 지정된 출력 디렉토리에 다음 파일들이 생성됩니다:

### 시각화 파일
- `object_class_distribution.png`: Real/Synthetic 데이터의 상위 20개 객체 클래스 분포
- `predicate_distribution.png`: Real/Synthetic 데이터의 상위 20개 predicate 분포  
- `top_classes_comparison.png`: 공통 클래스들의 Real/Synthetic 비교
- `top_predicates_comparison.png`: 모든 predicate들의 Real/Synthetic 비교 (50개 전체)
- `distribution_statistics.png`: 분포 통계 및 다양성 비교

### 보고서 파일
- `label_distribution_report.txt`: 상세한 분석 결과 텍스트 보고서

## 분석 내용

### 1. 객체 클래스 분석
- Real/Synthetic 데이터의 객체 클래스 분포 비교
- 공통 클래스, Real-only 클래스, Synthetic-only 클래스 식별
- 빈도 차이가 큰 클래스들 식별

### 2. Predicate 분석  
- Real/Synthetic 데이터의 predicate 분포 비교
- 공통 predicate, Real-only predicate, Synthetic-only predicate 식별
- 빈도 차이가 큰 predicate들 식별

### 3. 통계적 분석
- Chi-square 검정을 통한 분포 차이 검증
- 분포 다양성 비교
- 빈도 분포 히스토그램

## 요구사항

- Python 3.7+
- conda 환경: speaq_analysis
- 필요한 패키지: numpy, matplotlib, seaborn, pandas, scipy, h5py, pyyaml

## 설정 파일

기본적으로 `/home/junhwanheo/SpeaQ/configs/speaq_multi_dataset.yaml` 파일을 사용합니다.
이 파일에는 다음 정보가 포함되어야 합니다:

```yaml
DATASETS:
  TYPE: "MULTI_DATASET"
  VISUAL_GENOME:
    MAPPING_DICTIONARY: "/path/to/VG-SGG-dicts-with-attri.json"
    VG_ATTRIBUTE_H5: "/path/to/VG-SGG-with-attri.h5"
  VISUAL_GENOME_SYNTHETIC:
    MAPPING_DICTIONARY: "/path/to/VG-SGG-dicts-with-attri.json"  
    VG_ATTRIBUTE_H5: "/path/to/VG-SGG-with-attri-refined-synthetic_only.h5"
```

## 문제 해결

### 1. conda 환경 활성화 실패
```bash
# conda 환경 확인
conda env list

# speaq_analysis 환경이 없다면 생성
conda create -n speaq_analysis python=3.8
conda activate speaq_analysis
pip install numpy matplotlib seaborn pandas scipy h5py pyyaml
```

### 2. 파일 경로 오류
- 설정 파일의 경로가 올바른지 확인
- H5 파일과 JSON 파일이 존재하는지 확인
- 파일 읽기 권한이 있는지 확인

### 3. 메모리 부족
- 큰 데이터셋의 경우 메모리 사용량이 클 수 있습니다
- 필요시 배치 처리로 코드를 수정하거나 더 큰 메모리를 가진 시스템에서 실행

## 예제 결과 해석

분석 결과를 통해 다음을 확인할 수 있습니다:

1. **분포 유사성**: Real/Synthetic 데이터의 label 분포가 얼마나 유사한지
2. **도메인 갭**: 두 데이터셋 간의 차이점과 잠재적 도메인 갭
3. **균형성**: 클래스 불균형 문제의 심각도
4. **커버리지**: Synthetic 데이터가 Real 데이터의 다양성을 얼마나 잘 커버하는지

이러한 분석 결과는 도메인 적응 전략 수립과 데이터 증강 전략 개선에 활용할 수 있습니다.
