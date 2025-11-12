# 데이터셋 로딩 플로우 다이어그램

## 전체 플로우

```
┌─────────────────────────────────────────────────────────────────┐
│                    train_iterative_model.py                      │
│                         (메인 엔트리)                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                    setup(args)                                   │
│  - get_cfg()                                                    │
│  - add_dataset_config(cfg)                                      │
│  - add_scenegraph_config(cfg)                                   │
│  - cfg.merge_from_file(args.config_file)                        │
│  - register_datasets(cfg)  ← 데이터셋 등록                       │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              data/tools/utils.py::register_datasets()           │
│                                                                  │
│  ┌────────────────────────────────────────────────────────┐    │
│  │  if cfg.DATASETS.TYPE == 'VISUAL GENOME':             │    │
│  │      VisualGenomeTrainData(cfg, split='train')        │    │
│  │      VisualGenomeTrainData(cfg, split='val')          │    │
│  │      VisualGenomeTrainData(cfg, split='test')         │    │
│  │                                                        │    │
│  │  elif cfg.DATASETS.TYPE == 'SYNTHETIC GENOME':       │    │
│  │      SyntheticGenomeTrainData(cfg, split='train')    │    │
│  │      ...                                              │    │
│  │                                                        │    │
│  │  elif cfg.DATASETS.TYPE == 'MULTI_DATASET':          │    │
│  │      register_multi_datasets(cfg)                     │    │
│  └────────────────────────────────────────────────────────┘    │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  VISUAL GENOME       │   │  MULTI_DATASET       │
    │  또는 SYNTHETIC      │   │                      │
    └──────────────────────┘   └──────────────────────┘
```

## VisualGenomeTrainData 초기화 플로우

```
┌─────────────────────────────────────────────────────────────────┐
│         VisualGenomeTrainData.__init__(cfg, split)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  설정 로드            │   │  마스크 경로 설정     │
    │  - split              │   │  - TRAIN_MASKS       │
    │  - 필터링 옵션        │   │  - VAL_MASKS         │
    │  - 샘플링 옵션        │   │  - TEST_MASKS        │
    └──────────────────────┘   └──────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              _fetch_data_dict()                                  │
│                                                                  │
│  1. 캐시 파일명 생성 (설정 기반)                                 │
│     tmp/visual_genome_{split}_data_{options}_{hash}.pkl         │
│                                                                  │
│  2. 캐시 존재 여부 확인                                          │
│     ┌──────────────┐          ┌──────────────┐                 │
│     │  캐시 있음?  │──Yes──→  │  pickle.load │                 │
│     └──────────────┘          └──────────────┘                 │
│           │                                                      │
│          No                                                      │
│           │                                                      │
│           ▼                                                      │
│     ┌──────────────┐          ┌──────────────┐                 │
│     │ _process_data│──→       │ pickle.dump  │                 │
│     └──────────────┘          └──────────────┘                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              _process_data()                                     │
│                                                                  │
│  1. H5 파일 오픈                                                 │
│     h5py.File(VG_ATTRIBUTE_H5, 'r')                             │
│                                                                  │
│  2. 이미지 메타데이터 로드                                       │
│     json.load(IMAGE_DATA)                                        │
│                                                                  │
│  3. 손상된 이미지 제거                                           │
│     ['1592', '1722', '4616', '4617']                            │
│                                                                  │
│  4. 마스크 로드 (옵션)                                           │
│     pickle.load(TRAIN_MASKS)                                     │
│                                                                  │
│  5. 그래프 데이터 로드                                           │
│     _load_graphs()                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              _load_graphs()                                      │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  1. Split 필터링                                          │  │
│  │     split_mask = (data_split == split_flag)              │  │
│  │     split_mask &= (img_to_first_box >= 0)                │  │
│  │     if FILTER_EMPTY_RELATIONS:                           │  │
│  │         split_mask &= (img_to_first_rel >= 0)            │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  2. 전체 데이터 로드                                       │  │
│  │     all_labels = h5['labels'][:, 0]                       │  │
│  │     all_attributes = h5['attributes'][:, :]               │  │
│  │     all_boxes = h5['boxes_1024'][:]                       │  │
│  │     all_relations = h5['relationships'][:]                │  │
│  │     all_predicates = h5['predicates'][:, 0]               │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  3. 이미지별 반복 처리                                     │  │
│  │     for each image:                                       │  │
│  │       a) 이미지 메타데이터                                │  │
│  │       b) 박스 및 라벨 추출                                │  │
│  │       c) 관계 추출                                        │  │
│  │       d) 박스 형식 변환 (cx,cy,w,h → x1,y1,x2,y2)        │  │
│  │       e) 인덱스 변환 (1-indexed → 0-indexed)             │  │
│  │       f) 필터링 (FILTER_NON_OVERLAP 등)                  │  │
│  │       g) 객체 어노테이션 생성                             │  │
│  │       h) 세그멘테이션 마스크 추가 (옵션)                  │  │
│  │       i) record 생성                                      │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  4. dataset_dicts 리스트 반환                                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│              register_dataset()                                  │
│                                                                  │
│  1. DatasetCatalog.register('VG_{split}', lambda: dataset_dicts)│
│                                                                  │
│  2. 메타데이터 설정                                              │
│     - MAPPING_DICTIONARY 로드                                   │
│     - idx_to_classes, idx_to_predicates, idx_to_attributes 생성 │
│     - MetadataCatalog에 등록                                    │
│                                                                  │
│  3. 통계 정보 계산 (옵션)                                        │
│     get_statistics()                                             │
│       - fg_matrix: [num_objs, num_objs, num_rels]               │
│       - bg_matrix: [num_objs, num_objs]                         │
│       - fg_rel_count: [num_rels]                                │
│       - pred_dist: log probability distribution                 │
└─────────────────────────────────────────────────────────────────┘
```

## MultiDatasetTrainData 플로우

```
┌─────────────────────────────────────────────────────────────────┐
│         MultiDatasetTrainData.__init__(cfg, split)              │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 설정 로드                                                    │
│     - REAL_SAMPLING_RATIO                                       │
│     - SYNTHETIC_SAMPLING_RATIO                                  │
│     - REAL_LOSS_WEIGHT                                          │
│     - SYNTHETIC_LOSS_WEIGHT                                     │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  Real Dataset        │   │  Synthetic Dataset   │
    │  생성                │   │  생성                │
    │                      │   │                      │
    │  VisualGenomeTrain   │   │  VisualGenomeTrain   │
    │  Data(real_cfg)      │   │  Data(syn_cfg)       │
    └──────────────────────┘   └──────────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. MultiDatasetDynamicSampler 생성                              │
│                                                                  │
│     입력:                                                        │
│     - real_dataset.dataset_dicts                                │
│     - synthetic_dataset.dataset_dicts                           │
│     - real_ratio, synthetic_ratio                               │
│     - real_loss_weight, synthetic_loss_weight                   │
│     - batch_size                                                │
│                                                                  │
│     출력:                                                        │
│     - MultiDatasetDynamicSampler 인스턴스                       │
│       (Dataset 인터페이스 구현)                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. DatasetCatalog에 등록                                        │
│     DatasetCatalog.register('MULTI_{split}', lambda: sampler)   │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. 통계 정보 결합                                               │
│     get_statistics()                                             │
│       - real_stats = real_dataset.get_statistics()              │
│       - synthetic_stats = synthetic_dataset.get_statistics()    │
│       - combined_fg_rel_count = real + synthetic                │
│       - combined_fg_matrix = real + synthetic                   │
│       - combined_pred_dist = log(combined_fg_matrix / ...)      │
└─────────────────────────────────────────────────────────────────┘
```

## MultiDatasetDynamicSampler 동작

```
┌─────────────────────────────────────────────────────────────────┐
│         MultiDatasetDynamicSampler.__getitem__(idx)             │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│  1. 배치 정보 계산                                               │
│     batch_idx = idx // batch_size                               │
│     item_idx = idx % batch_size                                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                ┌────────────┴────────────┐
                │                         │
                ▼                         ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  item_idx <          │   │  item_idx >=         │
    │  real_samples_per_   │   │  real_samples_per_   │
    │  batch?              │   │  batch?              │
    │                      │   │                      │
    │        Yes           │   │         No           │
    └──────────────────────┘   └──────────────────────┘
                │                         │
                ▼                         ▼
    ┌──────────────────────┐   ┌──────────────────────┐
    │  Real Dataset에서    │   │  Synthetic Dataset   │
    │  샘플링              │   │ 에서 샘플링          │
    │                      │   │                      │
    │  real_idx = random() │   │  syn_idx = random()  │
    │  item = copy.deepcopy│   │  item = copy.deepcopy│
    │    (real_dicts[...]) │   │    (syn_dicts[...])  │
    │  item['dataset_type']│   │  item['dataset_type']│
    │    = 'real'          │   │    = 'synthetic'     │
    │  item['loss_weight'] │   │  item['loss_weight'] │
    │    = real_weight     │   │    = syn_weight      │
    └──────────────────────┘   └──────────────────────┘
                │                         │
                └────────────┬────────────┘
                             │
                             ▼
                        return item
```

## 학습 시 데이터 로딩 플로우

```
┌─────────────────────────────────────────────────────────────────┐
│         JointTransformerTrainer.build_train_loader()            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         build_detection_train_loader(cfg, mapper)               │
│                                                                  │
│  mapper = DetrDatasetMapper(cfg, is_train=True)                 │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         DataLoader 생성                                          │
│                                                                  │
│  DataLoader(                                                    │
│      dataset=DatasetCatalog.get('VG_train'),                   │
│      batch_sampler=...,                                         │
│      num_workers=cfg.DATALOADER.NUM_WORKERS,                    │
│      collate_fn=...,                                            │
│  )                                                               │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│         각 배치마다 반복                                         │
│                                                                  │
│  for batch in dataloader:                                       │
│      for item in batch:                                         │
│          item = DetrDatasetMapper(item)  ← 변환 적용            │
│              │                                                  │
│              ▼                                                  │
│      ┌──────────────────────────────────────┐                 │
│      │  1. 이미지 로드                       │                 │
│      │     image = read_image(file_name)    │                 │
│      │                                       │                 │
│      │  2. 이미지 변환                       │                 │
│      │     image, transforms = apply(...)    │                 │
│      │                                       │                 │
│      │  3. 어노테이션 변환                   │                 │
│      │     instances = annotations_to_      │                 │
│      │              instances(annos)        │                 │
│      │                                       │                 │
│      │  4. 관계 필터링 및 변환               │                 │
│      │     relations = filter_and_map(...)   │                 │
│      │                                       │                 │
│      │  5. 텐서 변환                         │                 │
│      │     item['image'] = tensor(image)    │                 │
│      │     item['instances'] = instances    │                 │
│      │     item['relations'] = tensor(...)  │                 │
│      └──────────────────────────────────────┘                 │
│              │                                                  │
│              ▼                                                  │
│          모델 입력 준비 완료                                    │
└─────────────────────────────────────────────────────────────────┘
```

## 데이터 구조 변환 과정

### H5 파일 → Detectron2 형식

```
H5 파일 구조:
┌─────────────────────────────────────────┐
│ 'split': [0, 0, 1, 2, ...]              │
│ 'img_to_first_box': [0, 5, 12, ...]     │
│ 'img_to_last_box': [4, 11, 18, ...]     │
│ 'img_to_first_rel': [0, 3, 8, ...]      │
│ 'img_to_last_rel': [2, 7, 15, ...]      │
│ 'boxes_1024': [[cx, cy, w, h], ...]     │
│ 'labels': [1, 2, 3, ...]  (1-indexed)   │
│ 'attributes': [[1,0,1], [0,1,0], ...]   │
│ 'relationships': [[0,1], [1,2], ...]    │
│ 'predicates': [1, 5, 3, ...]  (1-indexed)│
└─────────────────────────────────────────┘
                    │
                    ▼
            _load_graphs()
                    │
                    ▼
Detectron2 형식:
┌─────────────────────────────────────────┐
│ dataset_dicts = [                       │
│   {                                     │
│     'file_name': '/path/to/img1.jpg',   │
│     'image_id': 1,                      │
│     'height': 768,                      │
│     'width': 1024,                      │
│     'annotations': [                    │
│       {                                 │
│         'bbox': [x1, y1, x2, y2],      │
│         'bbox_mode': BoxMode.XYXY_ABS, │
│         'category_id': 0,  (0-indexed) │
│         'attribute': [1, 0, 1, ...],   │
│         'segmentation': [...]           │
│       },                                │
│       ...                               │
│     ],                                  │
│     'relations': np.array([             │
│       [0, 1, 4],  # sub, obj, pred     │
│       [1, 2, 2],                        │
│       ...                               │
│     ])                                  │
│   },                                    │
│   ...                                   │
│ ]                                       │
└─────────────────────────────────────────┘
```

### DatasetMapper 변환

```
Detectron2 형식 dict
        │
        ▼
┌─────────────────────────────────────────┐
│  1. 이미지 로드                          │
│     image = read_image(file_name)       │
│                                         │
│  2. 이미지 변환                          │
│     - Resize                            │
│     - RandomFlip (train only)           │
│     - Crop (optional)                   │
│                                         │
│  3. 어노테이션 변환                      │
│     - bbox 변환 (이미지 변환 적용)      │
│     - segmentation 변환                 │
│     - Instances 객체 생성               │
│                                         │
│  4. 관계 처리                            │
│     - 중복 관계 필터링                  │
│     - 빈 박스 제거 후 인덱스 재매핑     │
│                                         │
│  5. 텐서 변환                            │
│     - image → torch.Tensor              │
│     - relations → torch.Tensor          │
│     - instances (Detectron2 구조)       │
└─────────────────────────────────────────┘
        │
        ▼
모델 입력 형식
```

## 캐시 시스템

```
처음 실행:
┌─────────────────────────────────────────┐
│ _fetch_data_dict()                      │
│   │                                     │
│   ├─→ 캐시 파일명 생성                  │
│   │   visual_genome_train_data_...pkl  │
│   │                                     │
│   ├─→ 캐시 존재 확인                    │
│   │   os.path.isfile(fileName)         │
│   │   → False                           │
│   │                                     │
│   ├─→ _process_data() 실행              │
│   │   - H5 파일 읽기                    │
│   │   - 데이터 변환                     │
│   │   - dataset_dicts 생성              │
│   │                                     │
│   └─→ pickle.dump() 저장                │
│       캐시 파일 생성                    │
└─────────────────────────────────────────┘

두 번째 실행:
┌─────────────────────────────────────────┐
│ _fetch_data_dict()                      │
│   │                                     │
│   ├─→ 캐시 파일명 생성                  │
│   │                                     │
│   ├─→ 캐시 존재 확인                    │
│   │   os.path.isfile(fileName)         │
│   │   → True                            │
│   │                                     │
│   └─→ pickle.load() 로드                │
│       빠른 로딩                         │
└─────────────────────────────────────────┘

캐시 무효화 조건:
- H5 파일 경로 변경 (해시 변경)
- 필터링 옵션 변경
- 마스크 설정 변경
- 기타 설정 변경
```

## 주요 인덱싱 변환

```
1. 객체 클래스 인덱스
   H5: 1, 2, 3, ... (1-indexed)
   → Detectron2: 0, 1, 2, ... (0-indexed)
   변환: category_id = gt_classes[idx] - 1

2. 관계 타입 인덱스
   H5: 1, 2, 3, ... (1-indexed)
   → Detectron2: 0, 1, 2, ... (0-indexed)
   변환: predicate = predicates - 1

3. 객체 인덱스 (이미지 내)
   H5: 절대 인덱스 (전체 데이터셋 기준)
   → Detectron2: 상대 인덱스 (이미지 내)
   변환: objects = all_relations[...] - first_box_index

4. 박스 좌표 형식
   H5: (cx, cy, w, h) 중심점 + 크기
   → Detectron2: (x1, y1, x2, y2) 좌상단 + 우하단
   변환:
     x1 = cx - w/2
     y1 = cy - h/2
     x2 = cx + w/2
     y2 = cy + h/2
```

