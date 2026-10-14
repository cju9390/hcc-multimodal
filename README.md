# HCC 멀티모달 프로젝트

TCGA-LIHC(간세포암, Hepatocellular Carcinoma) 데이터를 이용해 조직 영상(WSI) + 임상 데이터 + RNA-seq을
결합한 멀티모달 모델을 만드는 프로젝트. 이미지 한 장이 들어왔을 때 (1) stage(early/late) 분리,
(2) 대사(metabolic) 서브타입 예측을 목표로 한다.

## 왜 이 스코프인가

- **간암(HCC)을 선택한 이유**: 영상+유전체 멀티모달 결합 연구가 상대적으로 덜 다뤄진 영역이고, 배경 간질환
  (바이러스성/NAFLD 등)이 임상적으로 병인론 자체와 얽혀 있어 임상 모달리티가 장식이 아니라 실제 신호로 작동함.
- **"대사 서브타입"을 타겟으로 잡은 이유**: 단순 유전자 변이(CTNNB1 등) 예측 대신, RNA-seq 기반으로 정의되는
  대사 서브타입(iHCC1/2/3 등, 선행 연구에서 이미 조직학적 분화도·AFP·혈관침범과의 상관관계가 보고됨)을 타겟으로 잡아
  "형태학이 대사 상태를 반영하는가"를 검증하는 것 자체가 프로젝트의 기여점이 되도록 설계함.
- **면역치료 반응 예측은 포함하지 않음**: TCGA-LIHC 환자는 대부분 면역항암제 도입 이전 수술 코호트라 실제
  치료 반응 라벨이 없음. MSI/대사 서브타입은 어디까지나 반응 가능성을 논할 때 쓰는 동기 부여용 배경 지식일 뿐,
  학습 타겟으로 쓰지 않음.
- **stage는 4단계(I~IV) 대신 early(I/II)/late(III/IV) 이진 분류로 단순화**: TCGA-LIHC 표본 수(300~400명 규모)로는
  4단계 전부를 클래스로 두면 특히 stage IV 표본이 너무 적어짐. 선행 연구에서도 early/late 이진 분할(173/177)이
  표준적으로 쓰임.

## 데이터 소스

모두 [GDC(Genomic Data Commons)](https://portal.gdc.cancer.gov/) open-access API에서 인증 없이 받을 수 있음.

| 모달리티 | 데이터 | GDC data_type |
|---|---|---|
| 영상 | H&E 진단 슬라이드(SVS) | `Slide Image` (experimental_strategy=`Diagnostic Slide`) |
| 유전체 | RNA-seq 유전자 발현량 (STAR-Counts) | `Gene Expression Quantification` |
| 임상 | 나이/생존상태/AJCC stage/tumor grade 등 | GDC `cases` 엔드포인트 |

TCGA-LIHC 전체 중 WSI와 RNA-seq을 **둘 다 가진 환자는 359명**, 전체 원본 용량은 WSI 약 495.5GB + RNA-seq 약 1.7GB
(합계 약 497GB). 처음엔 30명 파일럿으로 파이프라인을 검증했고, 이후 [Ally et al., Cell 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5680778/)
Table S1A의 iCluster 라벨이 있는 **196명 core cohort 중 GDC에 WSI+RNA-seq이 둘 다 있는 185명**으로 확장함
(전체 359명 대신 이 185명만 받은 이유: 359명 전체는 약 497GB인데 185명으로 줄여도 약 283GB로 별로 안 줄어들고,
다운로드 속도가 초당 4~5MB 수준이라 시간이 오래 걸림 — 그래도 라벨 있는 케이스 전부를 포함하는 걸 우선시함).
데이터는 D:\hcc_multimodal 에 저장 (용량 문제로 C드라이브에서 이동함).

## 폴더 구조

```
hcc_multimodal/
├── README.md
├── requirements.txt
├── scripts/
│   ├── 01_build_manifest.py          # GDC에서 WSI+RNA-seq 둘 다 있는 케이스를 찾아 파일럿 매니페스트 생성
│   ├── 02_download_pilot.py          # 매니페스트 기준으로 WSI(svs)+RNA-seq(tsv) 실제 다운로드
│   ├── 03_fetch_clinical.py          # 매니페스트 케이스들의 임상 데이터(GDC cases API) 조회
│   ├── 04_build_metabolic_labels.py  # TCGA-LIHC 논문(Ally et al., Cell 2017) Table S1A에서
│   │                                  # iCluster(1/2/3) 대사 서브타입 라벨을 파일럿 케이스에 매칭
│   ├── 05_tile_wsi.py                # WSI 타일링 + 조직 필터링 + 염색 정규화 (Macenko, 고정 참조 행렬)
│   ├── 06_train_val_test_split.py    # 환자 단위 patient-level train/val/test 분할 (대사 서브타입 기준 stratified)
│   ├── model.py                      # Dataset(WSI bag) + ResNet18 인코더 + Attention MIL + 임상 fusion 멀티태스크 모델
│   └── 07_train_model.py             # 학습 루프 (stage/대사 서브타입 헤드, 라벨 없는 쪽은 마스킹)
└── data/
    ├── raw/
    │   ├── pilot_manifest.csv     # case_id, submitter_id, slide/rnaseq file_id·file_name
    │   ├── wsi/                   # .svs 원본 슬라이드
    │   ├── rnaseq/                # .rna_seq.augmented_star_gene_counts.tsv
    │   ├── external/
    │   │   └── NIHMS882929-supplement-8.xlsx  # Ally et al. 2017 Table S1A (iCluster 라벨 소스, 수동 다운로드)
    │   └── clinical/
    │       ├── pilot_clinical_raw.json
    │       ├── pilot_clinical.csv
    │       └── pilot_metabolic_labels.csv     # submitter_id별 iCluster/Hoshida/혈관침범/AFP/grade
    └── processed/
        ├── tiles/                 # <submitter_id>/<submitter_id>_x{x}_y{y}.jpg (256x256, 20x, 염색 정규화됨)
        ├── tile_manifest.csv      # submitter_id, tile_path, x, y, tissue_fraction
        ├── pilot_split.csv        # submitter_id, split(train/val/test), metabolic_subtype
        └── checkpoints/
            └── multitask_model.pt # 학습된 모델 가중치
```

## 실행 순서

```bash
pip install -r requirements.txt
# torch/torchvision은 위 명령으로 CPU 버전이 깔릴 수 있음 (Windows + Python 3.14는 PyPI 기본 인덱스에
# CUDA 빌드가 없음). GPU를 쓰려면 아래처럼 CUDA 빌드로 따로 설치할 것 (드라이버가 지원하는 CUDA 버전에 맞게 cu128 등 조정):
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

python scripts/01_build_manifest.py     # data/raw/pilot_manifest.csv 생성
python scripts/02_download_pilot.py     # data/raw/wsi, data/raw/rnaseq 채움 (수십 GB, 시간 소요)
python scripts/03_fetch_clinical.py     # data/raw/clinical 채움
python scripts/04_build_metabolic_labels.py  # data/raw/clinical/pilot_metabolic_labels.csv 생성
python scripts/05_tile_wsi.py --max-tiles 100  # data/processed/tiles, tile_manifest.csv 생성
python scripts/06_train_val_test_split.py      # data/processed/pilot_split.csv 생성
python scripts/07_train_model.py --epochs 5    # data/processed/checkpoints/multitask_model.pt 생성
```

`04_build_metabolic_labels.py`는 [Ally et al., Cell 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC5680778/)의 Supplementary Table S1A를 입력으로 쓴다.
이 파일은 PMC의 봇 방지 장치 때문에 스크립트로 자동 다운로드가 안 되므로, 브라우저로 직접 받아서
`data/raw/external/NIHMS882929-supplement-8.xlsx`에 넣어둔 뒤 스크립트를 실행해야 한다.

`05_tile_wsi.py`는 슬라이드당 최대 타일 수를 `--max-tiles`로, 처리할 슬라이드 수를 `--limit`으로 제한할 수 있다
(파일럿 검증 단계라 기본값 100타일/슬라이드로 제한해둠 — 전체 학습 시엔 늘려야 함). 염색 정규화는 슬라이드마다
고유벡터로 커스텀 염색 행렬을 추정하는 대신(스트로마 위주 패치에서 두 염색 벡터가 거의 겹쳐버려 특이행렬이 되고
재구성 시 흑백 얼룩으로 깨지는 문제가 있었음) 문헌 참조 H&E 행렬로 고정 디콘볼루션하고, 슬라이드별 염색 강도만
보정하는 방식으로 구현함.

`01_build_manifest.py`의 `PILOT_SIZE` 상수로 규모를 조절한다. Table S1A core cohort(iCluster 라벨 있는 케이스)를
항상 우선 포함하도록 정렬해두므로, `PILOT_SIZE`를 core cohort 크기(현재 185)보다 크게 잡으면 라벨 없는 케이스로
나머지를 채우고, 작게 잡으면 라벨 있는 케이스만 그만큼 받는다. 전체 359명으로 확장하려면 `PILOT_SIZE = 359`로
바꾸면 됨 (단, 다운로드 전 디스크 여유 공간이 약 500GB 이상인지 먼저 확인할 것).

## 현재 상태 (185명 — Table S1A core cohort)

- 매니페스트: `data/raw/pilot_manifest.csv` — 185 케이스, 전부 Primary Tumor. iCluster 라벨이 있는
  Table S1A core cohort(196명 중 GDC에 WSI+RNA-seq 둘 다 있는 185명) 전원을 포함하도록 `01_build_manifest.py`를
  수정함 (기존엔 케이스 UUID 정렬 순서로 그냥 앞에서부터 잘랐음).
- 임상 데이터: 185명분 확보. `ajcc_pathologic_stage` 등 GDC 필드 커버리지는 이전 30명 파일럿과 비슷한 패턴.
  **성별, 음주력, 흡연력은 여전히 GDC 하모나이즈드 clinical에 값이 없음** (레거시 TCGA 데이터라 일부 필드가
  BCR clinical_supplement XML에만 남아있고 `cases` API로는 안 나옴).
- WSI/RNA-seq: 다운로드 진행 중 (약 283GB, 초당 4~5MB 기준 예상 18시간 내외). GDC API에 파일 ID 수백 개를
  한 번에 조회할 때 GET 요청 URL이 너무 길어 414 에러가 나는 버그가 있었음 — `02_download_pilot.py`,
  `03_fetch_clinical.py` 둘 다 POST로 고침.
- 대사 서브타입 라벨: Table S1A에서 **175/185명 라벨 확보** (iCluster1 63 / iCluster2 51 / iCluster3 61).
  나머지 10명은 core cohort에는 속하지만 iCluster 컬럼 값 자체가 "NA"로 기재된 케이스라 라벨로 못 씀
  (바코드 존재 여부만으로 "라벨 있음" 판정하던 버그를 고쳐서 걸러냄 — `04_build_metabolic_labels.py`).
- WSI 타일링 / train-val-test 분할 / 모델 학습: **다운로드 완료 후 185명 기준으로 재실행 필요**
  (`05_tile_wsi.py` → `06_train_val_test_split.py` → `07_train_model.py`). 30명 파일럿 때 만든 타일·split·체크포인트는
  구버전이라 지금 매니페스트와 안 맞음.

## 다음 단계 (미착수)

1. 185명 다운로드 완료 대기 → `05_tile_wsi.py`(`--max-tiles`를 100보다 늘리는 것 고려) → `06_train_val_test_split.py`
   → `07_train_model.py` 순서로 재실행
2. 175명 라벨(+stage 라벨 있는 나머지 10명)로 학습하면 이제 val/test 지표가 어느 정도 의미를 가지기 시작함 —
   그 이후에 하이퍼파라미터 튜닝 시작
3. 필요시 MIL 어텐션 가중치를 시각화해서 모델이 실제 종양 영역에 주목하는지 정성적으로 점검
4. 전체 359명까지 더 확장할지는 185명 결과를 보고 판단 (용량·시간 대비 라벨 없는 174명을 추가로 받을 가치가 있는지)
