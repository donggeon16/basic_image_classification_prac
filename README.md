# 프로젝트 A: 이미지 분류와 지식 증류

직접 구현하며 채워 나갈 프로젝트의 기본 디렉토리 구조입니다.

- A0: CIFAR-10으로 데이터 처리와 학습 흐름 검증
- A1: CIFAR-100에서 ResNet 교사, MobileNetV2 학생, 지식 증류 비교
- A2: 분류 정확도와 추론 비용 비교

```text
basic_image_classification_prac/
├── configs/                 # 실험별 설정
├── data/
│   ├── raw/                 # 다운로드한 원본 데이터
│   ├── splits/              # 고정한 train/val 인덱스
│   └── stats/               # 학습 분할에서 계산한 정규화 통계
├── src/                     # 여러 실행에서 재사용할 구현
│   ├── data/                # 데이터셋, 전처리, 데이터 로더
│   ├── models/              # 직접 구현할 ResNet, MobileNetV2
│   ├── losses/              # 학습 손실 구성, 지식 증류 손실
│   ├── training/            # 학습 루프
│   ├── evaluation/          # 분류 평가, 추론 비용 측정
│   └── utils/               # seed, 체크포인트, 로그 등 공통 기능
├── scripts/                 # 환경 점검, 데이터 준비, 학습·평가 실행 시작점
├── runs/                    # 실행별 로그, 체크포인트, 설정·환경 사본
├── reports/                 # 학습 곡선, 비교표, 오분류 사례와 설명
├── .gitignore
└── README.md
```

`scripts/`에서 `src/`의 기능을 불러와 실행하는 방식으로 구성하면 됩니다. A0·A1·A2마다 소스 폴더를 따로 복사하지 않고, 공통 코드를 사용합니다.

실험을 시작하면 `runs/<실험 ID>/` 아래에 해당 실행의 기록을 모읍니다. 설정, seed, 데이터 분할 식별 정보, 코드 버전, 환경 정보, 로그와 가중치를 연결해 보관합니다. 최종 비교 결과는 `reports/`에 정리합니다.

`data/raw/`와 `runs/`의 내용은 Git에서 제외합니다. 재현에 필요한 `configs/`, `data/splits/`, `data/stats/`와 정리한 보고서는 Git으로 관리합니다.

각 폴더의 `.gitkeep`은 빈 디렉토리를 Git에 남기기 위한 파일이며 실행에는 사용하지 않습니다. 구현 파일은 필요한 단계에서 직접 추가합니다.
