# 프로젝트 A: CIFAR-100 이미지 분류와 지식 증류

PyTorch로 직접 구현한 ResNet-18 교사와 MobileNetV2 학생을 비교하고, 일반 KD와 DKD가 학생의 분류 정확도와 실제 추론 비용에 미치는 영향을 확인하는 학습 프로젝트.
CIFAR-10으로 학습 흐름을 점검한 뒤, CIFAR-100의 교사·학생·증류 실험과 정확도·효율 분석까지 진행했다.

[상세 실험 보고서](reports/analysis/cifar100/experiment_report.md) · [수치 CSV](reports/analysis/cifar100/experiment_summary.csv)

## 분류 결과

분류 대상과 평가 기준은 다음과 같습니다.

- 분류 대상: CIFAR-100 fine label 100개.
- Best checkpoint 선택: validation accuracy 기준.
- Best Val / Test: 서로 다른 분할의 정확도. Test 열은 해당 best checkpoint의 기존 평가 결과.

| 모델 / 학습 | Best Val (%) | Best epoch | Test (%) | Parameters |
|---|---:|---:|---:|---:|
| ResNet-18 / CE (Teacher) | 78.12 | 194 | 77.96 | 11,220,132 |
| MobileNetV2 / CE (Baseline) | 70.44 | 198 | 69.99 | 2,351,972 |
| MobileNetV2 / KD | 72.94 | 185 | 72.83 | 2,351,972 |
| MobileNetV2 / DKD | 74.44 | 194 | 73.91 | 2,351,972 |

- Baseline 대비 Test 정확도: KD **+2.84%p**, DKD **+3.92%p**.
- DKD: KD보다 **1.08%p 높고**, Teacher보다 **4.05%p 낮은** 결과.
- 학생의 파라미터: Teacher 대비 **79.04% 감소**. 학생 세 방법의 구조·파라미터 수는 동일.
- DKD의 클래스별 변화: Baseline 대비 83개 클래스에서 개선, 12개 클래스에서 악화. 전체 정확도가 높아져도 모든 클래스가 개선되는 것은 아니다.

![네 모델의 정확도 비교](reports/analysis/cifar100/figures/accuracy_comparison.png)

## 모델 크기와 추론 비용

| 모델 | 가중치 state_dict (MiB) | Conv/Linear MACs/장 | GPU B=1 평균 (ms) | GPU B=64 평균 (ms) | CPU B=1 평균 (ms) | CPU B=64 평균 (ms) |
|---|---:|---:|---:|---:|---:|---:|
| Teacher | 42.886 | 555,468,800 | 1.645 | 6.423 | 3.485 | 84.762 |
| Baseline | 9.201 | 24,576,512 | 2.897 | 3.583 | 3.245 | 14.119 |
| KD | 9.201 | 24,576,512 | 3.033 | 3.568 | 3.143 | 13.025 |
| DKD | 9.201 | 24,576,512 | 2.984 | 3.539 | 3.246 | 11.793 |

표를 읽을 때 구분할 점:

- 지연시간: **배치 전체의 forward 시간**. GPU B=1에서는 DKD 학생이 Teacher보다 느렸고, B=64에서는 더 빨랐다.
- 파라미터·MACs 감소와 실제 지연시간 감소는 별개. 모든 배치에서 가속을 보장하지 않는다.
- 같은 구조인 학생 세 모델의 작은 측정 차이를 KD/DKD의 가속 효과로 해석하지 않는다.

벤치마크 조건과 측정 범위:

- 장치·입력: RTX 5080 Laptop GPU, FP32 입력 [B,3,32,32], eval/inference mode.
- CPU: threads 24, warmup 5회, B=1 50회/B=64 10회.
- GPU: warmup 30회 후 CUDA event로 각 200회 측정.
- 측정 제외: 전처리·데이터 로딩·CPU→GPU 복사. P95·환산 처리량·메모리와 측정 한계는 상세 보고서 참고.
- MACs: Conv2d/Linear만 집계하며 depthwise groups 반영. FLOPs=2×MACs는 해당 연산 범위의 근사.
- 가중치 크기: state_dict 직렬화 크기. optimizer/RNG 등이 포함된 best.pt 전체 파일 크기와 구분.

[모델 크기·연산량 그림](reports/analysis/cifar100/figures/model_efficiency.png) · [지연시간 그림](reports/analysis/cifar100/figures/latency_comparison.png)

## 데이터와 학습 구성

- 데이터: CIFAR-100 RGB 32×32, 공식 train에서 학습 45,000장/검증 5,000장, 공식 test 10,000장.
- 분할: stratified split, random_state=42, 클래스별 학습 450장/검증 50장. 저장 인덱스의 중복·교집합 없음 확인.
- 정규화: 학습 45,000장만 사용해 채널별 mean/std 계산.
- 학습 변환: RandomCrop(32,padding=4) → RandomHorizontalFlip → RandAugment(2,9) → ToTensor → Normalize.
- MixUp: 정규화된 학습 배치에 alpha=0.2 적용. 검증/test는 ToTensor → Normalize만 사용.
- 구조: ResNet-18은 3×3 stride1 stem과 [2,2,2,2] 블록, 초기 MaxPool 없음. MobileNetV2도 stride1 stem이며 마지막 채널은 1280.

| 설정 | Teacher / Baseline / KD / DKD |
|---|---|
| Epoch / batch / workers / seed | 200 / 64 / 0 / 42 |
| Optimizer | SGD, lr=0.01, momentum=0.9, weight_decay=5e-4 |
| Scheduler | CosineAnnealingLR, T_max=200, eta_min=1e-5 |
| CE | MixUp의 두 정답에 대한 CE 가중합 |
| KD | T=4, alpha=0.5; (1−alpha)CE + alpha·T²KL(teacher ∥ student) |
| DKD | T=4, alpha=1, beta=8; CE + min(epoch/20,1)·(alpha·TCKD + beta·NCKD) |

증류 구현과 초기화에서 확인한 사항:

- KD/DKD 입력: 같은 MixUp 입력을 Teacher와 Student에 전달. Teacher는 eval/no_grad 및 requires_grad=False로 고정.
- DKD 변형: MixUp의 두 target 각각에 대해 손실을 계산한 뒤 혼합. 원 논문 전체 설정을 재현한 실험은 아니다.
- 학생 초기화: 생성 직전 seed를 재설정하면 현재 코드·환경에서 초기 state_dict가 같음을 확인.

[학생 학습 곡선](reports/analysis/cifar100/figures/training_curves_comparison.png) · [클래스별 변화](reports/analysis/cifar100/figures/class_accuracy_comparison.png) · [실제 오류 사례](reports/analysis/cifar100/figures/error_analysis_examples.png)

## 디렉토리 구조

```text
basic_image_classification_prac/
├── configs/                 # 실험 설정용 위치
├── data/
│   ├── raw/                 # 다운로드한 원본 데이터
│   ├── splits/              # 고정한 train/val 인덱스
│   └── stats/               # 학습 분할에서 계산한 정규화 통계
├── src/                     # 여러 실행에서 재사용할 구현
│   ├── data/                # 데이터셋, 전처리, 데이터 로더
│   ├── models/              # 직접 구현한 ResNet, MobileNetV2
│   ├── losses/              # KD/DKD 손실
│   ├── training/            # 학습·검증 루프
│   ├── evaluation/          # 분류 평가, 추론 비용 측정
│   └── utils/               # seed, 체크포인트, 학습 곡선 등
├── scripts/                 # 환경 점검, 데이터 준비, 학습 실행 시작점
├── runs/                    # 실험별 체크포인트와 학습 곡선
├── reports/
│   ├── raw/cifar100/         # 기존 평가·벤치마크 원본: 로컬 자료
│   └── analysis/cifar100/    # 보고서·종합 CSV·분석 그림
├── .gitignore
└── README.md
```

폴더별 사용 방식:

- scripts/: src/의 공통 기능을 불러오는 실행 시작점.
- runs/<실험 ID>/: 가중치와 학습 곡선 보관. 전체 설정은 현재 학습 스크립트 참고.

## 환경과 실행 전 준비

확인한 실행 환경: Windows 11, Python 3.13.16, NVIDIA GeForce RTX 5080 Laptop GPU.

| 패키지 | 확인한 버전 |
|---|---|
| PyTorch / torchvision | 2.14.1+cu130 / 0.29.1+cu130 |
| NumPy / scikit-learn | 2.5.2 / 1.9.1 |
| Matplotlib / Pillow | 3.11.2 / 12.3.0 |

- CUDA 빌드 정보: 13.0.
- 환경 준비: GPU용 PyTorch/torchvision과 위 패키지를 갖춘 별도 환경 필요.

**아래 명령의 실행 위치는 별도 clone/복사본의 루트. 보존 중인 원본 저장소에서는 실행하지 않는다.**

단계별로 생성하거나 덮어쓰는 자료:

- 데이터 준비: splits/stats.
- 학습: 같은 run 이름의 체크포인트.
- 평가·벤치마크: reports/raw 출력.

다시 평가할 때는 원본을 보존한 채 별도 복사본에 데이터·분할·통계와 필요한 best.pt를 같은 상대경로로 준비한다.
처음부터 학습할 때는 기존 runs가 없는 별도 복사본에서 아래 순서로 진행한다.

가상환경을 준비한 뒤 실행할 PowerShell 예시:

```powershell
$python = (Resolve-Path ./.venv/Scripts/python.exe).Path
& $python -B -m scripts.check_environment
& $python -B -m scripts.inspect_cifar100
```

- inspect_cifar100: torchvision으로 공식 CIFAR-100을 다운로드하고 seed42 분할과 학습 통계 저장.
- 학습/평가 loader: download=False이므로 위 준비 단계 또는 동일한 로컬 자료 필요.

## 학습 명령

Teacher를 실행할 때 먼저 확인할 설정:

- 현재 기본값: load_file_name="best.pt".
- 다음 명령의 동작: **기존 Teacher best.pt에서 재개**. 새 clone에 checkpoint가 없으면 실패.

```powershell
& $python -B -m scripts.train_cifar100_teacher
```

새 Teacher 학습은 아래처럼 해당 한 줄을 **메모리에서만** 바꿔 실행한다. 소스 파일을 편집하지 않는 방식이다.
이 명령도 실제 학습을 시작하므로 기존 결과가 없는 별도 복사본에서 실행한다.

```powershell
@'
from pathlib import Path

path = Path("scripts/train_cifar100_teacher.py").resolve()
source = path.read_text(encoding="utf-8")
old = '    load_file_name = "best.pt"'
new = '    load_file_name = ""'
if source.count(old) != 1:
    raise RuntimeError("Teacher 재개 설정이 예상과 다릅니다. 소스를 확인하세요.")
source = source.replace(old, new, 1)
exec(compile(source, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})
'@ | & $python -B -
```

학생 세 스크립트의 기본값은 새 학습. Teacher 학습 완료 후 실행하며, 기존 결과가 있는 원본 저장소에서는 실행하지 않는다.

```powershell
& $python -B -m scripts.train_cifar100_student
& $python -B -m scripts.train_cifar100_kd
& $python -B -m scripts.train_cifar100_dkd
```

평가 대상은 각 run 아래의 best.pt. 실험 이름:

```text
Teacher:  cifar100_ResNet18_teacher_200_cosine_randaug_mixup0.2
Baseline: cifar100_MobileNetV2_student_200_cosine_randaug_mixup0.2
KD:       cifar100_MobileNetV2_kd_200_cosine_randaug_mixup0.2_T4.0_kd_a0.5
DKD:      cifar100_MobileNetV2_kd_200_cosine_randaug_mixup0.2_T4.0_dkd_a1.0_b8.0_warmup20
```

평가 스크립트의 EXPERIMENTS에 위 경로가 정의되어 있다. best.pt는 모델 외 optimizer/scheduler/RNG 상태도 포함한다.

## 평가와 벤치마크

실행 전제와 이번 문서화의 범위:

- 실행 위치: 네 checkpoint와 데이터가 준비된 별도 복사본의 루트.
- 이번 작업: 아래 전체 실행을 다시 수행하지 않음.

```powershell
& $python -B -m src.evaluation.cifar100_evaluate
& $python -B -m src.evaluation.cifar100_benchmark
```