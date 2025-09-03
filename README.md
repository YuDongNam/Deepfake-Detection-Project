# Deepfake Video Detection using Vision Transformers and CNN‑LSTM

PyTorch 기반의 딥페이크 영상 탐지 프로젝트입니다. **Video Swin Transformer**와 **XceptionNet + LSTM** 두 가지 모델 아키텍처를 동일한 데이터 분할·전처리·평가 설정에서 비교했습니다.

> **요약**: 단일 얼굴 영상으로 학습/검증하고, 다중 얼굴 영상으로 테스트하여 일반화 성능을 점검합니다. 하나의 샘플은 **연속된 72개 프레임**으로 구성되어 시간적 특징을 학습합니다.

---

## 주요 특징 (Key Features)

* **두 가지 모델 아키텍처**

  * Video Swin Transformer (비디오용 비전 트랜스포머 백본)
  * XceptionNet + LSTM (프레임 수준 특징 추출 + 시계열 모델링)
* **데이터 분할**: 단일 얼굴(Train/Val) ↔ 다중 얼굴(Test)
* **프레임 샘플링**: 각 클립당 연속 72프레임 입력
* **일관된 평가 절차**: Accuracy, F1-Score, AUC 보고

---

## 주요 성능 (Performance)

*다중 얼굴 영상 테스트 데이터셋 기준.*

| Model                  | Accuracy | F1-Score |   AUC  |
| ---------------------- | :------: | :------: | :----: |
| Video Swin Transformer |  0.9374  |  0.9650  | 0.8359 |
| XceptionNet + LSTM     |  0.9374  |  0.9648  | 0.8511 |

---

## 요구 사항 (Requirements)

* Python ≥ 3.9
* PyTorch 및 TorchVision (CUDA 사용 권장)
* ffmpeg (비디오 전처리/프레임 추출 시)
* 기타 패키지는 `requirements.txt` 참고

---

## 설치 (Installation)

### 1) 레포지토리 클론

```bash
git clone https://github.com/yourusername/deepfake-detection.git
cd deepfake-detection
```

### 2) 가상환경 생성 및 활성화

```bash
python -m venv venv
# macOS/Linux
source venv/bin/activate
# Windows (PowerShell)
# .\venv\Scripts\Activate.ps1
```

### 3) 패키지 설치

```bash
pip install -r requirements.txt
```

---

## 데이터셋 준비 (Datasets)

본 프로젝트는 프레임 기반 로더를 가정합니다. 원본 비디오를 프레임으로 추출한 뒤 아래와 같이 배치하세요.

> 예시: FaceForensics++(FF++) 또는 Google DeepFake Detection(DFD) 등 **외부 데이터셋은 별도로 약관에 따라 다운로드** 하신 후, 프레임을 추출해 디렉토리를 구성하세요. 실제 경로/이름은 자유롭게 조정 가능하며 `config.py` 또는 CLI 인자에서 지정합니다.

```
data/
├── original_sequences/
│   └── (원본 영상 프레임 폴더들)
└── manipulated_sequences/
    └── (조작 영상 프레임 폴더들)
```

---

## 사용법 (Usage)

모든 기능은 `main.py`로 실행합니다. 기본 설정은 `config.py`에 있으며, **CLI 인자가 우선** 적용됩니다.

### 1) 모델 학습 (Training)

**Video Swin Transformer**

```bash
python main.py train --model swin --data_dir ./data --epochs 30 --batch_size 4
```

**XceptionNet + LSTM**

```bash
python main.py train --model xception --data_dir ./data --epochs 40 --batch_size 2
```

### 2) 모델 평가 (Evaluation)

```bash
python main.py evaluate \
  --model swin \
  --model_path ./saved_models/best_model.pth \
  --data_dir ./data
```

### 3) 개별 모델 실험 (Individual Model Experiments)

**Video Swin Transformer 전용 실험**
```bash
python experiments/swin_experiment.py
```

**XceptionNet+LSTM 전용 실험**
```bash
python experiments/xception_experiment.py
```

> **참고**: 개별 실험 스크립트는 각 모델에 최적화된 하이퍼파라미터와 설정을 사용합니다.

---

## 프로젝트 구조 (Project Structure)

```
deepfake-detection/
├── README.md
├── requirements.txt
├── config.py          # 공통 설정 (경로, 하이퍼파라미터 등)
├── main.py            # 메인 엔트리 (train/evaluate/추가 서브커맨드)
├── src/
│   ├── data_loader.py # 프레임 로딩 및 전처리/샘플링(72프레임 등)
│   ├── models/        # 모듈화된 모델 구조
│   │   ├── __init__.py
│   │   ├── base.py    # 공통 기능 및 기본 클래스
│   │   ├── swin_transformer.py  # Video Swin Transformer 모델
│   │   └── xception_lstm.py     # XceptionNet+LSTM 모델
│   ├── train.py       # 학습 루프, 로그/체크포인트 저장
│   └── evaluate.py    # 검증/테스트 및 지표 계산
├── experiments/       # 개별 모델 실험 스크립트
│   ├── __init__.py
│   ├── swin_experiment.py      # Swin 전용 실험
│   └── xception_experiment.py  # Xception 전용 실험
├── saved_models/      # 체크포인트(.pth)
└── data/              # 데이터셋 (사용자 준비)
```

---

## 평가 지표 (Metrics)

* **Accuracy**: 전체 정답 비율
* **F1-Score**: 정밀도/재현율의 조화 평균 (불균형 데이터에 유용)
* **AUC (ROC-AUC)**: 임계값 전 범위에서 분류 성능 요약

---

## 설계 노트 (Design Notes)

* **72프레임 연속 샘플링**: 프레임 간 상관관계를 모델이 학습하도록 강제하여 영상 기반 위변조의 시계열 패턴(예: 미세한 깜빡임/경계 아티팩트)을 포착합니다.
* **단일 얼굴 → 다중 얼굴 일반화**: 학습 분포와 다른 테스트 분포에서의 강건성을 확인하기 위해 채택한 분할 방식입니다.
* **모듈화된 모델 구조**: 각 모델을 독립적인 모듈로 분리하여 유지보수성과 확장성을 향상시켰습니다.
* **개별 실험 지원**: 모델별 최적화된 실험 스크립트를 제공하여 세부 튜닝과 비교 분석을 용이하게 합니다.

---

## 트러블슈팅 (Troubleshooting)

* **CUDA Out of Memory**: `--batch_size`/해상도 축소, `num_workers` 조정, 혼합정밀(AMP) 사용 고려
* **FPS/길이 상이**: 로더에서 균일 샘플링을 강제하거나, 프레임 패딩/크롭 로직을 활성화
* **AUC 낮음**: 클래스 불균형/임계값 이슈 가능 → 클래스 가중치, Focal Loss, 임계값 튜닝 시도

---

## 라이선스 (License)

이 프로젝트는 **MIT License**를 따릅니다. 상세 내용은 `LICENSE` 파일을 확인하세요.
