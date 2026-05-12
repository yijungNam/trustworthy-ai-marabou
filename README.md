# Trustworthy AI Assignment #3 — Marabou Neural Network Verification

본 프로젝트는 SMT 기반 신경망 검증 도구인 [Marabou](https://github.com/NeuralNetworkVerification/Marabou)를
사용하여, 작은 MNIST 분류기의 국소적 robustness 성질을 형식적으로 검증한 결과를 담고 있다.

## Setup

 **Python 3.11** (maraboupy only supports 3.8–3.11) 버전이 필요함.

```bash
conda create -n marabou python=3.11 -y
conda activate marabou
pip install -r requirements.txt
```

## Usage

```bash
# 1.모델 학습 및 ONNX export
python scripts/train_mnist.py

# 2.검증 실험 실행
python test.py
```

## Project structure
.
├── scripts/
│   ├── train_mnist.py      # FC 네트워크 학습 및 ONNX export
│   └── verify.py           # Marabou 검증 query 실행
├── models/                 # 학습된 ONNX 모델
├── results/                # 검증 로그 및 counterexample 이미지
├── test.py                 # End-to-end 검증 데모
├── report.pdf              # 분석 보고서 (1–2 페이지)
└── requirements.txt
