# Multimodal ECG-Clinical Text Transformer for Diagnostic Classification

A multimodal transformer that fuses 12-lead ECG waveforms with cardiologist report text
to perform 5-class cardiac diagnostic classification on PTB-XL.

> **Status:** Work in progress — results and benchmark numbers will be filled in upon completion.

## Overview

This project studies how structured biosignal representations and clinical language
representations each contribute to ECG diagnostic classification. Three model variants
are trained and evaluated:

- **Stage A** — ECG signal only (CNN stem + transformer encoder)
- **Stage B** — Cardiologist report text only (frozen multilingual DistilBERT)
- **Stage C** — Late fusion of both modalities via cross-attention

The full pipeline covers data preprocessing, model training, ONNX export and quantization,
FastAPI serving, Docker containerization, and latency/memory benchmarking.

## Dataset

[PTB-XL](https://physionet.org/content/ptb-xl/1.0.3/) — 21,799 clinical 12-lead ECG
recordings from a German hospital, each paired with a cardiologist-written report in German.
Open access, no application required.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

