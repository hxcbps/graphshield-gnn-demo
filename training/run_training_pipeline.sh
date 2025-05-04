#!/usr/bin/env bash
set -e
echo "▶ Entrenando modelo…"
python -m training.src.train
echo "✅ Exportando ONNX"
python -m training.src.export_onnx
