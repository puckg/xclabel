# -*- coding: utf-8 -*-
import sys
import os
from ultralytics import YOLO

model_path = r"e:\Yolo\xclabel\xclabel-main\exports\custom_models\best.pt"
output_dir = r"e:\Yolo\xclabel\xclabel-main\exports\onnx"

print(f"LOADING_MODEL: Loading model from {model_path}")
model = YOLO(model_path)

print(f"EXPORTING: Exporting to ONNX format...")
export_kwargs = {
    'format': 'onnx',
    'imgsz': 320,
    'simplify': True,
    'opset': 12,
    'dynamic': False,
    'half': False,
    'project': output_dir
}

model.export(**export_kwargs)

print(f"EXPORT_COMPLETE: Export completed successfully")
