# Triton Deployment - SAM3 ONNX

## 📌 Overview
Deploy SAM3 ONNX model using NVIDIA Triton Inference Server optimized for RTX 3090 (24GB VRAM).

This repository contains:

* Exported SAM3 ONNX model
* Triton model repository structure
* Docker deployment setup

---

## 🧠 Model Info

* Model: SAM3 (ONNX)
* Framework: ONNXRuntime
* Inference Server: Triton 24.08+
* GPU: NVIDIA (CUDA required)

⚠️ Note: If model > 2GB, ONNX external data format must be used.

---

## 📂 Project Structure

```
├── build
│   ├── docker-compose.yml
│   └── Dockerfile
├── README.md
├── repo
│   ├── sam3_decoder
│   │   ├── 1
│   │   └── config.pbtxt
│   ├── sam3_image_encoder
│   │   ├── 1
│   │   └── config.pbtxt
│   ├── sam3_language_encoder
│   │   ├── 1
│   │   └── config.pbtxt
│   └── sam3_pipeline
│       ├── 1
│       └── config.pbtxt
└── source
    └── sam3-onnx
        ├── assets
        ├── check.py
        ├── export_onnx.py
        ├── images
        ├── infer_onnx.py
        ├── infer_torch.py
        ├── infer_triton.py
        ├── LICENSE
        ├── Makefile
        ├── models
        ├── pyproject.toml
        ├── README.md
        ├── sam3
        └── uv.lock
```

---

## 🚀 Run Triton Server

```bash
docker compose up --build 
```

---

## 🔍 Check Model Status


```
http://localhost:8000/v2/health/ready
```


---

## 🧪 Test Inference (Python Client)

Use the provided Triton client script:

👉 *[Run infer_triton.py](source/sam3-onnx/infer_triton.py)*

Example:

```bash
cd source/sam3-onnx
uv infer_triton.py \
    --image images/bus.jpg \
    --text-prompt "person"

