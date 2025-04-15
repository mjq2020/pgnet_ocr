# PGNet ONNX Inference (Based on PaddleOCR)

This is PGNET's onnxruntime inference implementation of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

## Setup

```bash
pip install -r requirements.txt
wget https://github.com/kuroko1t/onnx-ocr/releases/download/0.1/pgnet.onnx
```

## Run

```bash
python inference_pgnet.py pgnet.onnx images/test.png
```

## Result

| Original  | Result |
| ------------- | ------------- |
| ![image0](https://github.com/mjq2020/pgnet_ocr/blob/master/images/test.png?raw=true)  | ![image1](https://github.com/mjq2020/pgnet_ocr/blob/master/images/test_pgnet.jpg?raw=true) |
