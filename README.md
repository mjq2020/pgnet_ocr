# PGNet ONNX/HAILO Inference (Based on PaddleOCR)

This is PGNET's onnxruntime inference implementation of [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR).

This is PGNET's hailo inference implementation of [hailort](https://github.com/hailo-ai/hailort).

## Setup

```bash
git clone https://github.com/mjq2020/pgnet_ocr.git

cd pgnet_ocr
pip install -r requirements.txt
```

## Run

```bash
# onnx inference
python inference_pgnet.py https://github.com/mjq2020/pgnet_ocr/releases/download/v0.1/pgnet.onnx images/test.png

# hailo inference
python inference_pgnet.py https://github.com/mjq2020/pgnet_ocr/releases/download/v0.1/pgnet_640.hef images/test.png
```

## Result

| Original  | Result |
| ------------- | ------------- |
| ![image0](https://github.com/mjq2020/pgnet_ocr/blob/master/images/test.png?raw=true)  | ![image1](https://github.com/mjq2020/pgnet_ocr/blob/master/images/test_pgnet.jpg?raw=true) |
