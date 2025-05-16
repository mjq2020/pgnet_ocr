import os
os.environ["HAILO_MONITOR"] = "1"
import time
import argparse
import cv2
import numpy as np
import onnxruntime
from utils.utils import E2EResizeForTest, KeepKeys, NormalizeImage, ToCHWImage
from hailo.inference_hailo import HailoRTInference
from utils.download import download_model


class PGNetHailoPredictor:
    def __init__(self, model_path, cpu):
        self.model_path = model_path
        self.dict_path = "utils/ic15_dict.txt"
        self.sess = HailoRTInference(model_path)
        self.input_name = self.sess.get_inputs()[0].name

    def resize_image(self, img):
        img = cv2.resize(img, (640, 640))
        return img

    def preprocess(self, img):
        resized_img = self.resize_image(img)
        resized_img = np.transpose(resized_img, (2, 0, 1))
        resized_img = np.ascontiguousarray(np.expand_dims(resized_img, axis=0))
        return resized_img

    def predict(self, img, preprocess=True):
        if preprocess:
            resized_img = self.preprocess(img)
        else:
            resized_img = img

        ort_outs = self.sess.run(None, img)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default=r"/home/hk/github/pgnet_ocr/models/model_sim_.hef")
    parser.add_argument("--img_path", type=str, default="images/test.png")
    return parser.parse_args()


def preprocess(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (640, 640))
    img = np.transpose(img, (2, 0, 1)).astype(np.int8)
    img = np.ascontiguousarray(np.expand_dims(img, axis=0))
    return img


if __name__ == "__main__":
    args = get_args()
    pgnet_hailo_predictor = PGNetHailoPredictor(args.model_path, False)
    img = preprocess(args.img_path)
    print(f"img shape: {img.shape}, type: {img.dtype}")
    number = 100
    print(f"img shape: {img.shape}")
    ort_inputs = {pgnet_hailo_predictor.input_name: img}
    for i in range(100):
        pgnet_hailo_predictor.predict(ort_inputs, preprocess=False)
    start_time = time.time()
    for i in range(number):
        pgnet_hailo_predictor.predict(ort_inputs, preprocess=False)
    end_time = time.time()
    print(f"time: {(end_time - start_time)/number} seconds, fps: {number/(end_time - start_time)}")
