import argparse
import os
import urllib.request
import hashlib
import time

import cv2
import numpy as np
import onnxruntime
from utils.utils import E2EResizeForTest, KeepKeys, NormalizeImage, ToCHWImage
from e2e_utils.pg_postprocess import PGPostProcess
from hailo.inference_hailo import HailoRTInference


class PGNetPredictor:
    def __init__(self, img_path, cpu):
        self.img_path = img_path
        self.dict_path = "utils/ic15_dict.txt"
        if not os.path.exists(self.dict_path):
            with open(self.dict_path, "w") as f:
                f.writelines(chr_dct_list)
        if not cpu:
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        model_path = self._get_model_path(args.model_path)

        if model_path.endswith(".onnx"):
            self.sess = onnxruntime.InferenceSession(model_path, providers=providers)
        else:
            self.sess = HailoRTInference(model_path)
        self.infer_type = "onnx" if model_path.endswith(".onnx") else "hailo"

    def _get_model_path(self, model_path):
        """
        如果模型路径是URL，则下载模型到缓存目录
        """
        if model_path.startswith(("http://", "https://")):
            cache_dir = os.path.join(os.path.expanduser("~"), ".pgnet_models")
            os.makedirs(cache_dir, exist_ok=True)

            # 使用URL的MD5值作为文件名
            url_md5 = hashlib.md5(model_path.encode()).hexdigest()
            file_ext = os.path.splitext(model_path)[1] or ".onnx"
            cached_file = os.path.join(cache_dir, f"{url_md5}{file_ext}")
            if not os.path.exists(cached_file):
                print(f"正在从{model_path}下载模型...")
                try:
                    # 添加下载进度回调函数
                    def _progress_hook(count, block_size, total_size):
                        if total_size > 0:
                            percent = min(
                                int(count * block_size * 100 / total_size), 100
                            )
                            downloaded = count * block_size
                            # 计算下载速度（字节/秒）
                            if not hasattr(_progress_hook, "start_time"):
                                _progress_hook.start_time = time.time()
                                _progress_hook.last_size = 0
                                _progress_hook.last_time = _progress_hook.start_time

                            current_time = time.time()
                            interval = current_time - _progress_hook.last_time

                            # 每0.5秒更新一次显示
                            if interval > 0.5 or percent >= 100:
                                size_diff = downloaded - _progress_hook.last_size
                                speed = size_diff / interval if interval > 0 else 0

                                # 转换单位
                                if speed < 1024:
                                    speed_str = f"{speed:.2f} B/s"
                                elif speed < 1024 * 1024:
                                    speed_str = f"{speed / 1024:.2f} KB/s"
                                else:
                                    speed_str = f"{speed / (1024 * 1024):.2f} MB/s"

                                if total_size < 1024 * 1024:
                                    size_str = f"{downloaded / 1024:.2f}/{total_size / 1024:.2f} KB"
                                else:
                                    size_str = f"{downloaded / (1024 * 1024):.2f}/{total_size / (1024 * 1024):.2f} MB"

                                print(
                                    f"\r下载进度: [{percent}%] {size_str} 速度: {speed_str}",
                                    end="",
                                    flush=True,
                                )

                                _progress_hook.last_size = downloaded
                                _progress_hook.last_time = current_time

                    urllib.request.urlretrieve(model_path, cached_file, _progress_hook)
                    print("\n模型已下载并保存到{}".format(cached_file))
                except Exception as e:
                    raise Exception(f"下载模型时出错: {e}")
            else:
                print(f"使用缓存的模型: {cached_file}")

            return cached_file
        return model_path

    def preprocess(self, img_path):
        img = cv2.imread(img_path)
        self.ori_im = img.copy()
        data = {"image": img}
        transforms = [
            E2EResizeForTest(max_side_len=640, valid_set="totaltext"),
            NormalizeImage(
                scale=1.0 / 255.0,
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
                order="hwc",
            ),
            ToCHWImage(),
            KeepKeys(keep_keys=["image", "shape"]),
        ]
        for transform in transforms:
            if (
                self.infer_type == "hailo"
                and transform.__class__.__name__ == "NormalizeImage"
            ):
                continue
            data = transform(data)
        img, shape_list = data
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        return img, shape_list

    def predict(self, img):
        if self.infer_type == "hailo":
            ort_inputs = {self.sess.get_inputs()[0].name: img.transpose(0, 2, 3, 1)}
        else:
            ort_inputs = {self.sess.get_inputs()[0].name: img}
        outputs = self.sess.run(None, ort_inputs)
        preds = {}
        if isinstance(self.sess, onnxruntime.InferenceSession):
            preds["f_border"] = outputs[0]
            preds["f_char"] = outputs[1]
            preds["f_direction"] = outputs[2]
            preds["f_score"] = outputs[3]
        else:
            for key, output in outputs.items():
                if output.shape[-1] == 4:
                    preds["f_border"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                    preds["f_border"][:, :2, ...] = preds["f_border"][:, :2, ...] / 640
                    preds["f_border"][:, 2:, ...] = preds["f_border"][:, 2:, ...] / 100
                elif output.shape[-1] == 1:
                    preds["f_score"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                elif output.shape[-1] == 2:
                    preds["f_direction"] = output.transpose(0, 3, 1, 2).astype(
                        np.float32
                    )
                elif output.shape[-1] == 37:
                    preds["f_char"] = output.transpose(0, 3, 1, 2).astype(np.float32)
                else:
                    raise ValueError(f"output shape {output.shape} is not supported")
        return preds

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def postprocess(self, preds, shape_list):
        pgpostprocess = PGPostProcess(
            character_dict_path=self.dict_path,
            valid_set="totaltext",
            score_thresh=0.5,
            mode="fast",
        )
        post_result = pgpostprocess(preds, shape_list)
        points, strs = post_result["points"], post_result["texts"]
        dt_boxes = self.filter_tag_det_res_only_clip(points, self.ori_im.shape)
        return dt_boxes, strs

    def __call__(self):
        img, shape_list = self.preprocess(self.img_path)
        preds = self.predict(img)
        dt_boxes, strs = self.postprocess(preds, shape_list)
        return dt_boxes, strs

    def draw(self, dt_boxes, strs, img_path):
        src_im = cv2.imread(img_path)
        width, height, _ = src_im.shape
        for box, str in zip(dt_boxes, strs):
            box = box.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(src_im, [box], True, color=(255, 255, 0), thickness=2)
            cv2.putText(
                src_im,
                str,
                org=(int(box[0, 0, 0]), int(box[0, 0, 1])),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=0.7 / 500 * width / 2,
                color=(0, 255, 0),
                thickness=int(1 / 1000 * width),
            )
        img_out_name = os.path.basename(img_path).split(".")[0]
        img_out_name = f"{img_out_name}_pgnet.jpg"
        cv2.imwrite(img_out_name, src_im)
        return src_im


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PGPNET inference")
    parser.add_argument("model_path", type=str, help="onnxmodel path")
    parser.add_argument("img_path", type=str, help="image path")
    parser.add_argument(
        "--cpu", action="store_true", help="cpu inference, default device is gpu"
    )
    args = parser.parse_args()
    pgnetpredictor = PGNetPredictor(args.img_path, args.cpu)
    dt_boxes, strs = pgnetpredictor()
    print(f"Predict string:{strs}")
    pgnetpredictor.draw(dt_boxes, strs, args.img_path)
