import cv2
import numpy as np


class ToCHWImage(object):
    """convert hwc image to chw image"""

    def __init__(self, **kwargs):
        pass

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        data["image"] = img.transpose((2, 0, 1))
        return data


class NormalizeImage(object):
    """normalize image such as subtract mean, divide std"""

    def __init__(self, scale=None, mean=None, std=None, order="chw", **kwargs):
        if isinstance(scale, str):
            scale = eval(scale)
        self.scale = np.float32(scale if scale is not None else 1.0 / 255.0)
        mean = mean if mean is not None else [0.485, 0.456, 0.406]
        std = std if std is not None else [0.229, 0.224, 0.225]

        shape = (3, 1, 1) if order == "chw" else (1, 1, 3)
        self.mean = np.array(mean).reshape(shape).astype("float32")
        self.std = np.array(std).reshape(shape).astype("float32")

    def __call__(self, data):
        img = data["image"]
        from PIL import Image

        if isinstance(img, Image.Image):
            img = np.array(img)
        assert isinstance(img, np.ndarray), "invalid input 'img' in NormalizeImage"
        data["image"] = (img.astype("float32") * self.scale - self.mean) / self.std
        return data


class KeepKeys(object):
    def __init__(self, keep_keys, **kwargs):
        self.keep_keys = keep_keys

    def __call__(self, data):
        data_list = []
        for key in self.keep_keys:
            data_list.append(data[key])
        return data_list


class E2EResizeForTest(object):
    def __init__(self, **kwargs):
        super(E2EResizeForTest, self).__init__()
        self.max_side_len = kwargs["max_side_len"]
        self.valid_set = kwargs["valid_set"]

    def __call__(self, data):
        img = data["image"]
        src_h, src_w, _ = img.shape
        if self.valid_set == "totaltext":
            im_resized, [ratio_h, ratio_w] = self.resize_image_for_totaltext(
                img, max_side_len=self.max_side_len
            )
        else:
            im_resized, (ratio_h, ratio_w) = self.resize_image(
                img, max_side_len=self.max_side_len
            )
        data["image"] = im_resized
        data["shape"] = np.array([src_h, src_w, ratio_h, ratio_w])
        return data

    def resize_image_for_totaltext(self, im, max_side_len=512):
        h, w, _ = im.shape
        resize_w = w
        resize_h = h
        ratio = 1.25
        if h * ratio > max_side_len:
            ratio = float(max_side_len) / resize_h
        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)
        return im, (ratio_h, ratio_w)

    def resize_image(self, im, max_side_len=512):
        """
        resize image to a size multiple of max_stride which is required by the network
        :param im: the resized image
        :param max_side_len: limit of max image size to avoid out of memory in gpu
        :return: the resized image and the resize ratio
        """
        h, w, _ = im.shape

        resize_w = w
        resize_h = h

        # Fix the longer side
        if resize_h > resize_w:
            ratio = float(max_side_len) / resize_h
        else:
            ratio = float(max_side_len) / resize_w

        resize_h = int(resize_h * ratio)
        resize_w = int(resize_w * ratio)

        max_stride = 128
        resize_h = (resize_h + max_stride - 1) // max_stride * max_stride
        resize_w = (resize_w + max_stride - 1) // max_stride * max_stride
        im = cv2.resize(im, (int(resize_w), int(resize_h)))
        ratio_h = resize_h / float(h)
        ratio_w = resize_w / float(w)

        return im, (ratio_h, ratio_w)
