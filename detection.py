import os
from base64 import b64encode, b64decode
from typing import AnyStr, List, Dict
from collections import Counter

import numpy as np
import cv2 as cv
import keras
import tensorflow as tf
from yolo4.model import yolo4_body

from decode_np import Decode

__all__ = ("DetectJapan", "detect_japan_obj")

session = tf.Session()
keras.backend.set_session(session)


def get_class(classes_path):
    classes_path = os.path.expanduser(classes_path)
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names


def get_anchors(anchors_path):
    anchors_path = os.path.expanduser(anchors_path)
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(",")]
    return np.array(anchors).reshape(-1, 2)


class DetectJapan:
    model_path = "JPY_weight.h5"  # Keras model or weights must be a .h5 file.
    anchors_path = "model_data/yolo4_anchors.txt"
    classes_path = "model_data/JPY_classes.txt"
    jpy_classes = ('JPY_500', 'JPY_100', 'JPY_50', 'JPY_10', 'JPY_1', 'JPY_5')

    def __init__(self, conf_thresh: float = 0.2, nms_thresh: float = 0.45):
        class_names = get_class(self.classes_path)
        anchors = get_anchors(self.anchors_path)
        model_image_size = (416, 416)
        self._model: keras.Model = yolo4_body(
            inputs=keras.Input(shape=model_image_size + (3,)),
            num_anchors=len(anchors) // 3,
            num_classes=len(class_names),
        )
        self._model.load_weights(os.path.expanduser(self.model_path))
        self._decoder: Decode = Decode(
            obj_threshold=conf_thresh,
            nms_threshold=nms_thresh,
            input_shape=model_image_size,
            _yolo=self._model,
            all_classes=class_names,
        )

    @property
    def model(self) -> keras.Model:
        return self._model

    @property
    def decoder(self) -> Decode:
        return self._decoder

    def detect(self, image_b64: AnyStr, *, fmt: str = ".png") -> Dict:
        image_bin: bytes = b64decode(image_b64)
        image = cv.imdecode(np.frombuffer(image_bin, np.uint8), cv.IMREAD_COLOR)
        with session.as_default():
            with session.graph.as_default():
                detect_image, *_, classes = self._decoder.detect_image(image, True)
        is_success, buffer = cv.imencode(fmt, detect_image)

        return {
            "img": b64encode(buffer.tobytes()).decode(),
            "count": self.count(classes)
        }

    def count(self, classes: List[int]):
        counter = Counter(classes)
        for class_ in tuple(counter):
            counter[self.jpy_classes[class_]] = counter.pop(class_)
        return counter


detect_japan_obj = DetectJapan()
