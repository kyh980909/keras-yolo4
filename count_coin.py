import os
import colorsys
import collections
import io

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
from PIL import Image
import cv2
import base64
import matplotlib.pyplot as plt

from decode_np import Decode


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
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def init():
    # if (country == 'KRW'):
    model_path = 'KRW_weight.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/KRW_classes.txt'

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    model_image_size = (416, 416)

    # 分数阈值和nms_iou阈值
    conf_thresh = 0.2
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size + (3,)), num_anchors // 3, num_classes)

    model_path = os.path.expanduser(model_path)

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names) # 위 과정의 시간이 오래걸림
    # else:
    #     model_path = 'JPY_weight.h5'
    #     anchors_path = 'model_data/yolo4_anchors.txt'
    #     classes_path = 'model_data/JPY_classes.txt'

    return _decode

def jpy_count_coin(img): # img : str
    model_path = 'JPY_weight.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/JPY_classes.txt'
    jpy_classes = ['JPY_500', 'JPY_100', 'JPY_50', 'JPY_10', 'JPY_1', 'JPY_5']
    count = {}
    result = {}
    total = 0

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    model_image_size = (416, 416)

    # 分数阈值和nms_iou阈值
    conf_thresh = 0.2
    nms_thresh = 0.8

    yolo4_model = yolo4_body(Input(shape=model_image_size + (3,)), num_anchors // 3, num_classes)

    model_path = os.path.expanduser(model_path)

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    try:
        encoded_img = np.fromstring(base64.b64decode(img), dtype = np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    except:
        print('Open Error! Try again!')
    else:
        image, boxes, scores, classes = _decode.detect_image(img, True)
        cv2.imwrite('predict.png',image)
        with open('predict.png', 'rb') as img:
            base64_string = base64.b64encode(img.read()).decode('utf-8')
        count = collections.Counter(classes)
        for key in tuple(count.keys()):  # 딕셔너리 키 이름 변경
            count[jpy_classes[key]] = count.pop(key)

        for key, value in count.items():
            total += int(key[str(key).find('_') + 1:]) * value
        result['result'] = count
        result['total'] = total
        result['image'] = base64_string

    # yolo4_model.close_session()

    return result


def krw_count_coin(img, _decode): # img : str
    # model_path = 'KRW_weight.h5'
    # anchors_path = 'model_data/yolo4_anchors.txt'
    # classes_path = 'model_data/KRW_classes.txt'
    krw_classes = ['KRW_500', 'KRW_100', 'KRW_50', 'KRW_10']
    count = {}
    result = {}
    total = 0

    # class_names = get_class(classes_path)
    # anchors = get_anchors(anchors_path)

    # num_anchors = len(anchors)
    # num_classes = len(class_names)

    # model_image_size = (416, 416)

    # conf_thresh = 0.2
    # nms_thresh = 0.45

    # yolo4_model = yolo4_body(Input(shape=model_image_size + (3,)), num_anchors // 3, num_classes)

    # model_path = os.path.expanduser(model_path)

    # yolo4_model.load_weights(model_path)

    # _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names) # 위 과정의 시간이 오래걸림

    print(_decode)
    try:
        encoded_img = np.fromstring(base64.b64decode(img), dtype = np.uint8)
        img = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR)
    except:
        print('Open Error! Try again!')
    else:
        image, boxes, scores, classes = _decode.detect_image(img, True) # predict 부분
        cv2.imwrite('predict.png',image)
        with open('predict.png', 'rb') as img:
            base64_string = base64.b64encode(img.read()).decode('utf-8')
        count = collections.Counter(classes)
        for key in tuple(count.keys()):  # 딕셔너리 키 이름 변경
            count[krw_classes[key]] = count.pop(key)

        for key, value in count.items():
            total += int(key[str(key).find('_') + 1:]) * value
        result['result'] = count
        result['total'] = total
        result['image'] = base64_string

    # yolo4_model.close_session()

    return result


if __name__ == '__main__':
    model_path = 'JPY_weight.h5'
    anchors_path = 'model_data/yolo4_anchors.txt'
    classes_path = 'model_data/JPY_classes.txt'
    jpy_classes = ['JPY_500', 'JPY_100', 'JPY_50', 'JPY_10', 'JPY_1', 'JPY_5']
    count = {}
    result = {}
    total = 0

    class_names = get_class(classes_path)
    anchors = get_anchors(anchors_path)

    num_anchors = len(anchors)
    num_classes = len(class_names)

    model_image_size = (416, 416)

    # 分数阈值和nms_iou阈值
    conf_thresh = 0.2
    nms_thresh = 0.45

    yolo4_model = yolo4_body(Input(shape=model_image_size + (3,)), num_anchors // 3, num_classes)

    model_path = os.path.expanduser(model_path)
    assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

    yolo4_model.load_weights(model_path)

    _decode = Decode(conf_thresh, nms_thresh, model_image_size, yolo4_model, class_names)

    img = input('Input image filename:')
    try:
        image = cv2.imread(img)
    except:
        print('Open Error! Try again!')
    else:
        image, boxes, scores, classes = _decode.detect_image(image, True)
        count = collections.Counter(classes)

        for key in tuple(count.keys()):  # 딕셔너리 키 이름 변경
            count[jpy_classes[key]] = count.pop(key)

        for key, value in count.items():
            total += int(key[str(key).find('_') + 1:]) * value
        result['result'] = count
        result['total'] = total
        result['image'] = image
        cv2.imwrite('result.png', image)

    yolo4_model.close_session()
