import os
import itertools
import logging

import numpy as np
import cv2

from model import CRNN
from utils import transform, load_annotation
from param import CHAR_VECTOR


def load_model(model_path):
    crnn = CRNN(is_train=False)
    crnn.model.load_weights(model_path)
    return crnn


def decode_label(out):
    out_best = list(np.argmax(out[2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for i in out_best:
        if i < len(CHAR_VECTOR):
            outstr += CHAR_VECTOR[i]
    return outstr


def process_image(model, images):
    processed_images = []
    for img in images:
        processed_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        processed_image = transform(processed_image, img_w=128, img_h=64)
        processed_image = np.reshape(processed_image, (64, 128, 1))
        processed_image = (processed_image / 127.0) - 1.0
        processed_images.append(processed_image)

    predictions = model.predict(np.array(processed_images))
    predictions = [decode_label(p) for p in predictions]
    return predictions


def main():
    crnn = load_model('models/model-60--8.407.h5')
    root = '../FUNSD_TEXT_RECOGNITION/test_data'
    for i in os.listdir(root):
        if not i.endswith('.png'):
            continue
        img_fn = os.path.join(root, i)
        img = cv2.imread(img_fn)
        print(process_image(crnn.model, [img])[0], load_annotation(img_fn))


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
