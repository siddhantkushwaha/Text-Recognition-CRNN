import itertools
import logging

import numpy as np
import cv2

from model import CRNN
from utils import transform
from param import CHAR_VECTOR


def decode_label(out):
    out_best = list(np.argmax(out[0, 2:], axis=1))
    out_best = [k for k, g in itertools.groupby(out_best)]
    outstr = ''
    for i in out_best:
        if i < len(CHAR_VECTOR):
            outstr += CHAR_VECTOR[i]
    return outstr


def main():
    crnn = CRNN(is_train=False)
    crnn.model.load_weights('models/model-14--6.839.h5')

    img_fn = '../../FUNSD_TEXT_RECOGNITION/train_data/14.png'
    img = cv2.imread(img_fn, cv2.IMREAD_GRAYSCALE)

    img_pred = transform(img, img_w=128, img_h=64)
    img_pred = np.reshape(img_pred, (64, 128, 1))
    img_pred = (img_pred / 127.0) - 1.0

    out = crnn.model.predict(np.array([img_pred]))
    res = decode_label(out)

    print(res)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
