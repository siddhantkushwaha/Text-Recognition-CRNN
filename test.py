import time
import random
import logging

import cv2
import pytesseract as ts
from difflib import SequenceMatcher

from predict import load_model, process_image
from utils import load_annotation, get_image_paths

MODEL_PATH = 'models/model_synth_funsd2.h5'
# DIR_NAME = '../../FUNSD_TEXT_RECOGNITION/test_data'
DIR_NAME = 'data/synth_test_1'


# %%


def similarity_score(a, b):
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()


def eval_image(img_fn, crnn):
    img = cv2.imread(img_fn)

    true_text = load_annotation(img_fn)
    if true_text is None:
        true_text = ''
    ts_text = ts.image_to_string(img, config=("-l eng --oem 1 --psm 8"))
    crnn_text = process_image(crnn.model, [img])[0]

    res = (
        true_text,
        (ts_text, similarity_score(true_text, ts_text)),
        (crnn_text, similarity_score(true_text, crnn_text)),
    )

    return res


def main():
    crnn = load_model(MODEL_PATH)
    root = DIR_NAME

    images = get_image_paths(root)
    random.shuffle(images)

    ts_score = 0
    crnn_score = 0
    for n, image in enumerate(images, 1):
        res = eval_image(image, crnn)
        ts_score += res[1][1]
        crnn_score += res[2][1]

        print(image)
        print(res)
        print(ts_score/n, crnn_score/n)
        print()

        time.sleep(2)
        # break


# %%

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
