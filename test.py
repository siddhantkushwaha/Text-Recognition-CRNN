import time
import random
import logging

import cv2
import pytesseract as ts
from difflib import SequenceMatcher

from predict import load_model, process_image
from utils import load_annotation, get_image_paths
import spell

MODEL_NAME = 'model_new_best.h5'
DIR_NAME = '../../FUNSD_TEXT_RECOGNITION'


# %%


def similarity_score(a, b):
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()


def correction(word):
    corrected = spell.correction(word)
    score = similarity_score(word, corrected)
    return corrected if score > 0.8 else word


def eval_image(img_fn, crnn):
    img = cv2.imread(img_fn)

    true_text = load_annotation(img_fn)
    ts_text = ts.image_to_string(img, config=("-l eng --oem 1 --psm 8"))
    crnn_text = process_image(crnn.model, [img])[0]
    corrected = correction(crnn_text)

    res = (
        true_text,
        (ts_text, similarity_score(true_text, ts_text)),
        (crnn_text, similarity_score(true_text, crnn_text)),
        (corrected, similarity_score(true_text, corrected)),
    )

    return res


def main():
    crnn = load_model(f'models/{MODEL_NAME}')
    root = f'../../{DIR_NAME}//test_data'

    images = get_image_paths(root)
    random.shuffle(images)

    ts_score = 0
    crnn_score = 0
    corrected_score = 0
    for image in images:
        res = eval_image(image, crnn)
        ts_score += res[1][1]
        crnn_score += res[2][1]
        corrected_score += res[3][1]

        print(image)
        print(res)
        print(ts_score, crnn_score, corrected_score)
        print()

        time.sleep(2)
        # break


# %%

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
