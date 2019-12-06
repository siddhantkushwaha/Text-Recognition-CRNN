import time
import random
import logging

import cv2
import pandas as pd
import pytesseract as ts
from difflib import SequenceMatcher

from predict import load_model, process_image
from utils import load_annotation, get_image_paths

MODEL_PATH = 'models/model_synth_2_10.h5'
DIR_NAME = 'data/synth_test_1'


# %%


def similarity_score(a, b):
    if a is None or b is None:
        return 0
    a = a.lower()
    b = b.lower()
    return SequenceMatcher(None, a, b).ratio()


def eval_image(img_fn, crnn):
    img = cv2.imread(img_fn)

    true_text = load_annotation(img_fn)
    ts_text = ts.image_to_string(img, config=("-l eng --oem 1 --psm 8"))
    crnn_text = process_image(crnn.model, [img])[0]

    return {
        'true_text': true_text,
        'tesseract': ts_text,
        'crnn': crnn_text,
    }


def main():
    crnn = load_model(MODEL_PATH)
    root = DIR_NAME

    images = get_image_paths(root)
    random.shuffle(images)

    rec = []

    ts_score = 0
    crnn_score = 0
    for n, image in enumerate(images, 1):
        result = eval_image(image, crnn)
        ts_score += similarity_score(result['true_text'], result['tesseract'])
        crnn_score += similarity_score(result['true_text'], result['crnn'])
        rec.append({'image': image, 'true_text': result['true_text'], 'tesseract': result['tesseract'], 'crnn': result['crnn']})

        print(n, image)
        print(result)
        print(ts_score / n, crnn_score / n)
        print()

        # time.sleep(2)

    rec_df = pd.DataFrame(rec)
    rec_df.to_csv('records_10.csv')

    # %%


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
