import logging

import cv2
import pytesseract as ts
from difflib import SequenceMatcher

from predict import load_model, process_image
from utils import load_annotation, get_image_paths


# %%

def similarity_score(a, b):
    return SequenceMatcher(None, a, b).ratio()


def eval_image(img_fn, crnn):
    img = cv2.imread(img_fn)

    true_text = load_annotation(img_fn)
    ts_text = ts.image_to_string(img, config=("-l eng --oem 1 --psm 8"))
    crnn_text = process_image(crnn.model, [img])[0]

    return true_text, (ts_text, similarity_score(true_text, ts_text)), (
        crnn_text, similarity_score(true_text, crnn_text))


def main():
    crnn = load_model('models/model-60--8.407.h5')
    root = '../../FUNSD_TEXT_RECOGNITION/test_data'

    images = get_image_paths(root)
    ts_score = 0
    crnn_score = 0
    for image in images:
        res = eval_image(image, crnn)
        ts_score += res[1][1]
        crnn_score += res[2][1]

        print(image)
        print(res)
        print(ts_score, crnn_score)
        print()


# %%

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
