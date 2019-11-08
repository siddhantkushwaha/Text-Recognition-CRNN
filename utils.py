import os
import cv2

from param import letters, max_text_len


def get_image_paths(data_path):
    allowed_extensions = ['jpg', 'png', 'jpeg', 'JPG']
    files = [os.path.join(data_path, file) for file in os.listdir(data_path) if
             any(file.endswith(ext) for ext in allowed_extensions)]
    return files


def text_to_labels(text):
    labels = []
    for x in text:
        i = letters.get(x, -1)
        if i > -1:
            labels.append(i)
    return labels[:max_text_len]


def load_annotation(image_path):
    change_ext = lambda x, ext: x.replace(os.path.basename(x).split('.')[1], ext)
    text_path = change_ext(image_path, 'txt')

    if not os.path.exists(text_path):
        return None

    with open(text_path, 'r') as f:
        text = f.read()

    text = text.strip()
    return text


def transform(img, img_w, img_h):
    h, w = img.shape[0], img.shape[1]

    # resize without distorting
    aspect_ratio = h / w

    d1 = (img_w, int(img_w * aspect_ratio))
    d2 = (int(img_h / aspect_ratio), img_h)
    re_img = cv2.resize(img, min(d1, d2))

    # fill with constant color to meet size requirements
    delta_w = img_w - re_img.shape[1]
    delta_h = img_h - re_img.shape[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    return cv2.copyMakeBorder(re_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(255, 255, 255))
