import numpy as np
import cv2

from keras.utils import Sequence

from param import img_w, img_h, max_text_len
from utils import load_annotation, text_to_labels, transform, get_image_paths


class DataGenerator(Sequence):

    def __init__(self, data_path, batch_size=16):
        self.img_w = img_w
        self.img_h = img_h
        self.batch_size = batch_size
        self.image_paths = get_image_paths(data_path)

    def __getitem__(self, index):
        batch_image_paths = self.image_paths[index * self.batch_size:(index + 1) * self.batch_size]

        images = []
        labels = []
        label_lens = []
        for image_path in batch_image_paths:
            try:
                img, label, label_len = self.process(image_path)
                if label_len == 0:
                    continue
                images.append(np.reshape(img, (img_h, img_w, 1)))
                labels.append(label + [-1] * (max_text_len - label_len))
                label_lens.append(label_len)

            except Exception as e:
                print(str(e))

        input_length = np.ones((len(images), 1)) * ((self.img_w // 4) - 2)
        return [np.array(images), np.array(labels), input_length, np.array(label_lens)], [np.zeros([len(images)])]

    def __len__(self):
        return int(np.ceil(len(self.image_paths) / float(self.batch_size)))

    def on_epoch_end(self):
        np.random.shuffle(self.image_paths)

    def process(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = transform(img, img_w=img_w, img_h=img_h)
        img = img.astype(np.float32)
        img = (img / 127.0) - 1.0

        text = load_annotation(img_path)
        labels = text_to_labels(text)

        return img, labels, len(labels)
