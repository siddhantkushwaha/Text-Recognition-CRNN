import cv2
from flask import Flask, request, jsonify
from PIL import Image

import os
import json
import logging
import numpy as np

import tensorflow as tf

from predict import load_model, process_image

app = Flask(__name__)


@app.route('/')
def index():
    return 'Get request to CRNN server.'


@app.route('/process', methods=['POST'])
def process():
    image_buf = request.files['image']
    image = Image.open(image_buf).convert('RGB')
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    lines = json.loads(request.form['lines'])
    images = []
    for i, line in enumerate(lines, 0):
        x1, y1, x2, y2, x3, y3, x4, y4 = line
        images.append(image[min(y1, y2):max(y3, y4), min(x1, x4): max(x2, x3)])

    global graph
    with graph.as_default():
        outs = process_image(model.model, images)
    return jsonify(outs)


if __name__ == '__main__':
    # because CUDA is being used somwhere else
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    logging.getLogger().setLevel(logging.ERROR)
    model = load_model(model_path='models/model-60--8.407.h5')
    graph = tf.get_default_graph()

    app.run(host='127.0.0.1', port=5002)
