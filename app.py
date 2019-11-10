from flask import Flask, request, jsonify
from PIL import Image

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
    bufs = request.files.getlist('images')
    images = [np.array(Image.open(buf).convert('RGB')) for buf in bufs]
    global graph
    with graph.as_default():
        outs = process_image(model.model, images)
    return jsonify(outs)


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    model = load_model(model_path='models/model-60--8.407.h5')
    graph = tf.get_default_graph()
    app.run(host='0.0.0.0', port=8778)
