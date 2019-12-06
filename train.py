import os
import logging

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adadelta

from data_generator import DataGenerator
from model import CRNN

TRAIN_DIR_NAME = '../../FUNSD_TEXT_RECOGNITION/train_data'


def main():
    crnn = CRNN(is_train=True)

    ada = Adadelta()
    crnn.model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=ada)

    batch_size = 64
    train_gen = DataGenerator(data_path=TRAIN_DIR_NAME, batch_size=batch_size)

    os.system('mkdir -p models')
    early_stop = EarlyStopping(monitor='loss', min_delta=0.01, patience=8, mode='min', verbose=1)
    checkpoint = ModelCheckpoint(filepath='models/model_curr_{epoch:02d}_{loss:.3f}.h5', monitor='loss', verbose=1,
                                 mode='min', period=1, save_weights_only=True)

    # load previous checkpoints
    crnn.model.load_weights('models/model_synth_2_35.h5')

    crnn.model.fit_generator(
        generator=train_gen,
        steps_per_epoch=len(train_gen.image_paths) // batch_size,
        epochs=100,

        callbacks=[checkpoint, early_stop],

        workers=16,
        use_multiprocessing=True,
        max_queue_size=10,

        verbose=1,
    )


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.ERROR)
    main()
