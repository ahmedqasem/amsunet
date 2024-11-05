import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

from model import unet
from data import trainGenerator


im_height = 512
im_width = 512

train_folder = './data/train'
models_folder = './trained_models'

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(5, train_folder, 'image', 'label', data_gen_args, save_to_dir=None,
                        target_size=(im_height, im_width))

model = unet(input_size=(im_height, im_width, 1))
model_checkpoint = ModelCheckpoint(f'{models_folder}/model_cropped.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=5, epochs=1, callbacks=[model_checkpoint])
