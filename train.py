import os
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt


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
history = model.fit_generator(myGene, steps_per_epoch=5, epochs=20, callbacks=[model_checkpoint])

# Visualize the training history
plt.figure(figsize=(12, 4))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.legend()

# If accuracy metric is available
if 'accuracy' in history.history:
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

plt.show()
