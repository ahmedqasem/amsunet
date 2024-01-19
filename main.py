from model import *
from data import *
import tensorflow as tf

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# disable gpu
tf.config.experimental.set_visible_devices([], 'GPU')

im_height = 512
im_width = 512

data_gen_args = dict(rotation_range=0.2,
                     width_shift_range=0.05,
                     height_shift_range=0.05,
                     shear_range=0.05,
                     zoom_range=0.05,
                     horizontal_flip=True,
                     fill_mode='nearest')

myGene = trainGenerator(5, 'data/data_25/train', 'image', 'label', data_gen_args, save_to_dir=None,
                        target_size=(im_height, im_width))

model = unet(input_size=(im_height, im_width, 1))
model_checkpoint = ModelCheckpoint('model_cropped.hdf5', monitor='loss', verbose=1, save_best_only=True)
model.fit_generator(myGene, steps_per_epoch=5, epochs=1, callbacks=[model_checkpoint])

testGene = testGenerator("data/data_25/testing/image", target_size = (im_height, im_width))
results = model.predict_generator(testGene, 25, verbose=1)
saveResult("data/data_25/testing/label_predict", results)
