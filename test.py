from model import unet
from data import testGenerator, saveResult
from tensorflow.keras.models import load_model
import numpy as np

im_height = 512
im_width = 512

test_folder = './data/test/image'
models_folder = './trained_models'

# Load the pre-trained model
model_path = f'{models_folder}/model_cropped.hdf5'
model = unet(input_size=(im_height, im_width, 1))
model.load_weights(model_path)

# Generate test data
testGene = testGenerator(test_folder, target_size=(im_height, im_width))

# Generate test data and collect file names
test_data = list(testGenerator(test_folder, target_size=(512, 512)))

# Separate images and file names
test_images, file_names = zip(*test_data)

# Run predictions
results = model.predict(testGene, steps=25, verbose=1)

# Save results
saveResult("./data/predictions", results, file_names)