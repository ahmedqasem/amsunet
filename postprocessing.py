import os

import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_uint
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_minimum
from keras_preprocessing.image import load_img, img_to_array
from skimage.transform import resize
from scipy.ndimage import uniform_filter, gaussian_filter, median_filter
import matplotlib
from skimage import data
from skimage.filters import try_all_threshold


# create mono label using threshold and save to folder


def otsu_thresh(label_thresh_path, target_path):
    ''' takes in an image from the label thresh path and save it into the target path '''
    predictions = os.listdir(label_thresh_path)
    print('loading from ', label_thresh_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_thresh_path + predict, color_mode='grayscale')
        label_predict = img_to_array(label_predict)
        old_label_predict = np.squeeze(label_predict) / 255

        # convert predicted label to mono using otsu thresholding
        thresh = threshold_otsu(old_label_predict)
        mono_label = old_label_predict > thresh

        # save mono label
        imageio.imwrite(target_path + predict, img_as_uint(mono_label))


def min_thresh(label_thresh_path, target_path):
    predictions = os.listdir(label_thresh_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_thresh_path + predict, color_mode='grayscale')
        label_predict = img_to_array(label_predict)
        old_label_predict = np.squeeze(label_predict) / 255

        # convert predicted label to mono using otsu thresholding
        thresh = threshold_minimum(old_label_predict)
        mono_label = old_label_predict > thresh

        # save mono label
        imageio.imwrite(target_path + predict, img_as_uint(mono_label))


def restore_size(label_predict_path, target_path, target_height, target_width):
    predictions = os.listdir(label_predict_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_predict_path + predict, color_mode='grayscale')
        label_predict = np.squeeze(img_to_array(label_predict)) / 255
        new_label = resize(label_predict, (target_height, target_width, 1), mode='constant', preserve_range=True)
        imageio.imwrite(target_path + predict, img_as_uint(new_label))


def uniform_otsu_thresh(label_thresh_path, target_path, filter_size=50):
    predictions = os.listdir(label_thresh_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_thresh_path + predict, color_mode='grayscale')
        label_predict = img_to_array(label_predict)
        old_label_predict = np.squeeze(label_predict) / 255

        # apply uniform filter
        blured = gaussian_filter(old_label_predict, filter_size)

        # convert predicted label to mono using otsu thresholding
        thresh = threshold_otsu(blured)
        mono_label = blured > thresh

        # save mono label
        imageio.imwrite(target_path + predict, img_as_uint(mono_label))


def gaus_otsu_thresh(label_thresh_path, target_path, filter_size=15):
    ''' takes in a raw image then applies gaussian blur with the filter size then applies otsu threshold
        and saves it into the target path '''
    print('applying gaussian blur {} then Otsu on images in {}'.format(filter_size, label_thresh_path))
    predictions = os.listdir(label_thresh_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_thresh_path + predict, color_mode='grayscale')
        label_predict = img_to_array(label_predict)
        old_label_predict = np.squeeze(label_predict) / 255

        # apply gaussian filter
        blured = gaussian_filter(old_label_predict, filter_size)

        # convert predicted label to mono using otsu thresholding
        thresh = threshold_otsu(blured)
        mono_label = blured > thresh

        # save mono label
        imageio.imwrite(target_path + predict, img_as_uint(mono_label))


def median_otsu_thresh(label_thresh_path, target_path, filter_size=55):
    predictions = os.listdir(label_thresh_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_thresh_path + predict, color_mode='grayscale')
        label_predict = img_to_array(label_predict)
        old_label_predict = np.squeeze(label_predict) / 255

        # apply uniform filter
        blured = median_filter(old_label_predict, filter_size)

        # convert predicted label to mono using otsu thresholding
        thresh = threshold_otsu(blured)
        mono_label = blured > thresh

        # save mono label
        imageio.imwrite(target_path + predict, img_as_uint(mono_label))


def gaus_blur(label_thresh_path, target_path, filter_size=15):
    print('applying gaussian blur {} on images in {}'.format(filter_size, label_thresh_path))
    predictions = os.listdir(label_thresh_path)
    for predict in predictions:
        # load predicted label
        label_predict = load_img(label_thresh_path + predict, color_mode='grayscale')
        label_predict = img_to_array(label_predict)
        old_label_predict = np.squeeze(label_predict) / 255

        # apply uniform filter
        blured = gaussian_filter(old_label_predict, filter_size)

        # convert predicted label to mono using otsu thresholding
        # thresh = threshold_otsu(blured)
        # mono_label = blured > thresh

        # save mono label
        imageio.imwrite(target_path + predict, img_as_uint(blured))

label_predict_path = './experiments/predict/'
label_blur_path = './experiments/gaus_blur_15/'
label_thresh_path = './experiments/otsu/'

# predictions = os.listdir(label_predict_path)

''' Mono Image - different experiments '''
# otsu_thresh(label_predict_path, label_thresh_path)
# min_thresh(label_predict_path, label_thresh_path)
# uniform_otsu_thresh(label_predict_path, label_thresh_path)
# gaus_otsu_thresh(label_predict_path, label_thresh_path)
# median_otsu_thresh(label_predict_path, label_thresh_path)

# apply gaussian blur
# gaus_blur(label_predict_path, label_blur_path, filter_size=15)

# apply otsu alone
otsu_thresh(label_predict_path, label_thresh_path)

# apply gauss then otsu
# gaus_otsu_thresh(label_predict_path, label_thresh_path, filter_size=15)

''' restore size '''
# restore_size(label_thresh_path, target_path_final, 2048, 2048)


''' try all '''
# img = data.page()
#
# predictions = os.listdir(label_predict_path)
# print(predictions)
# label_predict = load_img(label_predict_path + predictions[10], color_mode='grayscale')
# label_predict = img_to_array(label_predict)
# old_label_predict = np.squeeze(label_predict) / 255
# fig, ax = try_all_threshold(old_label_predict, figsize=(10, 8), verbose=False)
# plt.show()
