import os
import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_uint
from skimage.filters import threshold_otsu
from keras_preprocessing.image import load_img, img_to_array

# save folder
save_to = './experiments/raw_analysis'
no_images = 25
# Load data
label_mono_path = './experiments/predict/'
label_truth_path = './experiments/true_label/'
# metrics
image_list = np.array([])
TPR = np.array([])
FPR = np.array([])
F_SCORE = np.array([])
IOU = np.array([])
IOU_SCORE = np.array([])
SENSITIVITY = np.array([])
SPECIFICITY = np.array([])
ACCURACY = np.array([])
TP = np.array([])
TN = np.array([])
FP = np.array([])
FN = np.array([])


def find_metrics(true_label, predict_label):
    # load truth label
    label_truth = load_img(true_label, color_mode='grayscale')
    label_truth = img_to_array(label_truth)
    original_label = np.squeeze(label_truth) / 255

    # load mono label
    label_mono = load_img(predict_label, color_mode='grayscale')
    label_mono = img_to_array(label_mono)
    label_predict = np.squeeze(label_mono) / 255

    # total pixels in truth label
    total_pixels = len(original_label.flatten())
    # print('total pixels in truth label: ', total_pixels)

    # find number of bg and label pixels in ground truth label
    flat_truth = original_label.flatten()
    positive_in_truth = flat_truth > 0
    total_bg_pxl_truth = list(positive_in_truth).count(False)
    total_obj_pxl_truth = list(positive_in_truth).count(True)
    # print('number of label pixels ', total_obj_pxl_truth)
    # print('number of background pixels ', total_bg_pxl_truth)

    # compare ground truth with prediction
    flat_predict = label_predict.flatten()

    tn = 0
    tp = 0
    fn = 0
    fp = 0

    for pixel in range(len(flat_truth)):
        if (flat_truth[pixel] == flat_predict[pixel]) and flat_truth[pixel] == 0:
            tn += 1
        elif (flat_truth[pixel] == flat_predict[pixel]) and flat_truth[pixel] != 0:
            tp += 1
        elif flat_truth[pixel] == 0 and flat_predict[pixel] != 0:
            fp += 1
        else:
            fn += 1

    # print('correct_bgd ', tn)
    # print('correct_lbl ', tp)
    # print('false_bgd ', fn)
    # print('false_lbl ', fp)

    return total_bg_pxl_truth, total_obj_pxl_truth, tp, tn, fp, fn


def tpr(true_positive, total_obj_pxl_in_truth):
    return true_positive / total_obj_pxl_in_truth


def fpr(false_positive, total_bg_pxl_in_truth):
    return false_positive / total_bg_pxl_in_truth


def f_score(true_positive, false_positive, false_negative):
    if true_positive == 0:
        return 0
    else:
        pres = true_positive / (true_positive + false_positive)
        recall = true_positive / (true_positive + false_negative)
        fscore = (2 * pres * recall) / (pres + recall)
        return fscore


def iou(true_label, predict_label):
    # load truth label
    label_truth = load_img(true_label, color_mode='grayscale')
    label_truth = img_to_array(label_truth)
    original_label = np.squeeze(label_truth) / 255
    original_label = original_label.flatten()

    # load mono label
    label_mono = load_img(predict_label, color_mode='grayscale')
    label_mono = img_to_array(label_mono)
    label_predict = np.squeeze(label_mono) / 255
    label_predict = label_predict.flatten()

    intersection = 0
    for i, j in zip(original_label, label_predict):
        if (i == j) and i == 0:
            continue
        elif (i == j) and i != 0:
            intersection += 1
    union = np.count_nonzero(original_label) + np.count_nonzero(label_predict) - intersection

    return intersection / union


def iou_score(true_label, predict_label):
    # load truth label
    label_truth = load_img(true_label, color_mode='grayscale')
    label_truth = img_to_array(label_truth)
    original_label = np.squeeze(label_truth) / 255
    original_label = original_label.flatten()

    # load mono label
    label_mono = load_img(predict_label, color_mode='grayscale')
    label_mono = img_to_array(label_mono)
    label_predict = np.squeeze(label_mono) / 255
    label_predict = label_predict.flatten()

    intersection = np.logical_and(original_label, label_predict)
    union = np.logical_or(original_label, label_predict)
    return np.sum(intersection) / np.sum(union)


def sensitivity(true_positive, false_negative):
    return true_positive / (true_positive + false_negative)


def specificity(false_positive, true_negative):
    return true_negative / (true_negative + false_positive)


def accuracy(true_positive, false_positive, true_negative, false_negative):
    return (true_positive + true_negative) / (true_positive + true_negative + false_positive + false_negative)


i = 0
predictions = os.listdir(label_mono_path)
print('working on {}'.format(label_mono_path))
print(predictions)
labels = os.listdir(label_truth_path)
print(labels)
for prediction, label in zip(predictions, labels):
    total_bg_pxl_truth, total_obj_pxl_truth, tp, tn, fp, fn = find_metrics(label_truth_path + label,
                                                                           label_mono_path + prediction)

    TP = np.append(TP, tp)
    TN = np.append(TN, tn)
    FP = np.append(FP, fp)
    FN = np.append(FN, fn)

    image_iou = iou(label_truth_path + label, label_mono_path + prediction)

    image_iou_score = iou_score(label_truth_path + label, label_mono_path + prediction)
    # if i == 0:
    #     print('number of label pixels ', total_obj_pxl_truth)
    #     print('number of background pixels ', total_bg_pxl_truth)
    #     print('true negative ', tn)
    #     print('true positive ', tp)
    #     print('false negative ', fn)
    #     print('false positive ', fp)

    TPR = np.append(TPR, tpr(tp, total_obj_pxl_truth))
    # TPR.append(tpr(tp, total_obj_pxl_truth))
    FPR = np.append(FPR, fpr(fp, total_bg_pxl_truth))
    # FPR.append(fpr(fp, total_bg_pxl_truth))
    F_SCORE = np.append(F_SCORE, f_score(tp, fp, fn))
    # F_SCORE.append(f_score(tp, fp, fn))
    IOU = np.append(IOU, image_iou)
    # image iou
    IOU_SCORE = np.append(IOU_SCORE, image_iou_score)
    # sensitivity
    SENSITIVITY = np.append(SENSITIVITY, sensitivity(tp, fn))
    # specificity
    SPECIFICITY = np.append(SPECIFICITY, specificity(fp, tn))
    # accuracy
    ACCURACY = np.append(ACCURACY, accuracy(tp, fp, tn, fn))

    image_list = np.append(image_list, i)
    i += 1
    if i % 10 == 0:
        print('analyzed ', i, ' images')

# print('number of label pixels ', total_obj_pxl_truth)
# print('number of background pixels ', total_bg_pxl_truth)
# print('true negative ', tn)
# print('true positive ', tp)
# print('false negative ', fn)
# print('false positive ', fp)
# print()
print('average TPR: ', np.average(TPR))
print('average FPR: ', np.average(FPR))
print('average F-Score: ', np.average(F_SCORE))
print('average IOU: ', np.average(IOU))
print('average sensitivity: ', np.average(SENSITIVITY))
print('average specificity: ', np.average(SPECIFICITY))
print('average accuracy: ', np.average(ACCURACY))
print('average IOU score: ', np.average(IOU_SCORE))

final = np.asarray([image_list, TPR, FPR, F_SCORE, IOU_SCORE, SENSITIVITY, SPECIFICITY, ACCURACY, TP, TN, FP, FN])
# save csv file
# np.savetxt(save_to + '/model_raw_metrics_super.csv', final, delimiter=',')

