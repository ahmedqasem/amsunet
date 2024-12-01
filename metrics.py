import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import img_as_uint
from skimage.filters import threshold_otsu
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd

"""
This script calculates and saves evaluation metrics for binary segmentation models by comparing predicted labels (or processed images)
to ground truth labels. The results are saved in a CSV file for further analysis.

INSTRUCTIONS:

1. Folder Setup:
    - `label_mono_path`: Set this variable to the directory containing the processed prediction images to analyze.
      Ensure that this folder contains only the predicted label images in grayscale format (e.g., PNG, JPEG).
    - `label_truth_path`: Set this variable to the directory containing the ground truth label images for comparison.
      The ground truth images should also be in grayscale format and should match the filenames in `label_mono_path`
      for accurate comparison.

2. Image Processing:
    - Ensure all images in `label_mono_path` have been processed and saved with the same dimensions as the images in `label_truth_path`.
    - Each image in `label_mono_path` should have a corresponding image with the same name in `label_truth_path`.

3. Metrics Calculated:
    - True Positive Rate (TPR)
    - False Positive Rate (FPR)
    - F-Score
    - Intersection over Union (IOU Score)
    - Sensitivity
    - Specificity
    - Accuracy
    - True Positives (TP), True Negatives (TN), False Positives (FP), False Negatives (FN)

4. Output:
    - A CSV  will be saved in the folder specified by `save_to`.
    - Each row in the CSV file corresponds to an image, identified by its filename, along with the computed metrics for that image.
"""

# save folder
save_to = './experiments/raw_analysis' # change output folder as needed
#no_images = 25
# Load data
label_mono_path = './data/predictions/' # folder for the predictions that we need to analyze
label_truth_path = './data/test/label/' # ground truth folder

# Metrics
image_list = []
TPR = []
FPR = []
F_SCORE = []
IOU = []
IOU_SCORE = []
SENSITIVITY = []
SPECIFICITY = []
ACCURACY = []
TP = []
TN = []
FP = []
FN = []


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

if __name__ =='__main__':
    i = 0
    predictions = os.listdir(label_mono_path)
    print('working on {}'.format(label_mono_path))
    print(predictions)
    labels = os.listdir(label_truth_path)
    print(labels)
    for prediction, label in zip(predictions, labels):
        # print(f'analyzing {prediction}') # optional

        total_bg_pxl_truth, total_obj_pxl_truth, tp, tn, fp, fn = find_metrics(label_truth_path + label,
                                                                            label_mono_path + prediction)

        # Append metrics for each image
        image_list.append(prediction)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TPR.append(tpr(tp, total_obj_pxl_truth))
        FPR.append(fpr(fp, total_bg_pxl_truth))
        F_SCORE.append(f_score(tp, fp, fn))
        IOU_SCORE.append(iou_score(label_truth_path + label, label_mono_path + prediction))
        SENSITIVITY.append(sensitivity(tp, fn))
        SPECIFICITY.append(specificity(fp, tn))
        ACCURACY.append(accuracy(tp, fp, tn, fn))

        # Append metrics for each image
        image_list.append(prediction)
        TP.append(tp)
        TN.append(tn)
        FP.append(fp)
        FN.append(fn)
        TPR.append(tpr(tp, total_obj_pxl_truth))
        FPR.append(fpr(fp, total_bg_pxl_truth))
        F_SCORE.append(f_score(tp, fp, fn))
        IOU_SCORE.append(iou_score(label_truth_path + label, label_mono_path + prediction))
        SENSITIVITY.append(sensitivity(tp, fn))
        SPECIFICITY.append(specificity(fp, tn))
        ACCURACY.append(accuracy(tp, fp, tn, fn))

        i += 1
        if i % 10 == 0:
            print('analyzed ', i, ' images')

    # Create a DataFrame
    df = pd.DataFrame({
        'Image': image_list,
        'TPR': TPR,
        'FPR': FPR,
        'F-Score': F_SCORE,
        'IOU Score': IOU_SCORE,
        'Sensitivity': SENSITIVITY,
        'Specificity': SPECIFICITY,
        'Accuracy': ACCURACY,
        'TP': TP,
        'TN': TN,
        'FP': FP,
        'FN': FN
    })


    print('average TPR: ', np.average(TPR))
    print('average FPR: ', np.average(FPR))
    print('average F-Score: ', np.average(F_SCORE))
    print('average IOU: ', np.average(IOU))
    print('average sensitivity: ', np.average(SENSITIVITY))
    print('average specificity: ', np.average(SPECIFICITY))
    print('average accuracy: ', np.average(ACCURACY))
    print('average IOU score: ', np.average(IOU_SCORE))

    # Save to CSV
    df.to_csv(os.path.join(save_to, 'model_raw_metrics_super.csv'), index=False)
    print(f"Metrics saved to {save_to}/model_raw_metrics_super.csv")
        
