import random
import numpy as np
import matplotlib.pyplot as plt


def describe_data(dataset, name):
    print('length of ' + name + ' is:', len(dataset), ' images')
    print('type of ' + name + ' is:', type(dataset))
    print('shape of ' + name + ' ', dataset.shape)
    print('\n')


def transform_image(img, target_width, target_height):
    pass


def paste_slices(tup):
    ''' https://stackoverflow.com/questions/7115437/how-to-embed-a-small-numpy-array-into-a-predefined-block-of-a-large-numpy-arra '''
    pos, w, max_w = tup
    wall_min = max(pos, 0)
    wall_max = min(pos + w, max_w)
    block_min = -min(pos, 0)
    block_max = max_w - max(pos + w, max_w)
    block_max = block_max if block_max != 0 else None
    return slice(wall_min, wall_max), slice(block_min, block_max)


def paste(wall, block, loc):
    ''' https://stackoverflow.com/questions/7115437/how-to-embed-a-small-numpy-array-into-a-predefined-block-of-a-large-numpy-arra '''
    loc_zip = zip(loc, block.shape, wall.shape)
    wall_slices, block_slices = zip(*map(paste_slices, loc_zip))
    wall[wall_slices] = block[block_slices]


def plot_sample(X, y, preds, binary_preds, ix=None):
    if ix is None:
        ix = random.randint(0, len(X))

    has_mask = y[ix].max() > 0

    fig, ax = plt.subplots(2, 3, figsize=(20, 10))
    # original image
    ax[0][0].imshow(X[ix, ..., 0], cmap='gray')
    ax[0][0].set_title('Original Image')
    # original mask
    ax[0][1].imshow(y[ix].squeeze(), cmap='gray')
    ax[0][1].set_title('Original Mask')
    # combined
    ax[0][2].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[0][2].contour(y[ix].squeeze(), colors='r', linewidth=1)
    ax[0][2].set_title('Original Image Localized')

    # predicted image
    ax[1][0].imshow(X[ix, ..., 0], cmap='gray')
    ax[1][0].set_title(' Image')
    # original mask
    ax[1][1].imshow(preds[ix].squeeze(), cmap='gray')
    ax[1][1].set_title('Predicted Mask')
    # combined
    ax[1][2].imshow(X[ix, ..., 0], cmap='gray')
    if has_mask:
        ax[1][2].contour(preds[ix].squeeze(), colors='r', linewidth=1)
    ax[1][2].set_title('Original Image Localized')

    plt.show()

    # fig, ax = plt.subplots(1, 4, figsize=(20, 10))
    # ax[0].imshow(X[ix, ..., 0], cmap='gray')
    # if has_mask:
    #     ax[0].contour(y[ix].squeeze(), colors='r', levels=[0.5])
    # ax[0].set_title('Original Image')
    #
    # ax[1].imshow(y[ix].squeeze(), cmap='gray')
    # ax[1].set_title('Original Mask')
    #
    # ax[2].imshow(preds[ix].squeeze(), vmin=0, vmax=1)
    # if has_mask:
    #     ax[2].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    # ax[2].set_title('Salt Predicted')
    #
    # ax[3].imshow(binary_preds[ix].squeeze(), vmin=0, vmax=1)
    # if has_mask:
    #     ax[3].contour(y[ix].squeeze(), colors='k', levels=[0.5])
    # ax[3].set_title('Salt Predicted binary');
    # plt.show()
