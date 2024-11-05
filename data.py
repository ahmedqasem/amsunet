from __future__ import print_function

import os
import glob
import shutil
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from skimage.transform import resize
import skimage.transform as trans
import imageio
from skimage import io, transform


def adjustData(img,mask,flag_multi_class,num_class):
    if(flag_multi_class):
        img = img / 255
        mask = mask[:,:,:,0] if(len(mask.shape) == 4) else mask[:,:,0]
        new_mask = np.zeros(mask.shape + (num_class,))
        for i in range(num_class):
            #for one pixel in the image, find the class in mask and convert it into one-hot vector
            #index = np.where(mask == i)
            #index_mask = (index[0],index[1],index[2],np.zeros(len(index[0]),dtype = np.int64) + i) if (len(mask.shape) == 4) else (index[0],index[1],np.zeros(len(index[0]),dtype = np.int64) + i)
            #new_mask[index_mask] = 1
            new_mask[mask == i,i] = 1
        new_mask = np.reshape(new_mask,(new_mask.shape[0],new_mask.shape[1]*new_mask.shape[2],new_mask.shape[3])) if flag_multi_class else np.reshape(new_mask,(new_mask.shape[0]*new_mask.shape[1],new_mask.shape[2]))
        mask = new_mask
    elif(np.max(img) > 1):
        img = img / 255
        mask = mask /255
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
    return (img,mask)



def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict,image_color_mode = "grayscale",
                    mask_color_mode = "grayscale",image_save_prefix  = "image",mask_save_prefix  = "mask",
                    flag_multi_class = False,num_class = 2,save_to_dir = None,target_size = (256,256),seed = 1):
    '''
    can generate image and mask at the same time
    use the same seed for image_datagen and mask_datagen to ensure the transformation for image and mask is the same
    if you want to visualize the results of generator, set save_to_dir = "your path"
    '''
    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = image_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = image_save_prefix,
        seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = mask_color_mode,
        target_size = target_size,
        batch_size = batch_size,
        save_to_dir = save_to_dir,
        save_prefix  = mask_save_prefix,
        seed = seed)
    train_generator = zip(image_generator, mask_generator)
    for (img,mask) in train_generator:
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        yield (img,mask)

def testGenerator(test_path, target_size=(256, 256), flag_multi_class=False, as_gray=True):
    """
    Loads all images from a specified folder, processes them, and yields them one by one with the file name.

    Parameters:
        test_path (str): The directory path where test images are stored.
        target_size (tuple): The desired size of each image (default is 256x256).
        flag_multi_class (bool): Indicates if multi-class labels are used (default is False).
        as_gray (bool): Determines whether to load images in grayscale (default is True).

    Yields:
        tuple: Processed image and original file name.
    """
    image_files = sorted([f for f in os.listdir(test_path) if f.endswith(('.png', '.jpg', '.jpeg'))])

    for img_name in image_files:
        img_path = os.path.join(test_path, img_name)
        img = io.imread(img_path, as_gray=as_gray)
        img = img / 255.0
        img = transform.resize(img, target_size)
        img = np.reshape(img, img.shape + (1,)) if not flag_multi_class else img
        img = np.reshape(img, (1,) + img.shape)
        
        yield img, img_name


def geneTrainNpy(image_path,mask_path,flag_multi_class = False,num_class = 2,image_prefix = "image",mask_prefix = "mask",image_as_gray = True,mask_as_gray = True):
    image_name_arr = glob.glob(os.path.join(image_path,"%s*.png"%image_prefix))
    image_arr = []
    mask_arr = []
    for index,item in enumerate(image_name_arr):
        img = io.imread(item,as_gray = image_as_gray)
        img = np.reshape(img,img.shape + (1,)) if image_as_gray else img
        mask = io.imread(item.replace(image_path,mask_path).replace(image_prefix,mask_prefix),as_gray = mask_as_gray)
        mask = np.reshape(mask,mask.shape + (1,)) if mask_as_gray else mask
        img,mask = adjustData(img,mask,flag_multi_class,num_class)
        image_arr.append(img)
        mask_arr.append(mask)
    image_arr = np.array(image_arr)
    mask_arr = np.array(mask_arr)
    return image_arr,mask_arr


def labelVisualize(num_class,color_dict,img):
    img = img[:,:,0] if len(img.shape) == 3 else img
    img_out = np.zeros(img.shape + (3,))
    for i in range(num_class):
        img_out[img == i,:] = color_dict[i]
    return img_out / 255


def saveResult2(save_path, npyfile, file_names, flag_multi_class=False, num_class=2):
    for i, (item, file_name) in enumerate(zip(npyfile, file_names)):
        img = labelVisualize(num_class, COLOR_DICT, item) if flag_multi_class else item[:, :, 0]
        # Convert the image to uint8 (0-255 range) for saving as PNG
        img = (img * 255).astype(np.uint8)
        # Save the image with the original file name
        save_name = os.path.join(save_path, file_name)
        io.imsave(save_name, img)

''' Operations Functions '''

def copy_image(path, destination, dest_img_type):
    # get all folders in path
    folders = os.listdir(path)
    no_of_folder = len(folders)
    counter = 0
    print('Copying ', no_of_folder, 'images from ', path, '\n')
    for folder in folders:
        src_img = path + '/' + folder + '/image.jpg'
        print('Copying  ', src_img, '\n')
        final_file_name = destination + folder + dest_img_type
        shutil.copy(src_img, final_file_name)
        counter += 1
    print(counter, ' images copied successfully')


def copy_label(path, destination, dest_img_type):
    # get all folders in path
    folders = os.listdir(path)
    no_of_folder = len(folders)
    counter = 0
    print('Copying ', no_of_folder, 'images from ', path, '\n')
    for folder in folders:
        src_img = path + '/' + folder + '/mask.jpg'
        print('Copying  ', src_img, '\n')
        final_file_name = destination + folder + dest_img_type
        shutil.copy(src_img, final_file_name)
        counter += 1
    print(counter, ' images copied successfully')


# function to add padding around image to make it a square
def pad_image(img, target_size, target_folder='', save_name='', save=False):
    # convert loaded image into numpy arrays
    original_img = img_to_array(img)
    # create an empty square canvas
    canvas = np.zeros((target_size, target_size))
    pasted_image = np.zeros((target_size, target_size))
    temp = np.squeeze(original_img)
    paste(pasted_image, temp, (0, 0))

    old_img = np.squeeze(original_img) / 255
    empty_canvas = np.squeeze(canvas) / 255
    new_img = pasted_image / 255

    # old = old_img / 255
    # empty = empty_canvas / 255
    # new = new_img / 255

    # save image
    if save:
        save_path = target_folder + save_name
        imageio.imwrite(save_path, pasted_image)
        print(save_name + ' padded and saved to ' + target_folder)
    return new_img

def pad_image2(img, target_size, start=0, end=0, target_folder='', save_name='', save=False):
    # convert loaded image into numpy arrays
    original_img = img_to_array(img)
    # create an empty square canvas
    canvas = np.zeros((target_size, target_size))
    pasted_image = np.zeros((target_size, target_size))
    temp = np.squeeze(original_img)
    paste(pasted_image, temp, (start, end))

    old_img = np.squeeze(original_img) / 255
    empty_canvas = np.squeeze(canvas) / 255
    new_img = pasted_image / 255

    # old = old_img / 255
    # empty = empty_canvas / 255
    # new = new_img / 255

    # save image
    if save:
        save_path = target_folder + save_name
        imageio.imwrite(save_path, pasted_image)
        print(save_name + ' padded and saved to ' + target_folder)
    return new_img


# Get and resize train images and masks
def get_data(path, im_height, im_width, train=True, re_size=True):
    # get specific image names and save into ids variable
    ids = next(os.walk(path + 'image/'))[2]
    # create an empty multi-dimentional numpy array to load the base images
    # X contains our base images
    X = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    if train:
        # create an empty multi-dimentional numpy array to load the masks
        y = np.zeros((len(ids), im_height, im_width, 1), dtype=np.float32)
    print('Getting and resizing images ... ')

    # tqdm notebook will show the progress
    # iterate to load the images into the previously created arrays
    for n, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        if n == 2:
            print('n is: ', n, 'id_ is: ', id_)
        # Load images
        img = load_img(path + 'image/' + id_, color_mode='grayscale')
        # convert loaded images into numpy arrays
        x_img = img_to_array(img)
        if n == 2:
            print('img before resize = ', x_img.shape)
        # resize the loaded images into the target size
        if re_size:
            # x_img = resize(x_img, (128, 128, 1), mode='constant', preserve_range=True)
            x_img = resize(x_img, (im_height, im_width, 1), mode='constant', preserve_range=True)
        if n == 2:
            print('img after resize: ', x_img.shape)
        if ((n % 10) == 0):
            print('loaded ', n, ' images')

        # Load masks
        if train:
            msk = load_img(path + 'label/' + id_, grayscale=True)
            mask = img_to_array(msk)
            if re_size:
                # mask = resize(mask, (128, 128, 1), mode='constant', preserve_range=True)
                mask = resize(mask, (im_height, im_width, 1), mode='constant', preserve_range=True)

        # Save images
        X[n, ..., 0] = x_img.squeeze() / 255
        if train:
            y[n] = mask / 255
    print('Done!')
    if train:
        return X, y
    else:
        return X

def save_predictions(preds, target_folder):
    for i in range(len(preds)):
        save_path = target_folder + 'image' + str(i) + '.png'
        imageio.imwrite(save_path, preds[i])
        print(str(i) + ' saved to ' + target_folder)
    print(len(preds), ' predictions saved to ', target_folder)

def save_predictions_l(preds, target_folder, file_name):
    for i in range(len(preds)):
        save_path = target_folder + 'image' + file_name[i]
        imageio.imwrite(save_path, preds[i])
        print(str(i) + ' saved to ' + target_folder)
    print(len(preds), ' predictions saved to ', target_folder)