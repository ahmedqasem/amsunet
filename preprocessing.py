from data import *
from general import paste
import cv2

"""
This script provides functions for processing, padding, cropping, and copying images to prepare them for analysis or machine learning tasks. 
It includes functions to pad images to a specified size, copy specific images to a folder, and crop images and masks into square patches.

each code chunk can be used independently as needed, comment/uncomment the unwanted section to avoid working on data that
has been already processed
"""


''' function to pad the image to make it divisible by 2'''
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

# load all the images and insert a for loop to padd all the images in the dessired folder 
# to the target size if need to do more than one
target_size = 1024
img = cv2.imread('', 0) # replace '' with image name (including path)
pad_image2(img, target_size, start=0, end=0, target_folder='', save_name='', save=False)


''' function to move specific images to a folder '''
def copy_cropped_images(target_folder, source_folder, from_folder, criteria):
    counter = 0
    for i in from_folder:
        if i in criteria:
            copy_file = source_folder+'/'+i
            target_file = target_folder+'/'+i
            print('copying ', i, ' to ', target_file)
            shutil.copy(copy_file, target_file)
            counter += 1
    print(counter, ' images copied.')

target_folder = './data/data/image'
source_folder = './data/square_1024_cropped/image'
images_to_copy = os.listdir(source_folder)
criteria = set(os.listdir('data/data/label'))

copy_cropped_images(target_folder, source_folder, images_to_copy, criteria)


''' new file '''
''' uncomment below to crop images into squares '''
# crop images suqares and save them in folder
target_size = 2457
target_size = 1024
img_path = 'square_2048/image/'
photos_to_change = 500
images = os.listdir(img_path)

print('found ', len(images), ' images in ', img_path)
print('first image is: ', images[0])
n=0
for image in images:
    img = load_img(img_path + image, color_mode='grayscale')
    target_path = 'data/square_1024_cropped/image/'
    old = np.squeeze(img_to_array(img)) / 255
    # pad_image(old, target_size, target_path, image, save=True)

    # 4 images = 1024*1024
    pad_image2(old, target_size, start=0, end=0, target_folder=target_path, save_name=image.split('.')[0]+'Q1.png', save=True)
    pad_image2(old, target_size, start=0, end=-1024, target_folder=target_path, save_name=image.split('.')[0]+'Q2.png', save=True)
    pad_image2(old, target_size, start=-1024, end=0, target_folder=target_path, save_name=image.split('.')[0]+'Q3.png', save=True)
    pad_image2(old, target_size, start=-1024, end=-1024, target_folder=target_path, save_name=image.split('.')[0]+'Q4.png', save=True)

    n +=1
    if n >= photos_to_change:
        break

''' uncomment below to crop masks into squares '''
# crop masks suqares and save them in folder
mask_path = 'square_2048/label/' # change folder for images 
masks = os.listdir(mask_path)

print('found ', len(masks), ' images in ', mask_path)
print('first mask is: ', masks[0])
n=0
for mask in masks:
    msk = load_img(mask_path + mask, color_mode='grayscale')
    target_path = 'data/square_1024_cropped/label/'
    old = np.squeeze(img_to_array(msk)) / 255
    # pad_image(old, target_size, target_path, mask, save=True)

    # 4 masks size 1024
    pad_image2(old, target_size, start=0, end=0, target_folder=target_path, save_name=mask.split('.')[0] + 'Q1.png', save=True)
    pad_image2(old, target_size, start=0, end=-1024, target_folder=target_path, save_name=mask.split('.')[0] + 'Q2.png', save=True)
    pad_image2(old, target_size, start=-1024, end=0, target_folder=target_path, save_name=mask.split('.')[0] + 'Q3.png', save=True)
    pad_image2(old, target_size, start=-1024, end=-1024, target_folder=target_path, save_name=mask.split('.')[0] + 'Q4.png', save=True)
    n += 1
    if n >= photos_to_change:
        break

    