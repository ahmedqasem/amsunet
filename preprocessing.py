from data import *


# pad_image2(img, target_size, start=0, end=0, target_folder='', save_name='', save=False):

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
mask_path = 'square_2048/label/'
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

    