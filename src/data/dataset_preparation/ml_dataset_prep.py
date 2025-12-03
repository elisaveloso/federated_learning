import cv2
import glob
import numpy as np
import os
import imgaug.augmenters as iaa

from PIL import Image
from tqdm import tqdm 
from uuid import uuid4

def is_corrupted(image_path):
    is_corrupted = False
    try:
        img = Image.open(image_path) # open the image file
        img.verify() # verify that it is, in fact an image
        img = Image.open(image_path) 
        img.transpose(Image.FLIP_LEFT_RIGHT)
        img.close()
    except:
        is_corrupted = True
    return is_corrupted


def remove_corrupted_images(dataset_dir):
    for dirpath, dirnames, filenames in os.walk(dataset_dir):
        for filename in [f for f in filenames if f.endswith(".jpg")]:
            image_path = os.path.join(dirpath, filename)
            if is_corrupted(image_path):
                os.remove(image_path)


##########   def rename_images


def get_batches(list_to_split, batch_size):
    for i in range(0, len(list_to_split), batch_size):
        yield list_to_split[i:i + batch_size]


def augment_images(n_images, dataset_dir, class_type, class_qty):
    aug_list = []
    aug_list.append(iaa.Multiply((0.5, 1.5)))
    aug_list.append(iaa.SaltAndPepper(0.001))
    aug_list.append(iaa.GaussianBlur(sigma=(0.0, 1.2)))
    aug_list.append(iaa.MultiplyBrightness((0.8, 1.2)))
    aug_list.append(iaa.GammaContrast((0.8, 1.2)))
    aug_list.append(iaa.LinearContrast((0.8, 1.2)))
    aug_list.append(iaa.imgcorruptlike.Brightness(severity=1))
    aug_list.append(iaa.imgcorruptlike.Spatter(severity=1))
    aug_list.append(iaa.Affine(shear=(-5, 5)))

    images_dir = os.path.join(dataset_dir, class_type)

    # collect common image extensions to avoid missing images with other extensions
    images_list = []
    for ext in ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tif', '*.tiff'):
        images_list.extend(glob.glob(os.path.join(images_dir, ext)))
    images_list = sorted(images_list)

    # keep only original (non-augmented) images
    original_images_list = [file for file in images_list if 'aug' not in os.path.basename(file)]

    # remove any old augmented images (if present with 'aug' in filename)
    old_aug_images_list = [file for file in images_list if 'aug' in os.path.basename(file)]
    for file_path in old_aug_images_list:
        try:
            os.remove(file_path)
        except OSError:
            pass

    # defensive checks
    if n_images <= 0:
        return

    if not original_images_list:
        raise ValueError(f"No original images found in '{images_dir}' to augment. \n"
                         f"Ensure the directory exists and contains image files (jpg/png/...)")

    # ensure we can sample (allow replacement) and make n_images divisible by number of augmenters
    q, mod = divmod(n_images, len(aug_list))
    if mod != 0:
        n_images = n_images + (len(aug_list) - mod)

    # allow sampling with replacement when requesting more samples than available originals
    selected_images = np.random.choice(original_images_list, n_images, replace=True)

    batch_size = max(1, n_images // len(aug_list))
    image_batches = list(get_batches(list(selected_images), batch_size))

    with tqdm(total=n_images, desc='Augmentating') as pbar:
        for i, aug in enumerate(aug_list):
            # some batches might be shorter depending on rounding; guard index
            if i >= len(image_batches):
                break
            image_batch = image_batches[i]
            images = []
            for image_path in image_batch:
                img = cv2.imread(image_path)
                if img is None:
                    continue
                images.append(img)
            if not images:
                continue
            images_aug = aug(images=images)
            for image in images_aug:
                img_id = uuid4().hex[:8]
                aug_img_name = f'{class_type}_aug_{img_id}.jpg'
                aug_image_path = os.path.join(images_dir, aug_img_name)
                cv2.imwrite(aug_image_path, image)
            pbar.update(len(images_aug))


def balance_dataset(dataset_dir, classes_list):
    max_proportion = 1
    counter_dict = {}
    for class_type in classes_list:
        class_dir = os.path.join(dataset_dir, class_type)
        files = os.listdir(class_dir)
        original_files = [file for file in files if 'aug' not in file]
        counter_dict[class_type] = len(original_files)
        
    print(counter_dict)
    max_qty = max(counter_dict.values())
    for class_type, class_qty in counter_dict.items():
        proportion = max_qty/class_qty
        if proportion > max_proportion:
            n_images = int((max_qty/max_proportion) - class_qty)
            augment_images(n_images, dataset_dir, class_type, class_qty)



def compress_dataset(dataset_dir):
    os.system("7z a {}.7z {}".format(dataset_dir, dataset_dir))