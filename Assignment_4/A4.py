# -*- coding: utf-8 -*-

import os
import numpy as np
import PIL.Image as Image

from scipy.io import loadmat
from sklearn.model_selection import train_test_split


DATA_PATH = "data/"
PIXEL_DEPTH = 255
NUM_LABELS = 10

OUT_HEIGHT = 64
OUT_WIDTH = 64
NUM_CHANNELS = 3
MAX_LABELS = 5

last_percent_reported = None


def read_data_file(file_name):
    file = open(file_name, 'rb')
    data = process_data_file(file)
    file.close()
    return data


def convert_imgs_to_array(img_array):
    rows = img_array.shape[0]
    cols = img_array.shape[1]
    chans = img_array.shape[2]
    num_imgs = img_array.shape[3]
    scalar = 1 / PIXEL_DEPTH
    # Note: not the most efficent way but can monitor what is happening
    new_array = np.empty(shape=(num_imgs, rows, cols, chans), dtype=np.float32)
    for x in range(0, num_imgs):
        # TODO reuse normalize_img here
        chans = img_array[:, :, :, x]
        # normalize pixels to 0 and 1. 0 is pure white, 1 is pure channel color
        norm_vec = (255-chans)*1.0/255.0
        # Mean Subtraction
        norm_vec -= np.mean(norm_vec, axis=0)
        new_array[x] = norm_vec
    return new_array


def convert_labels_to_one_hot(labels):
    labels = (np.arange(NUM_LABELS) == labels[:, None]).astype(np.float32)
    return labels


def process_data_file(file):
    data = loadmat(file)
    imgs = data['X']
    labels = data['y'].flatten()
    labels[labels == 10] = 0  # Fix for weird labeling in dataset
    labels_one_hot = convert_labels_to_one_hot(labels)
    img_array = convert_imgs_to_array(imgs)
    return img_array, labels_one_hot


def get_data_file_name(dataset):
    if dataset == "train":
        data_file_name = "train_32x32.mat"
    elif dataset == "test":
        data_file_name = "test_32x32.mat"
    elif dataset == "extra":
        data_file_name = "extra_32x32.mat"
    else:
        raise Exception('dataset must be either train, test or extra')
    return data_file_name


def create_svhn(dataset):
    data_file_name = get_data_file_name(dataset)
    data_file_pointer = os.path.join(DATA_PATH, data_file_name)

    if os.path.isfile(data_file_pointer):
        extract_data = read_data_file(data_file_pointer)
        return extract_data
    else:
        new_file = concatenate_path_file(DATA_PATH, data_file_name)
        return read_data_file(new_file)


def concatenate_path_file(path, filename):
    return path + filename


def get_expected_bytes(filename):
    if filename == "train_32x32.mat":
        byte_size = 182040794
    elif filename == "test_32x32.mat":
        byte_size = 64275384
    elif filename == "extra_32x32.mat":
        byte_size = 1329278602
    else:
        raise Exception("Invalid file name " + filename)
    return byte_size


def train_validation_split(train_dataset, train_labels):
    train_dataset, validation_dataset, train_labels, validation_labels = train_test_split(train_dataset, train_labels, test_size=0.1, random_state = 42)
    return train_dataset, validation_dataset, train_labels, validation_labels


def write_npy_file(data_array, lbl_array, data_set_name):
    np.save(os.path.join(DATA_PATH, data_set_name+'_imgs.npy'), data_array)
    print('Saving to %s_svhn_imgs.npy file done.' % data_set_name)
    np.save(os.path.join(DATA_PATH, data_set_name+'_labels.npy'), lbl_array)
    print('Saving to %s_svhn_labels.npy file done.' % data_set_name)


def load_svhn_data(data_type, data_set_name):
    # TODO add error handling here
    path = DATA_PATH + data_set_name
    imgs = np.load(os.path.join(path, data_set_name+'_'+data_type+'_imgs.npy'))
    labels = np.load(os.path.join(path, data_set_name+'_'+data_type+'_labels.npy'))
    return imgs, labels


def create_label_array(el):
    """[count, digit, digit, digit, digit, digit]"""
    num_digits = len(el)  # first element of array holds the count
    labels_array = np.ones([MAX_LABELS+1], dtype=int) * 10
    labels_array[0] = num_digits
    for n in range(num_digits):
        if el[n] == 10: el[n] = 0  # reassign 0 as 10 for one-hot encoding
        labels_array[n+1] = el[n]
    return labels_array


def create_img_array(file_name, top, left, height, width, out_height, out_width):
    img = Image.open(file_name)

    img_top = np.amin(top)
    img_left = np.amin(left)
    img_height = np.amax(top) + height[np.argmax(top)] - img_top
    img_width = np.amax(left) + width[np.argmax(left)] - img_left

    box_left = np.floor(img_left - 0.1 * img_width)
    box_top = np.floor(img_top - 0.1 * img_height)
    box_right = np.amin([np.ceil(box_left + 1.2 * img_width), img.size[0]])
    box_bottom = np.amin([np.ceil(img_top + 1.2 * img_height), img.size[1]])

    img = img.crop((box_left, box_top, box_right, box_bottom)).resize([out_height, out_width], Image.ANTIALIAS)
    pix = np.array(img)

    norm_pix = (255-pix)*1.0/255.0
    norm_pix -= np.mean(norm_pix, axis=0)
    return norm_pix


def generate_numpy_files():
    train_data, train_labels = create_svhn('train')
    train_data, valid_data, train_labels, valid_labels = train_validation_split(train_data, train_labels)

    write_npy_file(train_data, train_labels, 'train')
    write_npy_file(valid_data, valid_labels, 'valid')

    test_data, test_labels = create_svhn('test')
    write_npy_file(test_data, test_labels, 'test')
    print("Generate Numpy Files Done!!!")


if __name__ == '__main__':
    generate_numpy_files()

