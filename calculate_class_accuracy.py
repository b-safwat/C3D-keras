import os
import random

import time

import cv2
import numpy as np
from keras.optimizers import SGD

from models import c3d_model


def parse_cls_indx(path):

    with open(path) as fr:
        lines = fr.readlines()

    dic = {}

    for l in lines:
        cls, name = l.split(',')
        dic[int(cls)-1] = name.strip()
        dic[name.strip()] = int(cls)-1

    return dic

def calculate_image_resize(img, new_w_h_size):
    height, width, _ = img.shape

    if (width > height):
        scale = float(new_w_h_size) / float(height)
    else:
        scale = float(new_w_h_size) / float(width)

    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    height, width, _ = img.shape

    # crop_y = random.randint(0, int((height - new_w_h_size) / 2))
    crop_y = int((height - new_w_h_size) / 2)
    # crop_x = random.randint(0, int((width - new_w_h_size) / 2))
    crop_x = int((width - new_w_h_size) / 2)
    return img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]


def get_images(ds_path, lines, resize_img, stack_size=16, step=1):
    num = len(lines)
    batch = np.zeros((num, stack_size, 224, 224, 3), dtype='float32')
    labels = np.zeros(num, dtype='int')
    clip_names = []

    for i in range(num):
        path = lines[i].split(' ')[0]
        path = path.split("/")[-1]
        path = path.replace("HandStandPushups", "HandstandPushups")

        label = lines[i].split(' ')[-1]
        label = label.strip('\n')
        label = int(label) - 1

        symbol = lines[i].split(' ')[1]
        symbol = int(symbol) - 1

        imgs = os.listdir(os.path.join(ds_path, path))
        imgs.sort(key=str.lower)
        clip_names.append(path)

        for j in range(stack_size):
            if symbol+j*step >= len(imgs):
                print(symbol+j*step)
            img = imgs[symbol + j*step]
            image = cv2.imread(os.path.join(ds_path, path, img))
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if resize_img:
                image = calculate_image_resize(image, 224)
                batch[i][j][:][:][:] = image
                # image = cv2.resize(image, (171, 128))
                # batch[i][j][:][:][:] = image[8:120, 30:142, :]
            else:
                batch[i][j][:][:][:] = image

        labels[i] = label

    return batch, labels, clip_names


def calculate_class_accuracies(cls_indx_dic, ground_truth, predictions, save_file_path):
    dic_correct={}
    dic_incorrect={}

    for i in range(0, 18):
        dic_incorrect[i]=0
        dic_correct[i]=0

    for pred, ground in zip(predictions, ground_truth):
        if pred == ground:
            dic_correct[ground]+=1
        else:
            dic_incorrect[ground]+=1

    tot_correct = 0
    tot_wrong = 0

    with open(save_file_path, 'w') as fw:
        for i in range(18):
            tot_correct += dic_correct[i]
            tot_wrong += dic_incorrect[i]
            print (cls_indx_dic[i] + ":", dic_correct[i] / float(dic_correct[i] + dic_incorrect[i]))
            fw.write(cls_indx_dic[i] + ": " + str(dic_correct[i] / float(dic_correct[i] + dic_incorrect[i])) + "\n")

def preprocess(inputs, mean, std):
    # inputs[..., 0] -= mean[0]
    # inputs[..., 1] -= mean[1]
    # inputs[..., 2] -= mean[2]
    #
    # inputs[..., 0] /= std[0]
    # inputs[..., 1] /= std[1]
    # inputs[..., 2] /= std[2]
    inputs -= mean
    inputs /= std
    return inputs


def calculate_class_accuracy(path_to_model, test_list, ds_path, batch_size,
                             cls_indx, mean, std, save_file_path, input_shape):
    cls_indx_dic = parse_cls_indx(cls_indx)

    with open(test_list) as fr:
        lines = fr.readlines()

    num_batches = int(np.ceil(len(lines)/float(batch_size)))
    # input_shape=(112, 112, 16, 3)
    weight_decay=0.005
    nb_classes=18

    model = c3d_model.c3d_model(input_shape, weight_decay, nb_classes)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    model.load_weights(path_to_model)

    predictions = []
    ground_truth_labels = []

    mean = np.load(mean)
    std = np.load(std)
    print "total number of steps:", num_batches
    for batch_idx in range(num_batches):
        stime = time.time()
        start = batch_idx * batch_size
        end = start + batch_size
        batch_lines = lines[start:end]

        batch_images, batch_labels, _ = get_images(ds_path, batch_lines, True)
        batch_images = preprocess(batch_images, mean, std)
        batch_images = np.transpose(batch_images, (0, 2, 3, 1, 4))

        batch_predictions = model.predict(batch_images, batch_size)
        batch_predictions = list(map(lambda x: x.argmax(), batch_predictions))
        predictions += batch_predictions
        ground_truth_labels += batch_labels.tolist()
        print "step:", batch_idx, "took {} seconds".format(time.time()-stime)


    calculate_class_accuracies(cls_indx_dic, ground_truth_labels, predictions, save_file_path)

if __name__ == '__main__':
    stack_size = 10
    input_shape = (224, 224, stack_size, 3)

    calculate_class_accuracy("/home/bassel/PycharmProjects/C3D-keras/results_finetune_data_augmented_4times/augmented_oa18_fine_tuned_oa11_kinetics_weights_c3d.h5",
                        "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/test_stack_list.txt",
                        "/home/bassel/data/office-actions/office_actions_19/short_clips/resized_frms_224",
                        64,
                        "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt",
                        "means_stds/oa18_augmented_unstabilized_dataset_calculated_mean.npy",
                        "means_stds/oa18_augmented_unstabilized_dataset_calculated_std.npy",
                        "/home/bassel/PycharmProjects/C3D-keras/results_finetune_data_augmented_4times/class_accuracy.txt",
                        input_shape)