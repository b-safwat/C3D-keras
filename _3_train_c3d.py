# -*- coding:utf-8 -*-
from keras.callbacks import ModelCheckpoint

from models import c3d_model
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from schedules import onetenth_4_8_12
import numpy as np
import random
import cv2
import os
import random
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt
import keras


def plot_history(history, result_dir):
    plt.plot(history.history['acc'], marker='.')
    plt.plot(history.history['val_acc'], marker='.')
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.grid()
    plt.legend(['acc', 'val_acc'], loc='lower right')
    plt.savefig(os.path.join(result_dir, 'model_accuracy.png'))
    plt.close()

    plt.plot(history.history['loss'], marker='.')
    plt.plot(history.history['val_loss'], marker='.')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid()
    plt.legend(['loss', 'val_loss'], loc='upper right')
    plt.savefig(os.path.join(result_dir, 'model_loss.png'))
    plt.close()


def save_history(history, result_dir):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)

    with open(os.path.join(result_dir, 'result.txt'), 'w') as fp:
        fp.write('epoch\tloss\tacc\tval_loss\tval_acc\n')
        for i in range(nb_epoch):
            fp.write('{}\t{}\t{}\t{}\t{}\n'.format(
                i, loss[i], acc[i], val_loss[i], val_acc[i]))
        fp.close()


def calculate_image_resize(img, new_w_h_size):
    height, width, _ = img.shape

    if (width > height):
        scale = float(new_w_h_size) / float(height)
    else:
        scale = float(new_w_h_size) / float(width)

    img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

    height, width, _ = img.shape

    crop_y = int((height - new_w_h_size) / 2)
    crop_x = int((width - new_w_h_size) / 2)
    return img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]

    # crop_x = random.randint(0, 15)
    # crop_y = random.randint(0, 58)
    #
    # image = cv2.resize(image, (171, 128))
    # if val
    # return image[8:120, 30:142, :]
    # return image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]



def process_batch(lines, img_path, stack_size, train=True, use_step_size=False, input_shape=(224, 224)):
    num = len(lines)
    batch = np.zeros((num, stack_size, input_shape[0],
                      input_shape[1], 3), dtype='float32')
    labels = np.zeros(num, dtype='int')
    for i in range(num):
        path = lines[i].split(' ')[0]
        # path = path.split("/")[-1]
        path = path.replace("HandStandPushups", "HandstandPushups")
        label = lines[i].split(' ')[-1]
        starting_pos = lines[i].split(' ')[1]
        if use_step_size:
            step_size = int(lines[i].split(' ')[2])
        else:
            step_size = 1

        label = label.strip('\n')
        label = int(label)-1
        starting_pos = int(starting_pos)-1
        imgs = os.listdir(os.path.join(img_path,path))
        imgs.sort(key=str.lower)
        if train:
            is_flip = random.randint(0, 1)
            last_existing_j = 0

            for j in range(stack_size):

                if starting_pos + j*step_size >= len(imgs):
                    #duplicate_last_img
                    # img = imgs[starting_pos + (last_existing_j) * step_size]
                    # exit(1)
                    ## Method 1: repeat from beginning
                    img = imgs[(starting_pos + j*step_size)%len(imgs)]
                else:
                    img = imgs[starting_pos + j*step_size]
                    last_existing_j = j

                image = cv2.imread(os.path.join(img_path, path, img))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                if is_flip == 1:
                    image = cv2.flip(image, 1)

                if resize_img:
                    image = calculate_image_resize(image, 224)
                    batch[i][j][:][:][:] = image
                    # image = cv2.resize(image, (171, 128))
                    # batch[i][j][:][:][:] = image[crop_x:crop_x + 112, crop_y:crop_y + 112, :]

                else:
                    batch[i][j][:][:][:] = image
            labels[i] = label
        else:
            last_existing_j = 0

            for j in range(stack_size):
                try:
                    if starting_pos + j * step_size >= len(imgs):
                        # duplicate_last_img
                        # img = imgs[starting_pos + (last_existing_j) * step_size]
                        # exit(1)
                        ## Method 1
                        img = imgs[(starting_pos + j*step_size)%len(imgs)]
                    else:
                        img = imgs[starting_pos +  j*step_size]
                        last_existing_j = j
                except:
                    print("Exception @ img:", starting_pos + j, len(imgs))
                image = cv2.imread(os.path.join(img_path, path, img))
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if resize_img:
                    image = calculate_image_resize(image, 224)
                    batch[i][j][:][:][:] = image
                    # image = cv2.resize(image, (171, 128))
                    # batch[i][j][:][:][:] = image[8:120, 30:142, :]

                else:
                    batch[i][j][:][:][:] = image
            labels[i] = label
    return batch, labels

mean = None
std = None
resize_img = False

def preprocess(inputs):
    # mean = [ 90.48464058,  97.57829463, 102.25286134]
    # std = [70.59445308, 69.45308667, 71.02353467]

#     inputs[..., 0] -= mean[0] # 99.9
#     inputs[..., 1] -= mean[1] # 92.1
#     inputs[..., 2] -= mean[2] # 82.6
# #
#     inputs[..., 0] /= std[0] # 65.8
#     inputs[..., 1] /= std[1] # 62.3
#     inputs[..., 2] /= std[2] # 60.3
    inputs -= mean
    inputs /= std

    # inputs /=255.
    # inputs -= 0.5
    # inputs *=2.
    return inputs


def generator_train_batch(train_txt, batch_size, num_classes, img_path, stack_size, use_step_size, input_shape):
    ff = open(train_txt, 'r')
    lines = ff.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num/batch_size)):
            a = i*batch_size
            b = (i+1)*batch_size
            x_train, x_labels = process_batch(new_line[a:b],img_path, stack_size, train=True,
                                              use_step_size=use_step_size, input_shape=input_shape)
            x = preprocess(x_train)
            y = np_utils.to_categorical(np.array(x_labels), num_classes)
            x = np.transpose(x, (0,2,3,1,4))
            yield x, y


def generator_val_batch(val_txt,batch_size,num_classes,img_path, stack_size, use_step_size):
    f = open(val_txt, 'r')
    lines = f.readlines()
    num = len(lines)
    while True:
        new_line = []
        index = [n for n in range(num)]
        random.shuffle(index)
        for m in range(num):
            new_line.append(lines[index[m]])
        for i in range(int(num / batch_size)):
            a = i * batch_size
            b = (i + 1) * batch_size
            y_test,y_labels = process_batch(new_line[a:b], img_path, stack_size,
                                            train=False, use_step_size=use_step_size)
            x = preprocess(y_test)
            x = np.transpose(x,(0,2,3,1,4))
            y = np_utils.to_categorical(np.array(y_labels), num_classes)
            yield x, y


def main(_mean, _std, _resize_img, _img_path, _train_file, _test_file, _num_classes, _batch_size, _epochs,
         _input_shape, stack_size, results_dir='results/', use_step_size=True):
    '''
    Notes:
        1- make sure the labels are 0 based or 1 based: >> Line 64 in this file <<
        2- adjut the parameters/values in the main method
    :return:
    '''
    global mean, std, resize_img
    mean = _mean
    std = _std
    resize_img = _resize_img

    f1 = open(_train_file, 'r')
    f2 = open(_test_file, 'r')
    lines = f1.readlines()

    f1.close()
    train_samples = len(lines)
    lines = f2.readlines()

    f2.close()
    val_samples = len(lines)

    model = c3d_model.c3d_model(_input_shape, 0.005, _num_classes)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    checkpoint = ModelCheckpoint(results_dir+'model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', verbose=1, monitor='val_loss',
                                 save_best_only=True, mode='auto')

    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    history = model.fit_generator(generator_train_batch(_train_file, _batch_size, _num_classes,
                                                        _img_path, stack_size, use_step_size, _input_shape),
                                  steps_per_epoch=train_samples // _batch_size,
                                  epochs=_epochs,
                                  callbacks=[onetenth_4_8_12(lr), checkpoint],
                                  validation_data=generator_val_batch(_test_file,
                                                                      _batch_size, _num_classes,
                                                                      _img_path, stack_size, use_step_size),
                                  validation_steps=val_samples // _batch_size,
                                  verbose=1)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)

    plot_history(history, results_dir)
    save_history(history, results_dir)

    # model.save_weights(results_dir+'/oa18'+str(num_classes)+'_weights_c3d.h5')


if __name__ == '__main__':
    # mean = np.load("means_stds/oa18_side_unstabilized_dataset_calculated_mean.npy")
    # std = np.load("means_stds/oa18_side_unstabilized_dataset_calculated_std.npy")
    # # resize_img = False
    # resize_img = True
    #
    # # img_path = '/home/bassel/data/oa_kinetics/frms/'
    # # train_file = '/home/bassel/data/oa_kinetics/lbls/actions_stack_list.txt'
    # # test_file = '/home/bassel/data/oa_kinetics/lbls/oa18_test_stack_mapped_oa11_kinetics.txt'
    # img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/unstabilized_resized_frms_112'
    # train_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_train_stack_list.txt'
    # test_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_test_stack_list.txt'
    #
    # num_classes = 18
    # batch_size = 32
    # epochs = 16
    # stack_size = 16
    # input_shape = (112,112,stack_size,3)
    #
    # main(mean, std, resize_img, img_path, train_file, test_file, num_classes, batch_size, epochs, input_shape, stack_size,
    #      "results_side_batch_normalized/")

    #########################################################################################################

    # mean = np.load("/home/bassel/PycharmProjects/multi_action_recognition/c3d_saved_files/oa18_mean.npy")
    # std = np.load("/home/bassel/PycharmProjects/multi_action_recognition/c3d_saved_files/oa18_std.npy")
    # resize_img = False
    # img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec'
    # num_classes = 18
    # batch_size = 12
    # epochs = 10
    # stack_size = 10
    # input_shape = (224, 224, stack_size, 3)
    #
    # for step in [1, 2, 3, 4, 6, 12]:
    #     train_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels' \
    #                  '/multi_steps/rounded_5sec/step_'+str(step)+'/trainlist_stacks.txt'
    #     test_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels' \
    #                 '/multi_steps/rounded_5sec/step_'+str(step)+'/testlist_stacks.txt'
    #
    #     main(mean, std, resize_img, img_path, train_file, test_file, num_classes,
    #          batch_size, epochs, input_shape, stack_size,
    #          results_dir="results/oa18_step_"+str(step)+"/")

    #########################################################################################################

    mean = np.load("/home/bassel/PycharmProjects/C3D-keras/c3d_saved_files/ucf101_mean.npy")
    std = np.load("/home/bassel/PycharmProjects/C3D-keras/c3d_saved_files/ucf101_std.npy")
    resize_img = False
    img_path = '/home/bassel/data/UCF101/frms'
    num_classes = 101
    batch_size = 10
    epochs = 10
    stack_size = 10
    input_shape = (224, 224, stack_size, 3)

    for step in [1, 2, 3, 4, 6, 12]:
        train_file = "/home/bassel/data/UCF101/lbl/method1/multi_steps/step_"+str(step)+"/trainlist_stacks.txt"
        test_file = "/home/bassel/data/UCF101/lbl/method1/multi_steps/step_"+str(step)+"/testlist_stacks.txt"

        main(mean, std, resize_img, img_path, train_file, test_file, num_classes,
             batch_size, epochs, input_shape, stack_size,
             results_dir="results/ucf101_step_"+str(step)+"/")