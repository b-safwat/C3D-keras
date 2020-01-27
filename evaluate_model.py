import numpy as np
from models import c3d_model
from keras.optimizers import SGD
import time
import cv2
import os
from calculate_class_accuracy import preprocess, get_images


def evaluate_model(path_to_model, test_list, ds_path, batch_size, mean, std, save_file_path,
                   stack_size, step, input_shape):
    with open(test_list) as fr:
        lines = fr.readlines()

    # lines = lines[:100]

    num_batches = int(np.ceil(len(lines) / float(batch_size)))
    # input_shape = (112, 112, stack_size, 3)
    weight_decay = 0.005
    nb_classes = 18

    model = c3d_model.c3d_model(input_shape, weight_decay, nb_classes)
    lr = 0.005
    sgd = SGD(lr=lr, momentum=0.9, nesterov=True)
    model.load_weights(path_to_model)
    # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    mean = np.load(mean)
    std = np.load(std)
    print "total number of steps:", num_batches
    model_output = []

    for batch_idx in range(num_batches):
        stime = time.time()
        start = batch_idx * batch_size
        end = start + batch_size
        batch_lines = lines[start:end]

        batch_images, batch_labels, clip_paths = get_images(ds_path, batch_lines, False, stack_size, step)
        batch_images = preprocess(batch_images, mean, std)
        batch_images = np.transpose(batch_images, (0, 2, 3, 1, 4))

        batch_predictions = model.predict(batch_images, batch_size)
        batch_predictions = list(map(lambda x: x.argmax(), batch_predictions))

        for pred, correct_label, clip_path in zip(batch_predictions, batch_labels, clip_paths):
            model_output += [str(pred) + " " +  str(correct_label) + " " + clip_path + "\n"]

        print "step:", batch_idx, "took {} seconds".format(time.time() - stime)

    with open(save_file_path, 'w') as fw:
        fw.writelines(model_output)

if __name__ == '__main__':
    mean = "/home/bassel/PycharmProjects/C3D-keras/c3d_saved_files/oa18_mean.npy"
    std = "/home/bassel/PycharmProjects/C3D-keras/c3d_saved_files/oa18_std.npy"
    resize_img = False
    model_names = {1:"model-006-0.420537-0.368528.h5", 2:"model-006-0.568602-0.415453.h5",
                   3:"model-005-0.582834-0.438714.h5", 4:"model-005-0.652434-0.453479.h5",
                   6:"model-003-0.505680-0.398867.h5", 12:"model-010-0.990824-0.540251.h5"}
    for step in [1, 2, 3, 4, 6]:

        img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec'
        test_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_"\
                    +str(step)+"/testlist_stacks.txt"

        batch_size = 20
        stack_size = 10
        input_shape = (224, 224, stack_size, 3)
        path_to_model_weights = "results/oa18_step_" + str(step) + model_names[step]
        # path_to_model_weights = 'model-010-0.990824-0.540251.h5'
        evaluate_model(path_to_model_weights, test_file, img_path, batch_size, mean, std,
                       "model_outputs/predictions_step_"+str(step)+".txt", stack_size, step,
                       input_shape)