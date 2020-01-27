import numpy as np
import _5_fine_tune_c3d as finetune_c3d_main


## side train
# mean = np.load("oa18_side_unstabilized_dataset_calculated_mean.npy")
# std = np.load("oa18_side_unstabilized_dataset_calculated_std.npy")
# resize_img = False
#
# img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/unstabilized_resized_frms_112/'
# train_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_train_stack_list.txt'
# test_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_test_stack_list.txt'
#
# num_classes = 18
# batch_size = 32
# epochs = 16
# input_shape = (112, 112, 16, 3)
#
# train_c3d_main.main(mean, std, resize_img, img_path, train_file, test_file, num_classes, batch_size, epochs,
#                     input_shape, "results_side/")

## side fine_tune

mean = np.load("means_stds/oa18_side_unstabilized_dataset_calculated_mean.npy")
std = np.load("means_stds/oa18_side_unstabilized_dataset_calculated_std.npy")
resize_img = False

img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/unstabilized_resized_frms_112/'
train_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_train_stack_list.txt'
test_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/side_only_test_stack_list.txt'

num_classes = 18
batch_size = 32
epochs = 16
input_shape = (112, 112, 16, 3)

finetune_c3d_main.main(mean, std, resize_img, img_path, train_file, test_file, num_classes, batch_size, epochs,
                    input_shape, "results/saved_models/oa11_weights_c3d.h5", "results_side_finetune/")

########################################################################################################################
## front train
# mean = np.load("oa18_side_unstabilized_dataset_calculated_mean.npy")
# std = np.load("oa18_side_unstabilized_dataset_calculated_std.npy")
# resize_img = False
#
# img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/unstabilized_resized_frms_112/'
# train_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_train_stack_list.txt'
# test_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_test_stack_list.txt'
#
# num_classes = 18
# batch_size = 32
# epochs = 16
# input_shape = (112, 112, 16, 3)
#
# train_c3d_main.main(mean, std, resize_img, img_path, train_file, test_file, num_classes, batch_size, epochs,
#                     input_shape, "results_front/")

## front fine_tune

mean = np.load("means_stds/oa18_front_unstabilized_dataset_calculated_mean.npy")
std = np.load("means_stds/oa18_front_unstabilized_dataset_calculated_std.npy")
resize_img = False

img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/unstabilized_resized_frms_112/'
train_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_train_stack_list.txt'
test_file = '/home/bassel/data/office-actions/office_actions_19/short_clips/labels/front_only_test_stack_list.txt'

num_classes = 18
batch_size = 32
epochs = 16
input_shape = (112, 112, 16, 3)

finetune_c3d_main.main(mean, std, resize_img, img_path, train_file, test_file, num_classes, batch_size, epochs,
                    input_shape, "results/saved_models/oa11_weights_c3d.h5", "results_front_finetune/")
