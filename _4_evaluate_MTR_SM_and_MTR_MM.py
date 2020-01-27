from evaluate_model import evaluate_model
from calculate_MTR_SM_accuracy import calculate_MTR_SM_accuracy

mean = "/home/bassel/PycharmProjects/C3D-keras/c3d_saved_files/oa18_mean.npy"
std = "/home/bassel/PycharmProjects/C3D-keras/c3d_saved_files/oa18_std.npy"
resize_img = False
model_names = {1: "model-006-0.420537-0.368528.h5", 2: "model-006-0.568602-0.415453.h5",
               3: "model-005-0.582834-0.438714.h5", 4: "model-005-0.652434-0.453479.h5",
               6: "model-003-0.505680-0.398867.h5", 12: "model-010-0.990824-0.540251.h5"}
for step in [2, 3, 4, 6, 12]:
    img_path = '/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec'
    test_file = "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_" \
                + str(step) + "/testlist_stacks.txt"

    batch_size = 20
    stack_size = 10
    input_shape = (224, 224, stack_size, 3)
    path_to_model_weights = "results/oa18_step_" + str(step) + "/" + model_names[step]
    # path_to_model_weights = 'model-010-0.990824-0.540251.h5'
    evaluate_model(path_to_model_weights, test_file, img_path, batch_size, mean, std,
                   "model_outputs/predictions_step_" + str(step) + ".txt", stack_size, step,
                   input_shape)
    calculate_MTR_SM_accuracy("model_outputs/", "model_outputs/MTR-MM.txt",
          "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt",
                              [step])

calculate_MTR_SM_accuracy("model_outputs/",
      "model_outputs/MTR-MM.txt",
      "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt",
                          [1, 2, 3, 4, 6, 12])