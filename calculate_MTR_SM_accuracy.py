import os
from calculate_class_accuracy import parse_cls_indx
import numpy as np


def calculate_MTR_SM_accuracy(results_path, save_outputs, cls_indx_path, steps):
    clip_name_model_pred = {}
    clip_name_gt_label = {}
    cls_indx_dic = parse_cls_indx(cls_indx_path)

    for step in steps:
        model_output_path = os.path.join(results_path +"predictions_step_" + str(step) + ".txt")

        with open(model_output_path) as fr:
            for line in fr:
                prediction, gt, clip_name = line.split()
                prediction = int(prediction)
                gt = int(gt)

                if clip_name not in clip_name_model_pred.keys():
                    clip_name_model_pred[clip_name] = [prediction]
                    clip_name_gt_label[clip_name] = gt
                else:
                    clip_name_model_pred[clip_name].append(prediction)

    tot_correct = tot_wrong = 0
    for clip_name, votes in clip_name_model_pred.items():

        counts = np.bincount(votes)
        majority_voting_label = np.argmax(counts)
        if majority_voting_label == clip_name_gt_label[clip_name]:
            tot_correct += 1
        else:
            tot_wrong += 1

    with open(save_outputs, 'a') as fw:
        print "Accuracy for steps: ", steps, ": ", 100*tot_correct/float(tot_correct+tot_wrong)
        fw.write("Accuracy for steps: " + str(steps) + ": " + str(100*tot_correct/float(tot_correct+tot_wrong)) + str("\n"))

if __name__ == '__main__':
    calculate_MTR_SM_accuracy("model_outputs/",
          "model_outputs/MTR-MM.txt",
          "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/class_index.txt")