import Util as utl
import os
import sys
import shuffle_list as shuffle


def isValidFramesStack(clipPath, StartingFrame, NumStackingFrames, step=1):
    last_frame = StartingFrame + (NumStackingFrames-1)*step
    return os.path.exists(os.path.join(clipPath.strip(), "{:04}.jpg".format(last_frame)))

def stack_clips(ds_dir, dataset_txtfile, dataset_txtfile_output_path,
                NumStackingFrames = 16, overlapFrames=0, step=1):
    """
    Saves a zero based file with all the starts of each stack
    """
    if not os.path.isdir(utl.get_full_path_for_dir_containing_file(dataset_txtfile_output_path)):
        os.makedirs(utl.get_full_path_for_dir_containing_file(dataset_txtfile_output_path))

    with open(dataset_txtfile) as f:
        lines = f.readlines()

    NewGT = []

    cls_dic = '/home/bassel/data/UCF101/lbl/classInd.txt'

    with open(cls_dic) as f:
        cls_dic = {}

        for line in f.readlines():
            cls_dic[line.split()[1]] = int(line.split()[0])

    for line in lines:
        if len(line.split(',')) > 1:# i.e. oa18
            l = line.split(',')[0].replace(".mp4", "").replace(".avi", "")
            label = line.split(',')[1]
        else:# i.e. ucf101
            l = line.split(' ')[0].replace(".mp4", "").replace(".avi", "").split('/')[1]
            if len(line.split()) == 1: # i.e. testing
                label = cls_dic[line.split('/')[0]]
            else:
                label = line.split()[1]

        l = l.strip()
        # startingFrame = 1

        l = l.replace("HandStandPushups", "HandstandPushups")

        for startingFrame in range(1, step + 1):
            while True:
                if not isValidFramesStack(os.path.join(ds_dir, l), startingFrame, NumStackingFrames, step):
                    if os.path.exists(os.path.join(ds_dir, l, "{:04}.jpg".format(startingFrame))):
                            # and isValidFramesStack(os.path.join(ds_dir, l), startingFrame, NumStackingFrames/2, step):
                        NewGT += [l + " " + str(startingFrame) + " " + str(step) + " " + str(int(label)) + "\n"]
                    break
                NewGT += [l + " " + str(startingFrame) + " " + str(step) + " " + str(int(label)) + "\n"]
                startingFrame += NumStackingFrames*step

    with open(dataset_txtfile_output_path, 'w') as file_writer:
        file_writer.writelines(NewGT)


if __name__ == '__main__':
    for step in [1, 2, 3, 4, 6, 12]:
        stack_clips("/home/bassel/data/UCF101/frms/",
                    "/home/bassel/data/UCF101/lbl/trainlist01.txt",
                    "/home/bassel/data/UCF101/lbl/method1/multi_steps/step_"+str(step)+"/trainlist_stacks.txt",
                    NumStackingFrames=10, step=step)

        shuffle.shuffle_list("/home/bassel/data/UCF101/lbl/method1/multi_steps/step_"+str(step)+"/trainlist_stacks.txt")

        stack_clips("/home/bassel/data/UCF101/frms/",
                    "/home/bassel/data/UCF101/lbl/testlist01.txt",
                    "/home/bassel/data/UCF101/lbl/method1/multi_steps/step_"+str(step)+"/testlist_stacks.txt",
                    NumStackingFrames=10, step=step)

        # stack_clips(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
        # stack_clips("/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec",
        #             "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/trainlist.txt",
        #             "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_"+str(step)+"/trainlist_stacks.txt",
        #             NumStackingFrames=10, step=step)
        #
        # shuffle.shuffle_list("/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_"+str(step)+"/trainlist_stacks.txt")
        #
        # stack_clips("/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec",
        #             "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/testlist.txt",
        #             "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_"+str(step)+"/testlist_stacks.txt",
        #             NumStackingFrames=10, step=step)
