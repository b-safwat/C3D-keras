import threading

import numpy as np
import cv2
import glob
import os
import pdb


class ThreaddatasetMeanCalculator (threading.Thread):
    def __init__(self, threadID, ds_dir, ds_train_lst, stack_size, mean=None, new_w_h_size=224):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.ds_dir = ds_dir
        self.ds_train_lst = ds_train_lst
        self.sum=0
        self.count=0
        self.std_sum=0
        self.mean=mean
        self.stack_size=stack_size
        self.new_w_h_size = new_w_h_size

    def run(self):
        if self.mean is None:
            self.sum, self.count = calculate_thread_dataset_mean(self.ds_dir, self.ds_train_lst,
                                                                 self.stack_size,
                                                                 no_step_size= True,
                                                                 new_w_h_size=self.new_w_h_size)
        else:
            self.std_sum, self.count = calculate_thread_dataset_std(self.ds_dir, self.ds_train_lst, self.stack_size,
                                                                    self.mean, no_step_size=True,
                                                                    new_w_h_size=self.new_w_h_size)


def calculate_thread_dataset_mean(ds_dir, ds_train_lst, stack_size, new_w_h_size=112,
                                  no_step_size=False):
    sum = np.zeros((new_w_h_size, new_w_h_size, 3))
    count = 0

    for line in ds_train_lst:
        vid_path = line.strip().split(',')[0].split(' ')[0].replace(".mp4", "")\
            .replace(".avi", "").replace("HandStandPushups", "HandstandPushups")

        start_pos = int(line.split()[1])

        if len(line.split()) == 4:
            step_size = int(line.split()[2])
        else:
            step_size = 1

        num_frames = len(os.listdir(os.path.join(ds_dir, vid_path)))
        stack_frames = []

        for i in range(stack_size):
            frame_idx = start_pos + i*step_size
            img = cv2.imread(os.path.join(ds_dir, vid_path, "{:05}.jpg".format(frame_idx)))

            if img is None:
                img = cv2.imread(os.path.join(ds_dir, vid_path, "{:04}.jpg".format(frame_idx)))

            if img is None:
                print os.path.join(ds_dir, vid_path, "{:05}.jpg".format(frame_idx))
                break

            height, width, _ = img.shape

            if height != new_w_h_size or width != new_w_h_size:
                if (width > height):
                    scale = float(new_w_h_size) / float(height)
                else:
                    scale = float(new_w_h_size) / float(width)

                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

                height, width, _ = img.shape

                crop_y = int((height - new_w_h_size) / 2)
                crop_x = int((width - new_w_h_size) / 2)

                # stack_frames.append(img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :])
                sum += img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]
                # std_sum += (img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]-mean).sum(axis=0).sum(axis=0)
            else:
                # stack_frames.append(img)
                sum+=img
                # std_sum += ((img-mean)*(img-mean)).sum(axis=0).sum(axis=0)

        count += stack_size

            # stack_frames.append(img)

        # stack_frames = np.array(stack_frames)
        # sum += stack_frames
        # count += 1
    # mean/=float(count)
    # print mean
    # return sum, std_sum, count
    return sum, count

def calculate_thread_dataset_std(ds_dir, ds_train_lst, stack_size, mean, new_w_h_size=112, no_step_size=False):
    std_sum = np.zeros((3))
    count = 0

    for line in ds_train_lst:
        vid_path = line.strip().split(',')[0].split(' ')[0].replace(".mp4", "")\
            .replace(".avi", "").replace("HandStandPushups", "HandstandPushups")

        start_pos = int(line.split()[1])

        if not no_step_size:
            step_size = int(line.split()[2])
        else:
            step_size = 1

        # lbl = int(line.split()[2])
        num_frames = len(os.listdir(os.path.join(ds_dir, vid_path)))
        stack_frames = []

        for i in range(stack_size):
            frame_idx = start_pos + i * step_size
            img = cv2.imread(os.path.join(ds_dir, vid_path, "{:05}.jpg".format(frame_idx)))
            # img = cv2.resize(img, (new_w_h_size, new_w_h_size))
            if img is None:
                img = cv2.imread(os.path.join(ds_dir, vid_path, "{:04}.jpg".format(frame_idx)))
            if img is None:
                print os.path.join(ds_dir, vid_path, "{:05}.jpg".format(frame_idx))
            height, width, _ = img.shape

            if height != new_w_h_size or width != new_w_h_size:
                if (width > height):
                    scale = float(new_w_h_size) / float(height)
                else:
                    scale = float(new_w_h_size) / float(width)

                img = cv2.resize(img, (0, 0), fx=scale, fy=scale)

                height, width, _ = img.shape

                crop_y = int((height - new_w_h_size) / 2)
                crop_x = int((width - new_w_h_size) / 2)

                std_sum += (img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]-mean).sum(axis=0).sum(axis=0)
            else:
                std_sum += ((img-mean)*(img-mean)).sum(axis=0).sum(axis=0)

        count += stack_size

    return std_sum, count

def calculate_dataset_mean(ds_dir, ds_list, stack_size, mean=None, new_w_h_size=112, n_threads=12):
    with open(ds_list) as f:
        lines = f.readlines()

    block_size = len(lines)/n_threads

    threads=[]

    for i in range(n_threads):
        start = i*block_size
        end = start + block_size
        if i == n_threads-1:
            end = len(lines)

        if mean is None:
            threads.append(ThreaddatasetMeanCalculator(i, ds_dir, lines[start:end], stack_size,
                                                       new_w_h_size=new_w_h_size))
        else:
            threads.append(ThreaddatasetMeanCalculator(i, ds_dir, lines[start:end], stack_size,
                                                       mean, new_w_h_size=new_w_h_size))

        threads[-1].start()

    if mean is None:
        sum = np.zeros((new_w_h_size, new_w_h_size, 3))
    else:
        std = np.zeros((3))
    count = 0

    for i in range(n_threads):
        threads[i].join()

        if mean is None:
            sum += threads[i].sum
        else:
            std += threads[i].std_sum

        count += threads[i].count

    if mean is None:
        mean=sum.sum(axis=0).sum(axis=0)/float(count*new_w_h_size*new_w_h_size)
        return mean
    else:
        std=np.sqrt(std/float(count*new_w_h_size*new_w_h_size))
        return std


if __name__ == '__main__':

    # mean = calculate_dataset_mean(
    #     "/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec",
    #     "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_1/trainlist_stacks.txt",
    #     stack_size=10, new_w_h_size=224)
    # print(mean)
    # np.save("means_stds/oa18_rounded5_mean.npy", mean)
    #
    # std = calculate_dataset_mean(
    #     "/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec",
    #     "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_1/trainlist_stacks.txt",
    #     stack_size=10, new_w_h_size=224, mean=mean)
    #
    # np.save("means_stds/oa18_rounded5_std.npy", std)
    # # print(std)
    # print (np.load("means_stds/oa18_rounded5_mean.npy"),
    #        np.load("means_stds/oa18_rounded5_std.npy"))

    for step in [1, 2, 3, 4, 6, 12]:
        mean = calculate_dataset_mean(
            "/home/bassel/data/UCF101/frms/",
            "/home/bassel/data/UCF101/lbl/method1/multi_steps/step_" + str(step) + "/trainlist_stacks.txt",
            stack_size=10, new_w_h_size=224)
        print(mean)
        np.save("means_stds/oa18_rounded5_mean_step_"+str(step)+".npy", mean)

        std = calculate_dataset_mean(
            "/home/bassel/data/office-actions/office_actions_19/short_clips/rounded_5sec",
            "/home/bassel/data/office-actions/office_actions_19/short_clips/labels/multi_steps/rounded_5sec/step_"+str(step)+"/trainlist_stacks.txt",
            stack_size=10, new_w_h_size=224, mean=mean)

        np.save("means_stds/oa18_rounded5_std_step_"+str(step)+".npy", std)
        # print(std)
        print (np.load("means_stds/oa18_rounded5_mean.npy"),
               np.load("means_stds/oa18_rounded5_std.npy"))