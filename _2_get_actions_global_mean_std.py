import threading

import numpy as np
import cv2
import glob
import os
import pdb


class ThreaddatasetMeanCalculator (threading.Thread):
    def __init__(self, threadID, ds_dir, ds_train_lst, new_w_h_size, mean):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.ds_dir = ds_dir
        self.ds_train_lst = ds_train_lst
        self.sum=0
        self.count=0
        self.std_sum=0
        self.mean = mean
        self.new_w_h_size = new_w_h_size
    def run(self):
        self.sum, self.std_sum, self.count = calculate_thread_dataset_mean(self.ds_dir, self.ds_train_lst,
                                                                           self.new_w_h_size, self.mean)


def calculate_thread_dataset_mean(ds_dir, ds_train_lst, new_w_h_size, mean=None):
    sum = 0
    std_sum = 0
    count = 0
    # mean = [ 91.86219342,  98.20372432, 103.10237516]

    for line in ds_train_lst:
        line = line.replace("HandStandPushups", "HandstandPushups").replace("HandStandWalking", "HandstandWalking")

        if len(line.split(',')) > 1:# i.e. oa18
            vid_name = line.strip().split(',')[0].replace(".mp4", "").replace(".avi", "")
        else: # i.e. ucf101
            vid_name = line.strip().split(' ')[0].replace(".mp4", "").replace(".avi", "").split('/')[1]

        lst_frames = os.listdir(os.path.join(ds_dir, vid_name))
        lst_frames.sort()
        lst_frames = [os.path.join(ds_dir, vid_name, frame) for frame in lst_frames]

        for frame_path in lst_frames:
            img = cv2.imread(frame_path)

            if img is None:
                print os.path.join(ds_dir, frame_path)

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

                img = img[crop_y:crop_y + new_w_h_size, crop_x:crop_x + new_w_h_size, :]
                sum += img.sum()

                if mean is not None:
                    diff = img-mean
                    std_sum += (diff * diff).sum()
            else:
                # stack_frames.append(img)
                sum+=img.sum()

                if mean is not None:
                    diff = img-mean
                    std_sum += (diff * diff).sum()

        count += len(lst_frames)*new_w_h_size*new_w_h_size*3

    return sum, std_sum, count


def calculate_dataset_mean(ds_dir, ds_list, new_w_h_size=224, mean=None, n_threads=12):
    with open(ds_list) as f:
        lines = f.readlines()

    block_size = len(lines)/n_threads

    threads=[]

    for i in range(n_threads):
        start = i*block_size
        end = start + block_size
        if i == n_threads-1:
            end = len(lines)
        threads.append(ThreaddatasetMeanCalculator(i, ds_dir, lines[start:end], new_w_h_size, mean))
        threads[-1].start()

    sum_images = 0
    std_sum = 0
    count = 0

    for i in range(n_threads):
        threads[i].join()
        sum_images += threads[i].sum
        if mean is not None:
            std_sum += threads[i].std_sum
        count += threads[i].count

    mean=sum_images/float(count)

    if mean is not None:
        std_sum=np.sqrt(std_sum/float(count))

    return mean, std_sum


if __name__ == '__main__':
    mean_dataset = None
    mean_dataset, std = calculate_dataset_mean(
        "/home/bassel/data/UCF101/frms/",
        "/home/bassel/data/UCF101/lbl/trainlist01.txt")
    np.save("c3d_saved_files/ucf101_mean.npy", mean_dataset)
    # mean_dataset = np.load("../c3d_saved_files/oa18_mean.npy")
    mean_dataset, std = calculate_dataset_mean(
        "/home/bassel/data/UCF101/frms/",
        "/home/bassel/data/UCF101/lbl/trainlist01.txt",
        mean=mean_dataset)

    np.save("c3d_saved_files/ucf101_std.npy", std)
    print(mean_dataset, std)
    # import pdb
    # pdb.set_trace()
