import numpy as np
import os
from torch.utils.data import Dataset
import cv2
import torch

class MyDataset(Dataset):
    def __init__(self, txt_path='annotations.txt', img_dir='data', transform=None, character_set=None):
        self.transform = transform
        #self.H, self.W, self.C = 50, 100, 3
        #self.H, self.W, self.C = 24, 94, 3
        self.H, self.W, self.C = 48, 144, 3
        self.len_label = 8 #max labels of every plate

        self.character_set = character_set
        self.num_classes = len(self.character_set)  # as the network output

        lines = open(os.path.join(img_dir, txt_path)).readlines()
        self.images = np.zeros((len(lines), self.C, self.H, self.W), dtype=np.float32)
        self.labels = np.zeros((len(lines), self.len_label), dtype=np.uint8)
        self.label_lenghts = np.zeros((len(lines), ), dtype=np.uint8)

        for i, line in enumerate(lines):
            img, label = line.strip().split()
            img = cv2.imread(os.path.join(img_dir, img))
            #img = img.permute(2, 0, 1)  #channels go first for torch tensor
            img = img.transpose(2,0,1) #channel go first for torch in numpy
            img = img / 255.0
            self.images[i, :, :, :] = img
            self.labels[i] = self.label_to_net_output_format(label)
            self.label_lenghts[i] = len(label)

    def label_to_net_output_format(self, label):
        # label_coded = np.zeros((self.len_label, self.num_classes), dtype=np.uint8)
        label_coded = []

        # for i, c in enumerate(label):
        #     pos = self.character_set.find(c)
        #     c_coded = [0] * self.num_classes
        #     c_coded[pos] = 1
        #     label_coded[i] = c_coded

        for i, c in enumerate(label):
            pos = self.character_set.find(c)
            label_coded.extend([pos])

        if len(label_coded) < self.len_label:
            pos_space = 100 #pad no matter with what
            label_coded.extend([pos_space] * (self.len_label - len(label_coded)))

        return label_coded

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        img = self.images[index]
        target = self.labels[index]
        target_len = self.label_lenghts[index]

        # if self.transform is not None:
        #     img = self.transform(img)

        #target = target.reshape((self.len_label * self.num_classes))
        img = torch.from_numpy(img)
        target = torch.from_numpy(target)
        #nr_timesteps = 16 = self.nr_digits*2, nr of timesteps from ocrnet
        return img, target, 16, target_len

#d = MyDataset(img_dir='/home/vadim/Downloads/reviewed_plates/train_data_lmdb/train/')
# for i, (input, target) in enumerate(d):
#     print('-------------')
#     print(target.shape)