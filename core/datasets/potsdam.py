import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
import cv2
import pickle


class potsdamDataSet(data.Dataset):
    def __init__(self,
        data_root,
        data_list,
        max_iters=None,
        num_classes=6,
        split="train",
        transform=None,
        ignore_label=255,
        debug=False,):

        self.split = split
        self.NUM_CLASS = num_classes
        self.data_root = data_root
        self.data_list = []
        with open(data_list, "r") as handle:
            content = handle.readlines()
        self.img_ids = [i_id.strip() for i_id in content]

        # load_data
        if max_iters is not None:
            self.label_to_file, self.file_to_label = pickle.load(open(osp.join(data_root, "potsdam_label_info.p"), "rb"))
            self.img_ids = []
            SUB_EPOCH_SIZE = 3000
            tmp_list = []
            ind = dict()
            for i in range(self.NUM_CLASS):
                ind[i] = 0
            for e in range(int(max_iters/SUB_EPOCH_SIZE)+1):
                cur_class_dist = np.zeros(self.NUM_CLASS)
                for i in range(SUB_EPOCH_SIZE):
                    if cur_class_dist.sum() == 0:
                        dist1 = cur_class_dist.copy()
                    else:
                        dist1 = cur_class_dist/cur_class_dist.sum()
                    w = 1/np.log(1+1e-2 + dist1)
                    w = w/w.sum()
                    c = np.random.choice(self.NUM_CLASS, p=w)
                    if ind[c] > (len(self.label_to_file[c])-1):
                        np.random.shuffle(self.label_to_file[c])
                        ind[c] = ind[c]%(len(self.label_to_file[c])-1)
                    c_file = self.label_to_file[c][ind[c]]
                    tmp_list.append(c_file)
                    ind[c] = ind[c]+1
                    cur_class_dist[self.file_to_label[c_file]] += 1

            self.img_ids = tmp_list

        for name in self.img_ids:
            self.data_list.append(
                {
                    "img": os.path.join(self.data_root, "images/%s" % name),
                    "label": os.path.join(self.data_root, "labels/%s" % name),
                    "name": name,
                }
            )
        
        if max_iters is not None:
            self.data_list = self.data_list * int(np.ceil(float(max_iters) / len(self.data_list)))
        
        print('length of potsdam', len(self.data_list))

        self.id_to_trainid = {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5}
        self.trainid2name = {
            0: "Clutter background",
            1: "Imprevious surfaces",
            2: "Car",
            3: "Tree",
            4: "Low vegetation",
            5: "Building"
        }
        self.transform = transform
        self.ignore_label = ignore_label
        self.debug = debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        if self.debug:
            index = 0
        datafiles = self.data_list[index]
        image = Image.open(datafiles["img"]).convert('RGB')
        label = np.array(Image.open(datafiles["label"]),dtype=np.float32)
        name = datafiles["name"]
        label_copy = self.ignore_label * np.ones(label.shape, dtype=np.float32)
        for k, v in self.id_to_trainid.items():
            label_copy[label == k] = v
        label = Image.fromarray(label_copy)

        if self.transform is not None:
            image, label = self.transform(image, label)

        return image, label, name
