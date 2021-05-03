import os

import cv2
import numpy as np
import torch

from utils import load_list


class CarsDataset(object):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        self.image_paths = load_list(os.path.join(root, "list.txt"))
        self.bbox_array = np.loadtxt(os.path.join(root, "annotation.csv"), delimiter=",")

        #self.label_list = []
        # for i, image_path in enumerate(self.image_paths):
        #    dirpath = os.path.dirname(image_path)
        #    self.label_list.append(int(os.path.basename(dirpath))+1)  # background is always 0

        #self.__num_classes = len(set(self.label_list)) + 1
        self.__num_classes = 2

    def __getitem__(self, idx):
        # load images
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # get bounding boxes
        height, width = image.shape[:2]
        ymin = int(self.bbox_array[idx][0] * height)
        xmin = int(self.bbox_array[idx][1] * width)
        ymax = int(self.bbox_array[idx][2] * height)
        xmax = int(self.bbox_array[idx][3] * width)
        # one box per image
        boxes = [[xmin, ymin, xmax, ymax]]
        # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        #labels = torch.as_tensor([self.label_list[idx]], dtype=torch.int64)
        labels = torch.ones((1,), dtype=torch.int64)

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        #target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.image_paths)

    @property
    def num_classes(self):
        return self.__num_classes
