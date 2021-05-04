# coding: utf-8

import argparse

import torch

import transforms as T
import utils
from cars_dataset import CarsDataset
from engine import evaluate
from model import DetectionModel


def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    parser = argparse.ArgumentParser("Evaluate Object Detection")
    parser.add_argument("--input", "-i", default="data/cars196/test", type=str, help="directory")
    parser.add_argument("--inmodel", "-im", default="model/fasterRCNN_state_dict.pth", type=str, help="input model")
    parser.add_argument("--batch", "-b", default=8, type=int, help="batch size")

    args = parser.parse_args()

    # load data
    dataset_test = CarsDataset(args.input, get_transform(train=False))
    num_classes = dataset_test.num_classes  # keep the number of classes including background(class 0)

    # define test data loaders
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get model
    detection = DetectionModel(num_classes)
    model = detection.get_detection_model()
    model.load_state_dict(torch.load(args.inmodel))

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # evaluate model
    evaluate(model, data_loader_test, device=device)


if __name__ == "__main__":
    main()
