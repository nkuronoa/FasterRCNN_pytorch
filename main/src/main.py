# coding: utf-8

import argparse

import torch

import transforms as T
import utils
from cars_dataset import CarsDataset
from engine import evaluate, train_one_epoch
from model import DetectionModel


def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def main():
    parser = argparse.ArgumentParser("Object Detection")
    parser.add_argument("--input", "-i", default="data/cars196/train", type=str, help="directory")
    parser.add_argument("--outmodel", "-om", default="model/fasterRCNN_state_dict.pth", type=str, help="output model")
    parser.add_argument("--train_rate", "-tr", default=0.6, type=float, help="rate of train dataset")
    parser.add_argument("--epoch", "-e", default=5, type=int, help="training epochs")
    parser.add_argument("--batch", "-b", default=2, type=int, help="batch size")

    args = parser.parse_args()

    assert (0 < args.train_rate) and (args.train_rate < 1)

    # load data
    dataset_train = CarsDataset(args.input, get_transform())
    dataset_val = CarsDataset(args.input, get_transform(train=False))
    num_classes = dataset_train.num_classes  # keep the number of classes including background(class 0)

    # split dataset in train and validation dataset
    train_num = int(len(dataset_train) * args.train_rate)
    indices = torch.randperm(len(dataset_train)).tolist()
    dataset_train = torch.utils.data.Subset(dataset_train, indices[:train_num])  # extract data
    dataset_val = torch.utils.data.Subset(dataset_val, indices[train_num:])  # extract data

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # get model
    detection = DetectionModel(num_classes)
    model = detection.get_detection_model()

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(args.epoch):
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        # update the learning rate
        lr_scheduler.step()
        # evaluate on the test dataset
        evaluate(model, data_loader_val, device=device)

        print("epoch {} end".format(epoch + 1))

    # save model
    torch.save(model.state_dict(), args.outmodel)


if __name__ == "__main__":
    main()
