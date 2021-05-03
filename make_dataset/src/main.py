# coding: utf-8
import argparse
import pathlib

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


def main():
    parser = argparse.ArgumentParser("Save Public Dataset")
    parser.add_argument("-datasets", default="cars196", type=str, choices=["cars196"], help="public dataset name for tensorflow-datasets")
    parser.add_argument("-output", default="data", type=str, help="dataset path")
    parser.add_argument("-mode", default="train", type=str, choices=["train", "test"], help="dataset path")

    args = parser.parse_args()

    ds, info = tfds.load(args.datasets, split=args.mode, with_info=True)

    # for cars196
    outputpath = pathlib.Path(args.output) / args.datasets / args.mode
    outputpath.mkdir(parents=True, exist_ok=True)

    with open(outputpath / "list.txt", "w") as f:
        bbox_array = None
        for i, item in enumerate(ds):
            if(i % 100 == 0):
                print(f"i:{i}")
            outputfolder = outputpath / "images" / str(item["label"].numpy())
            outputfolder.mkdir(parents=True, exist_ok=True)
            filepath = str(outputfolder / (str(i).zfill(5) + ".jpg"))
            tf.keras.preprocessing.image.save_img(filepath, item["image"])
            f.write("{}\n".format(filepath))
            bbox = np.reshape(item["bbox"].numpy(), [1, 4])  # tfds.feature.BBox(ymin, xmin, ymax, xmax)
            if bbox_array is None:
                bbox_array = bbox
            else:
                bbox_array = np.concatenate([bbox_array, bbox])

    np.savetxt(outputpath / "annotation.csv", bbox_array, delimiter=",")


if __name__ == "__main__":
    main()
