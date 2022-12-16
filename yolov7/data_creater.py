import os
import glob
import argparse
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--test_split")
    args = parser.parse_args()
    paths = list(glob.glob(args.data_path + "/" + "*.png"))
    train, val = train_test_split(paths, test_size=float(args.test_split))
    training_string = ""
    val_string = ""
    print(len(train), len(val))
    for path in train:
        name = path.split("/")[-1][:-4]
        training_string += name + "\n"
    with open("train.txt", "w") as f:
        f.write(training_string)

    for path in val:
        name = path.split("/")[-1][:-4]
        val_string += name + "\n"
    with open("val.txt", "w") as f:
        f.write(val_string)
