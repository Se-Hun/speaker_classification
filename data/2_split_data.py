import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import prepare_dir

def split_data(fns, task, seed):
    # read data file
    in_fn = fns["input"]
    data = pd.read_csv(in_fn, sep='\t', index_col=0)

    # train/dev/test split --> Ratio : 60 / 20 / 20
    train_data, dev_data = train_test_split(data, test_size=0.4, random_state=seed)
    dev_data, test_data = train_test_split(dev_data, test_size=0.5, random_state=seed)

    # dump files
    to_train_fn = fns["output"]["train"]
    to_dev_fn = fns["output"]["dev"]
    to_test_fn = fns["output"]["test"]

    train_data.to_csv(to_train_fn, sep="\t")
    print("[Train] {} data is dumped at  ".format(task), to_train_fn)

    dev_data.to_csv(to_dev_fn, sep="\t")
    print("[Dev] {} data is dumped at  ".format(task), to_dev_fn)

    test_data.to_csv(to_test_fn, sep="\t")
    print("[Test] {} data is dumped at  ".format(task), to_test_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turn_change",
                        help="What task do you want ?")
    parser.add_argument("--seed", help='seed', type=int, default=42)
    args = parser.parse_args()

    task = args.task
    seed = args.seed

    # in_folder = os.path.join("./", task + "_temp")
    # to_folder = os.path.join("./", task + "_temp", "run")
    in_folder = os.path.join("./", task)
    to_folder = os.path.join("./", task, "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join(in_folder, "data.tsv"),
        "output": {
            "train" : os.path.join(to_folder, "train.tsv"),
            "dev" : os.path.join(to_folder, "dev.tsv"),
            "test" : os.path.join(to_folder, "test.tsv")
        }
    }

    split_data(fns, task, seed)