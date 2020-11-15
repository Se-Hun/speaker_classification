import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from common.utils import prepare_dir

def split_data(fns, task, seed):
    in_fn = fns["input"]
    to_train_fn = fns["output"]["train"]
    to_test_fn = fns["output"]["test"]

    train_data = []
    test_data = []

    data = pd.read_csv(in_fn, sep='\t', index_col=0)

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=seed)

    train_data.to_csv(to_train_fn, sep="\t")
    print("[Train] {} data is dumped at  ".format(task), to_train_fn)
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

    in_folder = os.path.join("./", task)
    to_folder = os.path.join("./", task, "run")
    prepare_dir(to_folder)

    fns = {
        "input": os.path.join(in_folder, "data.tsv"),
        "output": {
            "train" : os.path.join(to_folder, "train.tsv"),
            "test" : os.path.join(to_folder, "test.tsv")
        }
    }

    split_data(fns, task, seed)