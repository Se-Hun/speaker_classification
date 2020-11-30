import os
import argparse

def build_vocab(fns):
    import pandas as pd
    train_df = pd.read_csv(fns["input"]["train"], sep='\t')
    test_df = pd.read_csv(fns["input"]["test"], sep='\t')
    dev_df = pd.read_csv(fns["input"]["dev"], sep='\t')

    # label coverage check
    _train_set = set(train_df['label'].unique().tolist())
    _test_set = set(test_df['label'].unique().tolist())
    _dev_set = set(dev_df['label'].unique().tolist())

    # validation
    assert len(_test_set - _train_set) <= 0, "labels in test set are not in train"
    assert len(_dev_set - _train_set) <= 0, "labels in dev set are not in train"

    # building vocab
    label_vocab = ['<PAD>'] + [x for x in list(sorted(list(_train_set)))]

    # dumping vocab file
    label_vocab_fn = fns["output"]["label_vocab"]
    with open(label_vocab_fn, 'w', encoding='utf-8') as f:
        for idx, label in enumerate(label_vocab):
            print("{}\t{}".format(label, idx), file=f)
        print("[Label] vocab is dumped at ", label_vocab_fn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turn_change",
                        help="What task do you want ?")
    args = parser.parse_args()

    task = args.task

    # data_folder = os.path.join("./", task + "_temp", "run")
    data_folder = os.path.join("./", task, "run")

    fns = {
        "input": {
            "train" : os.path.join(data_folder, "train.tsv"),
            "dev" : os.path.join(data_folder, "dev.tsv"),
            "test" : os.path.join(data_folder, "test.tsv")
        },
        "output": {
            "label_vocab" : os.path.join(data_folder, "label.vocab")
        }
    }

    build_vocab(fns)