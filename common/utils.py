"""
    Utilities
    Author : Sangkeun Jung (hugmanskj@gmail.com), 2019
"""
# ----------------------- Directory ---------------------------- #
import os


def prepare_dir(dir_name):
    if not os.path.exists(dir_name): os.makedirs(dir_name)

def exist_dir(dir_name):
    if not os.path.exists(dir_name):
        return False
    else:
        return True

# ----------------------- NLP Utilities ---------------------------- #
def load_vocab(fn):
    print("Vocab loading from {}".format(fn))

    vocab = {}
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.rstrip()
            symbol, _id = line.split('\t')
            vocab[symbol] = int(_id)

    return vocab


def simple_text_reader(fn):
    lines = []
    with open(fn, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.lstrip().rstrip()
            lines.append(line)
        print("[Text] data is loaded from {} -- {}".format(fn, len(lines)))
    return lines


def simple_json_reader(fn):
    import json
    with open(fn, 'r', encoding='utf-8') as f:
        data = json.load(f)
        print("[Json] data is loaded from {} -- {}".format(fn, len(data)))
    return data


## -------------------------------------- ML progress check --------------------------------------- ##
import numpy as np


class Monitor:
    def __init__(self):
        self.keys = {}
        self.storage = {}

        self.key_order = []

    def register(self, key):
        self.key_order.append(key)
        self.keys[key] = []  # averages
        self.storage[key] = []

    def add_value(self, key, value):
        self.keys[key].append(value)
        self.storage[key].append(value)

    def avg(self, key):
        return np.mean(self.keys[key])

    def reset_window(self):
        for key in self.keys.keys():
            self.keys[key] = []

    # print method #
    def print_screen(self, epoch, step, additional=None):
        items = []
        items.append('Epoch : {}'.format(epoch))
        items.append('Step : {}'.format(step))
        for key in self.key_order:
            items.append("{} : {:6.4f}".format(key, self.avg(key)))

        if additional != None:
            for a_key, a_value in additional.items():
                items.append("{} : {:6.4f}".format(a_key, a_value))

        line = " ".join(items)
        print(line)

    def dump_to_pandas(self, epoch, step, to_f, additional=None, init=False):
        items = []

        if init == True:
            # dump header
            items.append('epoch')
            items.append('step')
            for key in self.key_order:
                items.append(key)

            if additional != None:
                for a_key, a_value in additional.items():
                    items.append("{}".format(a_key))

            line = "\t".join(items)
            print(line, file=to_f)
            return

            # normal
        items.append(str(epoch))
        items.append(str(step))
        for key in self.key_order:
            items.append("{:6.4f}".format(self.avg(key)))

        if additional != None:
            for a_key, a_value in additional.items():
                items.append("{:6.4f}".format(a_value))

        line = "\t".join(items)
        print(line, file=to_f)
        to_f.flush()


## -------------------------------------- Pytorch utilities --------------------------------------- ##
import torch


def recursive_to_device(item, device):
    if type(item) == type(torch.tensor([])):
        return item.to(device)
    if type(item) == type({}):
        _item = {}
        for key, value in item.items():
            _item[key] = recursive_to_device(value, device)
        return _item


def prepare_device(hps):
    device = torch.device("cpu")  # default
    if torch.cuda.is_available():
        if hps.use_gpu:
            device = torch.device("cuda")
    return device


def is_gpu_available():
    return torch.cuda.is_available()


def load_model(model_fn, map_location=None):
    if map_location:
        return torch.load(model_fn, map_location=map_location)
    else:
        if torch.cuda.is_available():
            return torch.load(model_fn)
        else:
            return torch.load(model_fn, map_location='cpu')