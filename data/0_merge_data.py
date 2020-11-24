import os
import json

from glob import glob

from common.utils import prepare_dir

def build_data(fns):
    data_dir = fns["input"]
    data_fns = glob(os.path.join(data_dir, "*.json"))

    to_fn = fns["output"]

    all_data = []
    for fn in data_fns:
        with open(fn, 'r', encoding='utf-8') as f:
            data = json.load(f)

            if data["id"] == "MDRW1900006926": # modify typo ....
                # utterances = data["document"][0]["utterance"]
                # new_utterances = []
                # for utterance in utterances:
                #     if utterance["speaker_id"] == "2️" or utterance["speaker_id"] == "2":
                #         utterance["speaker_id"] == "2"
                #     new_utterances.append(utterance)
                # data["document"][0]["utterance"] = new_utterances
                #
                # print("hi")
                continue

            all_data.append(data)

    print("Number of data : {}".format(len(all_data)))

    with open(to_fn, 'w', encoding='utf-8') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)
        print("All data is dumped at  ", to_fn)


if __name__ == '__main__':
    data_dir = os.path.join("./", "original")

    to_dir = os.path.join("./", "original-all")
    prepare_dir(to_dir)

    fns = {
        "input" : data_dir,
        "output" : os.path.join(to_dir, "data.json")
    }

    build_data(fns)