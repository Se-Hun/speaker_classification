import os
import json
import argparse

import pandas as pd

from tqdm.auto import tqdm

from common.utils import prepare_dir

def build_turn_change_exmaple(origin_ex):
    id = origin_ex["id"]
    document = origin_ex["document"]
    assert (len(document) == 1), "Document has more than two data at {}".format(id)

    examples = []
    utterances = document[0]["utterance"]
    for u_idx in range(0, len(utterances) - 2, 3):
        first_utterance = utterances[u_idx]
        second_utterance = utterances[u_idx + 1]
        target_utterance = utterances[u_idx + 2]

        first_id = first_utterance["id"]
        second_id = second_utterance["id"]
        target_id = target_utterance["id"]

        first_text = first_utterance["form"].replace("\n", " ")
        second_text = second_utterance["form"].replace("\n", " ")
        target_text = target_utterance["form"].replace("\n", " ")

        label = 0 if second_utterance["speaker_id"] == target_utterance["speaker_id"] else 1  # speaker가 같으면 0, 틀리면 1

        # remove data using emoticon, uploading photos ect...
        if first_text == "" or second_text == "" or target_text == "":
            continue

        example = [[first_id, second_id], target_id, first_text + " " + second_text, target_text, label]
        examples.append(example)

    return examples

def build_turn_change_exmaple_document(origin_ex):
    id = origin_ex["id"]
    document = origin_ex["document"]
    assert (len(document) == 1), "Document has more than two data at {}".format(id)

    examples = []
    utterances = document[0]["utterance"]

    # text_store = ""
    text_ids = []
    prev_speaker_id = -1
    for u_idx, utterance in enumerate(utterances):
        id = utterance["id"]
        utterance_text = utterance["form"].replace("\n", " ")
        speaker_id = utterance["speaker_id"]

        if utterance_text == "":
            continue

        label = 0 if prev_speaker_id == speaker_id else 1 # 마지막 speaker와 현재 speaker가 같으면 0, 틀리면 1

        if u_idx == 0:
            # text_store = text_store + " " + utterance_text
            text_ids.append(id)
            prev_speaker_id = speaker_id
            continue

        example = [text_ids, id, label]
        # example = [text_store, utterance_text, label]
        examples.append(example)

        text_ids.append(id)
        # text_store = text_store + " " + utterance_text

    return examples


def buil_turn_change_example_previous(origin_ex):
    id = origin_ex["id"]
    document = origin_ex["document"]
    assert (len(document) == 1), "Document has more than two data at {}".format(id)

    examples = []
    utterances = document[0]["utterance"]
    for u_idx in range(len(utterances)-1):
        first_utterance = utterances[u_idx]
        second_utterance = utterances[u_idx+1]

        first_id = first_utterance["id"]
        second_id = second_utterance["id"]
        first_text = first_utterance["form"].replace("\n", " ")
        second_text = second_utterance["form"].replace("\n", " ")
        label = 0 if first_utterance["speaker_id"] == second_utterance["speaker_id"] else 1 # speaker가 같으면 0, 틀리면 1

        # remove data using emoticon, uploading photos ect...
        if first_text == "" or second_text == "":
            continue

        example = [first_id, second_id, first_text, second_text, label]
        examples.append(example)

    return examples

def build_examples(fns, task, task_process_function, task_column_names):
    in_fn = fns["input"]
    to_fn = fns["output"]

    data = []
    with open(in_fn, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

        # for debugging
        N = 70
        original_data = original_data[:N]

        data_iterator = tqdm(original_data, desc="Iteration")
        for ex_idx, ex in enumerate(data_iterator):
            data = data + task_process_function[task](ex)

    df = pd.DataFrame(data, columns=task_column_names[task])

    df.to_csv(to_fn, index=False, sep="\t")
    print("{} data is dumped at ".format(task), to_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turn_change",
                        help="What task do you want ?")
    args = parser.parse_args()

    task = args.task

    in_folder = os.path.join("./", "original-all")
    to_folder = os.path.join("./", task)
    prepare_dir(to_folder)

    fns = {
        "input" : os.path.join(in_folder, "data.json"),
        "output" : os.path.join(to_folder, "data.tsv")
    }

    task_process_function = {
        "turn_change" : build_turn_change_exmaple
    }
    task_column_names = {
        "turn_change" : ["utterance1_ids", "utterance2_id", "utterance1", "utterance2", "label"]
        # "turn_change": ["utterance1_ids", "utterance2_id", "label"]
    }

    build_examples(fns, task, task_process_function, task_column_names)