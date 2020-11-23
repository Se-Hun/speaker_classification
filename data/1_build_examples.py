import os
import json
import argparse

import pandas as pd

from tqdm.auto import tqdm

from common.utils import prepare_dir


def build_turn_change_example_all(origin_ex):
    id = origin_ex["id"]
    document = origin_ex["document"]
    assert (len(document) == 1), "Document has more than two data at {}".format(id)

    examples = []
    utterances = document[0]["utterance"]

    input_utterances, target_utterance = utterances[:-1], utterances[-1]

    input_ids = [v["id"] for v in input_utterances]
    input_text = ""
    for v in input_utterances:
        text = v["form"].replace("\n", " ")
        input_text = input_text + " " + text
    last_speaker = input_utterances[-1]["speaker_id"]

    target_id = target_utterance["id"]
    target_text = target_utterance["form"].replace("\n", " ")
    target_speaker = target_utterance["speaker_id"]

    label = 0 if last_speaker == target_speaker else 1  # speaker가 같으면 0, 틀리면 1
    example = [input_ids, target_id, input_text, target_text, label]
    examples.append(example)

    return examples


def build_turn_change_example_params(origin_ex, sentence_num):
    id = origin_ex["id"]
    document = origin_ex["document"]
    assert (len(document) == 1), "Document has more than two data at {}".format(id)

    examples = []
    utterances = document[0]["utterance"]
    for u_idx in range(0, (len(utterances)-sentence_num-1), sentence_num):
        input_utterances, target_utterance = utterances[u_idx:(u_idx+sentence_num-1)], utterances[u_idx+sentence_num-1]

        target_id = target_utterance["id"]
        target_text = target_utterance["form"].replace("\n", " ")

        last_speaker = input_utterances[-1]["speaker_id"]
        id_list = []
        input_text = ""
        for v in input_utterances:
            text = v["form"].replace("\n", " ")

            # remove data using emoticon, uploading photos ect...
            if text == "":
                break;

            id_list.append(v["id"])
            input_text = input_text + " " + text

        # remove data using emoticon, uploading photos ect...
        if (len(id_list) != (sentence_num-1)) or target_text == "":
            continue

        label = 0 if last_speaker == target_utterance["speaker_id"] else 1  # speaker가 같으면 0, 틀리면 1
        example = [id_list, target_id, input_text, target_text, label]
        examples.append(example)

    # for u_idx in range(0, len(utterances) - 2, 3):
    #     first_utterance = utterances[u_idx]
    #     second_utterance = utterances[u_idx + 1]
    #     target_utterance = utterances[u_idx + 2]
    #
    #     first_id = first_utterance["id"]
    #     second_id = second_utterance["id"]
    #     target_id = target_utterance["id"]
    #
    #     first_text = first_utterance["form"].replace("\n", " ")
    #     second_text = second_utterance["form"].replace("\n", " ")
    #     target_text = target_utterance["form"].replace("\n", " ")
    #
    #     label = 0 if second_utterance["speaker_id"] == target_utterance["speaker_id"] else 1  # speaker가 같으면 0, 틀리면 1
    #
    #     # remove data using emoticon, uploading photos ect...
    #     if first_text == "" or second_text == "" or target_text == "":
    #         continue
    #
    #     example = [[first_id, second_id], target_id, first_text + " " + second_text, target_text, label]
    #     examples.append(example)

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

def build_topic_example(origin_ex, sentence_num=None):
    id = origin_ex["id"]
    document = origin_ex["document"]
    assert (len(document) == 1), "Document has more than two data at {}".format(id)

    utterances = document[0]["utterance"]
    topic = document[0]["metadata"]["topic"]

    merged_text = ""
    for utterance in utterances:
        utterance_text = utterance["form"].replace("\n", " ")
        merged_text = merged_text + " " + utterance_text

    example = [[id, merged_text, topic]]
    return example

def build_examples(fns, task, task_process_function, task_column_names, sentence_num):
    in_fn = fns["input"]
    to_fn = fns["output"]

    data = []
    with open(in_fn, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

        # for debugging
        # N = 70
        # original_data = original_data[:N]

        data_iterator = tqdm(original_data, desc="Iteration")
        for ex_idx, ex in enumerate(data_iterator):
            data = data + task_process_function[task](ex, sentence_num)
            # data = data + task_process_function[task](ex)

    print("Number of Examples : {}".format(len(data)))
    df = pd.DataFrame(data, columns=task_column_names[task])

    df.to_csv(to_fn, index=False, sep="\t")
    print("{} data is dumped at ".format(task), to_fn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="turn_change",
                        help="What task do you want ?")
    parser.add_argument("--sentence_num", default=3, type=int,
                        help="How many sentences use for prediction ?")
    args = parser.parse_args()

    task = args.task
    sentence_num = args.sentence_num

    in_folder = os.path.join("./", "original-all")
    # to_folder = os.path.join("./", task + "_temp")
    to_folder = os.path.join("./", task)
    prepare_dir(to_folder)

    fns = {
        "input" : os.path.join(in_folder, "data.json"),
        "output" : os.path.join(to_folder, "data.tsv")
    }

    task_process_function = {
        "turn_change" : build_turn_change_example_params,
        "topic" : build_topic_example
    }
    task_column_names = {
        "turn_change" : ["utterance1_ids", "utterance2_id", "utterance1", "utterance2", "label"],
        # "turn_change": ["utterance1_ids", "utterance2_id", "label"]
        "topic" : ["id", "text", "topic"]
    }

    build_examples(fns, task, task_process_function, task_column_names, sentence_num)
