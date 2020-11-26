import os
import json
import argparse
from string import ascii_uppercase

import pandas as pd

from tqdm.auto import tqdm

from common.utils import prepare_dir

def build_turn_change_with_window(origin_ex, window_size):
    id = origin_ex["id"]
    document = origin_ex["document"][0]

    speaker_num = len(document["metadata"]["speaker"])
    speaker_tokens = list(ascii_uppercase)[:speaker_num]  # A, B, ...
    speaker_id_to_tokens = {str(k + 1): v for k, v in enumerate(speaker_tokens)}

    examples = []
    utterances = document["utterance"]
    for u_idx, utterance in enumerate(utterances):
        if (u_idx+window_size) >= len(utterances):
            break

        input_queue = list(reversed(utterances[u_idx:(u_idx+window_size)])) # for _truncate_seq_pair() function

        id_list = []
        speaker_list = []
        text_list = []
        while len(input_queue) != 0:
            u = input_queue.pop(0)

            id = u["id"]
            speaker_id = u["speaker_id"]
            text = u["form"].replace("\n", " ")
            if speaker_id not in list(speaker_id_to_tokens.keys()): # check typo
                break
            if text == "": # remove special messages [ex) emotion, upload photo, etc...]
                break
            speaker = speaker_id_to_tokens[speaker_id]

            id_list.append(id)
            text_list.append(text)
            speaker_list.append(speaker)

        if len(input_queue) > 0: # validation data for checking typo, special messages
            continue

        input_text = ",".join("{}({})".format(t, v) for t, v in zip(speaker_list, text_list))
        last_speaker = speaker_list[0]

        target_utterance = utterances[u_idx+window_size]
        target_id = target_utterance["id"]
        target_speaker_id = target_utterance["speaker_id"]
        label = -1
        if (target_speaker_id not in list(speaker_id_to_tokens.keys())):
            label = 1 # 리스트에도 없으면 화자가 다른 것이므로
        elif speaker_id_to_tokens[target_speaker_id] != last_speaker:
            label = 1 # 화자가 다름
        else:
            label = 0 # 화자가 같음

        example = [id_list, target_id, input_text, label]
        examples.append(example)

    return examples

def build_turn_change_final2(origin_ex, sentence_num):
    id = origin_ex["id"]
    document = origin_ex["document"][0]

    speaker_num = len(document["metadata"]["speaker"])
    speaker_tokens = list(ascii_uppercase)[:speaker_num]  # A, B, ...
    speaker_id_to_tokens = {str(k + 1): v for k, v in enumerate(speaker_tokens)}

    examples = []
    utterances = document["utterance"]
    for u_idx in range(0, (len(utterances) - sentence_num - 1), sentence_num):
        input_utterances = utterances[u_idx:(u_idx + sentence_num)]

        input_id_list = []
        input_text_list = []
        input_speaker_list = []
        for input_utterance in input_utterances:
            # remove typo error
            if input_utterance["speaker_id"] not in list(speaker_id_to_tokens.keys()):
                break

            text = input_utterance["form"].replace("\n", " ")
            speaker = speaker_id_to_tokens[input_utterance["speaker_id"]]

            # remove data using emoticon, uploading photos ect...
            if text == "":
                break

            input_id_list.append(input_utterance["id"])
            input_text_list.append(text)
            input_speaker_list.append(speaker)

        # remove data using emoticon, uploading photos ect...
        if (len(input_id_list) != sentence_num):
            continue

        target_text = input_text_list[-1]
        input_text = ",".join("{}({})".format(t, v) for t, v in zip(input_speaker_list, input_text_list))

        target_id = input_id_list[-1]
        target_speaker = input_speaker_list[-1]

        last_speaker = input_speaker_list[-1]
        label = 0 if target_speaker == last_speaker else 1  # speaker가 같으면 0, 틀리면 1

        example = [input_id_list, target_id, input_text, target_text, label]
        examples.append(example)

    return examples

def build_turn_change_final(origin_ex, sentence_num):
    id = origin_ex["id"]
    document = origin_ex["document"][0]

    speaker_num = len(document["metadata"]["speaker"])
    speaker_tokens = list(ascii_uppercase)[:speaker_num] # A, B, ...
    speaker_id_to_tokens = {str(k+1) : v for k, v in enumerate(speaker_tokens)}

    examples = []
    utterances = document["utterance"]
    for u_idx in range(0, (len(utterances)-sentence_num-1), sentence_num):
        input_utterances, target_utterance = utterances[u_idx:(u_idx + sentence_num - 1)], utterances[u_idx + sentence_num - 1]

        input_id_list = []
        input_text_list = []
        input_speaker_list = []
        for input_utterance in input_utterances:
            # remove typo error
            if input_utterance["speaker_id"] not in list(speaker_id_to_tokens.keys()):
                break

            text = input_utterance["form"].replace("\n", " ")
            speaker = speaker_id_to_tokens[input_utterance["speaker_id"]]

            # remove data using emoticon, uploading photos ect...
            if text == "":
                break

            input_id_list.append(input_utterance["id"])
            input_text_list.append(text)
            input_speaker_list.append(speaker)

        target_text = target_utterance["form"].replace("\n", " ")
        input_text = ",".join("{}({})".format(t, v) for t, v in zip(input_speaker_list, input_text_list))

        # remove data using emoticon, uploading photos ect...
        if (len(input_id_list) != (sentence_num - 1)) or target_text == "":
            continue

        target_id = target_utterance["id"]
        target_speaker = speaker_id_to_tokens[target_utterance["speaker_id"]]

        last_speaker = input_speaker_list[-1]
        label = 0 if target_speaker == last_speaker else 1  # speaker가 같으면 0, 틀리면 1

        example = [input_id_list, target_id, input_text, target_text, label]
        examples.append(example)

    return examples


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
    # useful_labels = ["일상", "교육 및 학교 (교과목, 진로, 입시, 성적, 스터디)", "주거와 생활 (집안일, 육아, 부동산, 경제 활동, 생활 정보)",
    #                  "예술, 문화 생활 (문학, 음악, 미술, 공연, 전시, 관람)", "교통 (위치, 거리, 이동 수단, 대중교통)",
    #                  "식음료 (식사, 음식, 배달, 맛집, 요리)", "시사, 사회 (정치, 경제, 여론, 사건과 사고)",
    #                  "일과 직업 (취업, 스펙, 업무, 급여, 회의)", "상거래(쇼핑)", "여행 (여행지, 계획 등)",
    #                  "여가와 오락 (유흥, 취미, 관심사, 휴일 활동, 동아리, 동호회)", "개인 및 관계 (가족관계, 고향 등 개인의 신상, 인간 관계 등)",
    #                  "날씨와 계절", "미용과 건강 (질병과 치료, 운동, 다이어트, 미용)"]

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
            # for removing unnecessary labels
            # if task == "topic" and not ex["document"][0]["metadata"]["topic"] in useful_labels:
            #     continue
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
    to_folder = os.path.join("./", task)
    # to_folder = os.path.join("./", task + "_temp")
    prepare_dir(to_folder)

    fns = {
        "input" : os.path.join(in_folder, "data.json"),
        "output" : os.path.join(to_folder, "data.tsv")
    }

    task_process_function = {
        "turn_change" : build_turn_change_with_window,
        "topic" : build_topic_example
    }
    task_column_names = {
        "turn_change" : ["context_utterance_ids", "target_utterance_id", "context", "label"],
        # "turn_change": ["utterance1_ids", "utterance2_id", "label"]
        "topic" : ["id", "text", "topic"]
    }

    build_examples(fns, task, task_process_function, task_column_names, sentence_num)
