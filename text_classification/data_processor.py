import os
import csv
import json

from transformers import DataProcessor, InputExample, InputFeatures

csv.field_size_limit(1000000000000)

def convert_examples_to_features_using_batch_encoding(examples, tokenizer, max_seq_length, task_name=None, label_list=None, output_mode=None):
    if max_seq_length is None:
        max_seq_length = tokenizer.max_len

    if task_name is not None:
        processor = text_classification_processors[task_name]()
        if label_list is None:
            label_list = processor.get_labels()
            print("Using label list %s for task %s" % (label_list, task_name))
        if output_mode is None:
            output_mode = "classification"
            print("Using output mode %s for task %s" % (output_mode, task_name))

    label_map = {label: i for i, label in enumerate(label_list)}

    def label_from_example(example):
        if example.label is None:
            return None
        if output_mode == "classification":
            return label_map[example.label]

    labels = [label_from_example(example) for example in examples]

    batch_encoding = tokenizer(
        [(example.text_a, example.text_b) for example in examples],
        max_length=max_seq_length,
        padding="max_length",
        truncation=True,
    )

    features = []
    for i in range(len(examples)):
        inputs = {k: batch_encoding[k][i] for k in batch_encoding}

        feature = InputFeatures(**inputs, label=labels[i])
        features.append(feature)

    for i, example in enumerate(examples[:5]):
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("features: %s" % features[i])

    return features

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.

    # for debugging
    # total_length = len(tokens_a) + len(tokens_b)
    # assert (total_length <= max_length), "Beyond Max Sequence Length. total_token_length is {}, max_length is {}".format(total_length, max_length)

    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

# this code is from https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/examples/run_classifier.py
def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer, output_mode):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    num_tokens = []

    from tqdm.auto import tqdm
    example_iterator = tqdm(examples, desc="Iteration")

    for (ex_index, example) in enumerate(example_iterator):
        tokens_a = tokenizer.tokenize(example.text_a)
        num_tokens.append(len(tokens_a))

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        token_type_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            token_type_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        attention_mask += padding
        token_type_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(attention_mask) == max_seq_length
        assert len(token_type_ids) == max_seq_length

        if output_mode == "classification":
            label = label_map[example.label]
        elif output_mode == "regression":
            label = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          label=label))

    for i, example in enumerate(examples[:5]):
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("features: %s" % features[i])

    return features, num_tokens

class TurnChangeProcessor(DataProcessor):
    """Processor for the Turn Change data set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        print("train data is loading at {}".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "train.json")), "train")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        print("val data is loading at {}".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "dev")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev") # In current, fixed at test.tsv

    def get_test_examples(self, data_dir):
        print("test data is loading at {}".format(data_dir))
        # return self._create_examples(self._read_json(os.path.join(data_dir, "test.json")), "test")
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

    def get_num_labels(self):
        return len(self.get_labels())

    def _read_json(self, fn):
        examples = []
        with open(fn, 'r', encoding='utf-8') as f:
            data = json.load(f)

            for ex in data:
                utterances = ex["document"][0]["utterance"]
                input_utterances, target_utterance = utterances[:-1], utterances[-1]

                input_text = ""
                for v in input_utterances:
                    text = v["form"].replace("\n", " ")
                    input_text = input_text + " " + text
                last_speaker = input_utterances[-1]["speaker_id"]

                target_text = target_utterance["form"].replace("\n", " ")
                target_speaker = target_utterance["speaker_id"]

                label = "0" if last_speaker == target_speaker else "1"  # speaker가 같으면 0, 틀리면 1
                example = [input_text, target_text, label]
                examples.append(example)

        return examples

    # def _create_examples(self, examples, set_type):
    #     # for debugging
    #     N = 40
    #     examples = examples[:N]
    #
    #     examples_for_model = []
    #     for i, ex in enumerate(examples):
    #         guid = "%s-%s" % (set_type, i)
    #         text_a = ex[0]
    #         text_b = ex[1]
    #         label = ex[2]
    #         # label = None if set_type == "test" else line[4]
    #         examples_for_model.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    #     return examples_for_model

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # for debugging
        # N = 40
        N = 40000
        lines = lines[:N]

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[2]
            text_b = line[3]
            label = line[4]
            # label = None if set_type == "test" else line[4]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class TopicProcessor(DataProcessor):
    """Processor for the Topic Prediction data set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_example_from_tensor_dict(self, tensor_dict):
        return InputExample(
            tensor_dict["idx"].numpy(),
            tensor_dict["sentence1"].numpy().decode("utf-8"),
            tensor_dict["sentence2"].numpy().decode("utf-8"),
            str(tensor_dict["label"].numpy()),
        )

    def get_train_examples(self, data_dir):
        print("train data is loading at {}".format(data_dir))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        print("val data is loading at {}".format(data_dir))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev")  # In current, fixed at test.tsv

    def get_test_examples(self, data_dir):
        print("test data is loading at {}".format(data_dir))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["일상", "공공 서비스 (전화, 통신, 인터넷 서비스, 관공서)", "교육 및 학교 (교과목, 진로, 입시, 성적, 스터디)",
                "주거와 생활 (집안일, 육아, 부동산, 경제 활동, 생활 정보)", "예술, 문화 생활 (문학, 음악, 미술, 공연, 전시, 관람)",
                "교통 (위치, 거리, 이동 수단, 대중교통)", "여행 (여행지, 계획 등), 주거와 생활 (집안일, 육아, 부동산, 경제 활동, 생활 정보)",
                "행사 및 모임(초대, 방문, 소개팅, 약속, 친목 모임)", "식음료 (식사, 음식, 배달, 맛집, 요리)",
                "시사, 사회 (정치, 경제, 여론, 사건과 사고)", "일과 직업 (취업, 스펙, 업무, 급여, 회의)", "상거래(쇼핑)",
                "여행 (여행지, 계획 등)", "여가와 오락 (유흥, 취미, 관심사, 휴일 활동, 동아리, 동호회)",
                "개인 및 관계 (가족관계, 고향 등 개인의 신상, 인간 관계 등)", "날씨와 계절", "미용과 건강 (질병과 치료, 운동, 다이어트, 미용)",
                "전공/전문 지식"]

    def get_num_labels(self):
        return len(self.get_labels())

    def _create_examples(self, lines, set_type):
        """Creates examples for the training, dev and test sets."""
        # for debugging
        # N = 70
        # lines = lines[:N]

        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            text_b = None
            label = line[2]
            # label = None if set_type == "test" else line[4]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

text_classification_processors = {
    "turn_change" : TurnChangeProcessor,
    "topic" : TopicProcessor
}
