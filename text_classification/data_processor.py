import os

from transformers import DataProcessor, InputExample, InputFeatures

def convert_examples_to_features(examples, tokenizer, max_length, task_name=None, label_list=None, output_mode=None):
    if max_length is None:
        max_length = tokenizer.max_len

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
        max_length=max_length,
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
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        print("val data is loading at {}".format(data_dir))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "dev") # In current, fixed at test.tsv

    def get_test_examples(self, data_dir):
        print("test data is loading at {}".format(data_dir))
        return self._create_examples(self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        return ["0", "1"]

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
            text_a = line[2]
            text_b = line[3]
            label = line[4]
            # label = None if set_type == "test" else line[4]
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

text_classification_processors = {
    "turn_change" : TurnChangeProcessor
}
