from collections import OrderedDict

import torch
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader
import pytorch_lightning as pl

from transformers import BertForSequenceClassification, BertTokenizer, AdamW

from text_classification.data_processor import text_classification_processors, convert_examples_to_features, \
    convert_examples_to_features_using_batch_encoding


def get_dataloader(task_name, tokenizer, data_dir, data_processor, max_seq_length=128, batch_size=32):
    label_list = data_processor.get_labels()
    output_mode = "classification"

    train_examples = data_processor.get_train_examples(data_dir)
    train_features, num_tokens = convert_examples_to_features(train_examples, label_list, max_seq_length, tokenizer, output_mode)
    # train_features = convert_examples_to_features_using_batch_encoding(train_examples, tokenizer, max_seq_length, task_name=task_name)
    train_dataset = TensorDataset(torch.tensor([f.input_ids for f in train_features], dtype=torch.long),
                                  torch.tensor([f.attention_mask for f in train_features], dtype=torch.long),
                                  torch.tensor([f.token_type_ids for f in train_features], dtype=torch.long),
                                  torch.tensor([f.label for f in train_features], dtype=torch.long))

    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size, num_workers=4)

    val_examples = data_processor.get_dev_examples(data_dir)
    val_features, num_tokens = convert_examples_to_features(val_examples, label_list, max_seq_length, tokenizer, output_mode)
    # val_features = convert_examples_to_features_using_batch_encoding(val_examples, tokenizer, max_seq_length, task_name=task_name)
    val_dataset = TensorDataset(torch.tensor([f.input_ids for f in val_features], dtype=torch.long),
                                torch.tensor([f.attention_mask for f in val_features], dtype=torch.long),
                                torch.tensor([f.token_type_ids for f in val_features], dtype=torch.long),
                                torch.tensor([f.label for f in val_features], dtype=torch.long))
    val_sampler = SequentialSampler(val_dataset)
    val_dataloader = DataLoader(val_dataset, sampler=val_sampler, batch_size=batch_size, num_workers=4)

    test_examples = data_processor.get_test_examples(data_dir)
    test_features, num_tokens = convert_examples_to_features(test_examples, label_list, max_seq_length, tokenizer, output_mode)
    # test_features = convert_examples_to_features_using_batch_encoding(test_examples, tokenizer, max_seq_length, task_name=task_name)
    test_dataset = TensorDataset(torch.tensor([f.input_ids for f in test_features], dtype=torch.long),
                                 torch.tensor([f.attention_mask for f in test_features], dtype=torch.long),
                                 torch.tensor([f.token_type_ids for f in test_features], dtype=torch.long),
                                 torch.tensor([f.label for f in test_features], dtype=torch.long))
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size, num_workers=4)

    return train_dataloader, val_dataloader, test_dataloader


class TextClassifier(pl.LightningModule):
    def __init__(self, task_name, data_dir, max_seq_length, batch_size):
        super(TextClassifier, self).__init__()
        self.task_name = task_name

        self.data_dir = data_dir
        self.data_processor = text_classification_processors[task_name]()
        num_labels = self.data_processor.get_num_labels()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        self.max_seq_length = max_seq_length

        model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased", num_labels=num_labels)
        self.model = model

        train_dataloader, val_dataloader, test_dataloader = get_dataloader(
            self.task_name,
            self.tokenizer,
            self.data_dir,
            self.data_processor,
            self.max_seq_length,
            batch_size
        )

        self._train_dataloader = train_dataloader
        self._val_dataloader = val_dataloader
        self._test_dataloader = test_dataloader

    def configure_optimizers(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "gamma", "beta"]
        optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.01
                    },
                {
                    "params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
                    "weight_decay_rate": 0.0
                    },
                ]
        optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=2e-5,
                )
        return optimizer

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        loss, _ = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        tqdm_dict = {"train_loss": loss}
        output = OrderedDict({
            "loss": loss,
            "progress_bar": tqdm_dict,
            "log": tqdm_dict
        })

        return output

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        # self.log('my_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        output = OrderedDict({
            "val_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })
        return output

    def validation_epoch_end(self, outputs):
        val_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        val_loss = sum([out["val_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
            "val_loss": val_loss.item(),
            "val_acc": val_acc.item(),
        }

        result = {"progress_bar": tqdm_dict, "log": tqdm_dict, "val_loss": val_loss}
        return result

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, token_type_ids, labels = batch

        loss, logits = self.model(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        labels_hat = torch.argmax(logits, dim=1)

        correct_count = torch.sum(labels == labels_hat)

        if self.on_gpu:
            correct_count = correct_count.cuda(loss.device.index)

        output = OrderedDict({
            "test_loss": loss,
            "correct_count": correct_count,
            "batch_size": len(labels)
        })

        return output

    def test_epoch_end(self, outputs):
        test_acc = sum([out["correct_count"] for out in outputs]).float() / sum(out["batch_size"] for out in outputs)
        test_loss = sum([out["test_loss"] for out in outputs]) / len(outputs)
        tqdm_dict = {
            "test_loss": test_loss.item(),
            "test_acc": test_acc.item(),
        }
        result = {"progress_bar": tqdm_dict, "log": tqdm_dict}
        return result

    def train_dataloader(self):
        return self._train_dataloader

    def val_dataloader(self):
        return self._val_dataloader

    def test_dataloader(self):
        return self._test_dataloader