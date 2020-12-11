import os
import argparse
import platform
from glob import glob

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

class TextClassification(pl.LightningModule):
    def __init__(self,
                 task,
                 text_reader,
                 num_labels,
                 learning_rate: float=2e-5):
        super().__init__()
        self.save_hyperparameters()

        # prepare text reader
        from utils.readers import get_text_reader
        text_reader = get_text_reader(self.hparams.text_reader, num_labels)
        self.text_reader = text_reader

    def forward(self, input_ids, token_type_ids, attention_mask, label_id):
        output = self.text_reader(
            input_ids=input_ids.long(),
            token_type_ids=token_type_ids.long(),
            attention_mask=attention_mask.float(),
            labels=label_id.long()
        )

        return output

    def training_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label_id = batch

        loss, _ = self(input_ids, token_type_ids, attention_mask, label_id)

        result = {"loss": loss}
        return result

    def validation_step(self, batch, batch_idx):
        input_ids, token_type_ids, attention_mask, label_id = batch

        loss, logits = self(input_ids, token_type_ids, attention_mask, label_id)

        preds = torch.argmax(logits, dim=1)

        labels = label_id
        result = {"loss": loss, "preds": preds, "labels": labels}
        return result

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])
        loss = torch.stack([x["loss"] for x in outputs]).mean()

        correct_count = torch.sum(labels == preds)
        val_acc = correct_count.float() / float(len(labels))

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", val_acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx, dataloader_idx):
        input_ids, token_type_ids, attention_mask, label_id = batch

        _, logits = self(input_ids, token_type_ids, attention_mask, label_id)

        preds = torch.argmax(logits, dim=1)

        labels = label_id
        result = {"preds": preds, "labels": labels}
        return result

    def test_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])

        correct_count = torch.sum(labels == preds)
        test_acc = correct_count.float() / float(len(labels))

        self.log("test_acc", test_acc, prog_bar=True)
        return test_acc

    def configure_optimizers(self):
        from transformers import AdamW

        param_optimizer = list(self.named_parameters())
        no_decay = ['bias', 'gamma', 'beta']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay_rate': 0.0}
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=2e-5)
        return parser


def main():
    pl.seed_everything(42) # set seed

    # Argument Setting -------------------------------------------------------------------------------------------------
    parser = argparse.ArgumentParser()

    # task specific --------------------------------------------------------------------------------
    parser.add_argument("--task", help="What task do you want? : turn_change, ...", default="turn_change")

    # mode specific --------------------------------------------------------------------------------
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to train text classifier.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to test on test dataset.")

    # model specific -------------------------------------------------------------------------------
    parser.add_argument("--text_reader", help="bert, kobert, kcbert, koelectra, others, ...", default="bert")

    # experiment settings --------------------------------------------------------------------------
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")  # bert has 512 tokens.
    parser.add_argument("--batch_size", help="batch_size", default=32, type=int)
    parser.add_argument("--gpu_id", help="gpu device id", default="0")

    parser = pl.Trainer.add_argparse_args(parser)
    parser = TextClassification.add_model_specific_args(parser)
    args = parser.parse_args()
    # ------------------------------------------------------------------------------------------------------------------

    # Dataset ----------------------------------------------------------------------------------------------------------
    # This version only support type of Text Classification tasks.
    from dataset import Text_Classification_Data_Module
    dm = Text_Classification_Data_Module(args.task, args.text_reader, args.max_seq_length, args.batch_size)
    dm.prepare_data()
    # ------------------------------------------------------------------------------------------------------------------

    # Model Checkpoint -------------------------------------------------------------------------------------------------
    from pytorch_lightning.callbacks import ModelCheckpoint
    model_name = '{}'.format(args.text_reader)
    model_folder = './model/{}/{}'.format(args.task, model_name)
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          dirpath=model_folder,
                                          filename='{epoch:02d}-{val_loss:.2f}')
    # ------------------------------------------------------------------------------------------------------------------

    # Early Stopping ---------------------------------------------------------------------------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        patience=3,
        verbose=True
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Trainer ----------------------------------------------------------------------------------------------------------
    trainer = pl.Trainer(
        gpus=args.gpu_id if platform.system() != 'Windows' else 1,  # <-- for dev. pc
        checkpoint_callback=checkpoint_callback,
        callbacks=[early_stop_callback]
    )
    # ------------------------------------------------------------------------------------------------------------------

    # Do train !
    if args.do_train:
        model = TextClassification(args.task, args.text_reader, dm.num_labels)
        trainer.fit(model, dm)

    # Do test !
    if args.do_test:
        model_files = glob(os.path.join(model_folder, '*.ckpt'))
        best_fn = model_files[-1]
        model = TextClassification.load_from_checkpoint(best_fn)
        trainer.test(model, test_dataloaders=[dm.test_dataloader()])

if __name__ == '__main__':
    main()