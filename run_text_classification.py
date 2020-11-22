import os
import argparse
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger

from transformers import set_seed

from text_classification.classifier import TextClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", help="task : turn_change, ...", default="turn_change")

    parser.add_argument("--data_dir", help="Should contain the data files for the task.", default="./data/turn_change/run")
    parser.add_argument("--tb_dir", help="directory saving tensor board log",
                        default="./tb/turn_change")

    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.") # bert has 512 tokens.
    parser.add_argument("--batch_size", help="batch_size", type=int, default=32)
    parser.add_argument("--seed", help='seed', type=int, default=42)

    parser.add_argument("--gpu_id", help="gpu device id", default="0")

    args = parser.parse_args()

    task_name = args.task
    data_dir = args.data_dir
    tb_dir = args.tb_dir
    batch_size = args.batch_size
    seed = args.seed
    max_seq_length = args.max_seq_length

    # setting seed -----------------------------------------------------------------------------------------------------
    set_seed(seed)

    # init early stopping ----------------------------------------------------------------------------------------------
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.0,
        patience=3,
        verbose=True,
        mode="min"
    )

    # init tensorboard callback ----------------------------------------------------------------------------------------
    tb_logger = TensorBoardLogger(
        save_dir=tb_dir,
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )

    # init model -------------------------------------------------------------------------------------------------------
    model = TextClassifier(task_name, data_dir, max_seq_length, batch_size)

    # setting gpu ------------------------------------------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # init pytorch lightning trainer -----------------------------------------------------------------------------------
    trainer = pl.Trainer(gpus=1,
                         logger=tb_logger,
                         callbacks=[early_stop_callback])

    # training and testing ---------------------------------------------------------------------------------------------
    trainer.fit(model)
    trainer.test()