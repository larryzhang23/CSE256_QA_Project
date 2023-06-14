from train_bert import *

import torch
import collections
import numpy as np
import pickle

from data.dataset import SQuADBert
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm

if __name__ == "__main__":
    # model_name = "distilbert-base-uncased"
    # model_name = "microsoft/deberta-v3-xsmall"
    model_name = "albert-base-v2"
    outfile = "results/albert_val.pkl"
    accelerator = Accelerator()
    model = AutoModelForQuestionAnswering.from_pretrained("./weights/albert_trained")
    model = accelerator.prepare(
        model
    )

    all_preds = []
    all_labels = []

    for i in tqdm(range(0, 11873, 100)):
        val_set = SQuADBert(f"validation[{i}:{min(i + 100, 11873)}]", model_name)
        val_batch_size = 32

        val_for_model = val_set.tokenized_dataset.remove_columns(
            ["example_id", "offset_mapping"]
        )
        val_for_model.set_format("torch")
        val_dataloader = DataLoader(
            val_for_model,
            shuffle=False,
            collate_fn=default_data_collator,
            batch_size=val_batch_size,
        )

        predicted_answers = predict_answers(model, val_dataloader, val_set)
        theoretical_answers = [
            {"id": ex["id"], "answers": ex["answers"]} for ex in val_set.dataset
        ]

        all_preds += predicted_answers
        all_labels += theoretical_answers

        # for pred, label in zip(predicted_answers, theoretical_answers):
        #     print(pred, label)
    val_result = {"preds": all_preds, "labels": all_labels}

    with open(outfile, "wb") as f:
        pickle.dump(val_result, f)
