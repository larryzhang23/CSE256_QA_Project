import torch
import collections
import numpy as np
import evaluate
from data.dataset import SQuADBert
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm

def calculate_exact_match():
    return 1.0

def calculate_f1():
    return 1.0

def forward_one_step(model, inputs, accelerator, optimizer, lr_scheduler):
    outputs = model(**inputs)
    loss = outputs.loss
    
    accelerator.backward(loss)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


def train_one_epoch(model, dataloader, accelerator, optimizer, lr_scheduler):
    # Train step
    model.train()

    num_training_steps = len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    for step, batch in enumerate(dataloader):
        forward_one_step(model, batch, accelerator, optimizer, lr_scheduler)
        progress_bar.update(1)


def evaluate_model(model, dataloader):
    with torch.no_grad():
        pass


def train_model(model, train_set, accelerator, optimizer, num_train_epochs=5):
    train_dataloader = DataLoader(train_set, shuffle=True, collate_fn=default_data_collator, batch_size=8)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps,)

    for epoch in range(num_train_epochs):
        print("="*10 + f"Epoch {epoch + 1}", "="*10)
        train_one_epoch(model, train_dataloader, accelerator, optimizer, lr_scheduler)
        print()


model_name = "distilbert-base-uncased"
model_name = "distilbert-base-cased-distilled-squad"
squad_train = SQuADBert("train[100:200]", model_name)
squad_val = SQuADBert("validation[:10]", model_name)
print(squad_val.dataset["answers"])
# print(squad_train[1:2])
# print(squad_val[1:2])

model = AutoModelForQuestionAnswering.from_pretrained(model_name)
optimizer = AdamW(model.parameters(), lr=2e-5)
accelerator = Accelerator()

with torch.no_grad():
    squad_val_for_model = squad_val.tokenized_dataset.remove_columns(["example_id", "offset_mapping"])
    squad_val_for_model.set_format("torch")
    model, squad_val_for_model = accelerator.prepare(model, squad_val_for_model)
    model.eval()
    outputs = model(**squad_val_for_model[:])

    start_logits = outputs.start_logits.cpu().numpy()
    end_logits = outputs.end_logits.cpu().numpy()

    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(squad_val.tokenized_dataset):
        example_to_features[feature["example_id"]].append(idx)

    n_best = 10
    max_answer_length = 30
    predicted_answers = []

    for example in squad_val.dataset:
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_idx in example_to_features[example_id]:
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offsets = squad_val.tokenized_dataset["offset_mapping"][feature_idx]

            start_indexes = np.argsort(start_logit)[-1: -n_best - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -n_best - 1: -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (end_index < start_index) or (end_index - start_index + 1 > max_answer_length):
                        continue

                    answers.append({"text": context[offsets[start_index].cpu()[0].item(): offsets[end_index].cpu()[1].item()],
                                    "logit_score": start_logit[start_index] + end_logit[end_index]})
        
        best_answer = max(answers, key=lambda x: x["logit_score"])
        predicted_answers.append({"id": example_id, "prediction_text": best_answer["text"]})
    
    print(predicted_answers)

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in squad_val.dataset]
    print(theoretical_answers)

    metric = evaluate.load("squad")
    eval_result = metric.compute(predictions=predicted_answers, references=theoretical_answers)

    print(eval_result)


train_model(model, train_set=squad_train, accelerator=accelerator, optimizer=optimizer)