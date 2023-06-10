import torch
import collections
import numpy as np

from data.dataset import SQuADBert
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm


def calculate_exact_match(predicted_answers, theoretical_answers):
    correct_count = 0
    for predicted_answer, theoretical_answer in zip(
        predicted_answers, theoretical_answers
    ):
        # print(predicted_answer)
        # print(theoretical_answer)
        pred = predicted_answer["prediction_text"]
        for answer in theoretical_answer["answers"]["text"]:
            if pred == answer:
                correct_count += 1
                break
        if len(pred) == 0 and len(theoretical_answer["answers"]["text"]) == 0:
            correct_count += 1
    return correct_count / len(predicted_answers)


def calculate_f1(predicted_answers, theoretical_answers):
    overall_f1 = 0.0
    for predicted_answer, theoretical_answer in zip(
        predicted_answers, theoretical_answers
    ):
        pred = predicted_answer["prediction_text"]
        pred_tokens = set(pred.strip().split())
        best_f1 = 0.0
        for answer in theoretical_answer["answers"]["text"]:
            answer_tokens = set(answer.strip().split())
            current_tp = len(pred_tokens.intersection(answer_tokens))
            current_fp = len(pred_tokens.difference(answer_tokens))
            current_fn = len(answer_tokens.difference(pred_tokens))
            if current_tp + current_fp == 0:
                current_precision = 0
            else:
                current_precision = current_tp / (current_tp + current_fp)
            if current_tp + current_fp == 0:
                current_recall == 0
            else:
                current_recall = current_tp / (current_tp + current_fn)
            if current_recall + current_precision == 0:
                current_f1 = 0
            else:
                current_f1 = (
                    2
                    * current_precision
                    * current_recall
                    / (current_precision + current_recall)
                )
            if current_f1 > best_f1:
                best_f1 = current_f1
        if len(pred) == 0 and len(theoretical_answer["answers"]["text"]) == 0:
            best_f1 = 1.0
        overall_f1 += best_f1
        # print(best_f1)
    return overall_f1 / len(predicted_answers)


def forward_one_step(model, inputs, accelerator, optimizer, lr_scheduler):
    outputs = model(**inputs)
    loss = outputs.loss

    accelerator.backward(loss)

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()

    return loss.item()


def train_one_epoch(model, dataloader, accelerator, optimizer, lr_scheduler):
    # Train step
    model.train()

    num_training_steps = len(dataloader)
    progress_bar = tqdm(range(num_training_steps))

    total_loss = 0.0
    for step, batch in enumerate(dataloader):
        total_loss += forward_one_step(
            model, batch, accelerator, optimizer, lr_scheduler
        )
        progress_bar.update(1)

    return total_loss / len(dataloader)


@torch.no_grad()
def predict_answers(model, dataloader, val_set):
    model.eval()
    predicted_answers = []

    start_logits = []
    end_logits = []

    print("predicting...")
    num_eval_steps = len(dataloader)
    progress_bar = tqdm(range(num_eval_steps))
    for step, batch in enumerate(dataloader):
        features = {k: v.to(model.device) for k, v in batch.items()}
        outputs = model(**features)

        start_logits += list(outputs.start_logits.cpu().numpy())
        end_logits += list(outputs.end_logits.cpu().numpy())

        progress_bar.update(1)

    print("preprocessing...")
    example_to_features = collections.defaultdict(list)
    for idx, feature in tqdm(enumerate(val_set)):
        example_to_features[feature["example_id"]].append(idx)

    n_best = 1
    max_answer_length = 30

    print("generating answers:")
    for example in tqdm(val_set.dataset):
        example_id = example["id"]
        context = example["context"]
        answers = []

        for feature_idx in example_to_features[example_id]:
            start_logit = start_logits[feature_idx]
            end_logit = end_logits[feature_idx]
            offsets = val_set.tokenized_dataset["offset_mapping"][feature_idx]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()

            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (end_index < start_index) or (
                        end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answers.append(
                        {
                            "text": context[
                                offsets[start_index][0] : offsets[end_index][1]
                            ],
                            "logit_score": start_logit[start_index]
                            + end_logit[end_index],
                        }
                    )

            if len(answers) == 0:
                best_answer = {"text": "", "logit_score": 0}
            else:
                best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
    return predicted_answers


@torch.no_grad()
def evaluate_model(model, dataloader, val_set):
    predicted_answers = predict_answers(model, dataloader, val_set)
    theoretical_answers = [
        {"id": ex["id"], "answers": ex["answers"]} for ex in val_set.dataset
    ]

    em = calculate_exact_match(predicted_answers, theoretical_answers)
    f1 = calculate_f1(predicted_answers, theoretical_answers)

    return {"em": em, "f1": f1}


def train_model(
    model,
    train_set,
    val_set,
    accelerator,
    optimizer,
    num_train_epochs=5,
    train_batch_size=32,
    val_batch_size=64,
):
    train_dataloader = DataLoader(
        train_set, shuffle=True, collate_fn=default_data_collator, batch_size=train_batch_size
    )
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

    model, optimizer, train_dataloader = accelerator.prepare(
        model, optimizer, train_dataloader
    )

    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = num_train_epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    for epoch in range(num_train_epochs):
        print("=" * 10 + f"Epoch {epoch + 1}", "=" * 10)
        loss = train_one_epoch(
            model, train_dataloader, accelerator, optimizer, lr_scheduler
        )
        eval_result = evaluate_model(
            model=model, dataloader=val_dataloader, val_set=val_set
        )
        print(f"Loss: {loss}")
        print(f"Evaluation: EM={eval_result['em']}, F1={eval_result['f1']}")
        print()


if __name__ == "__main__":
    model_name = "distilbert-base-uncased"
    # model_name = "distilbert-base-cased-distilled-squad"
    # model_name = "./weights/albert_1_epoch"
    squad_train = SQuADBert("train[:100]", model_name)
    val_set = SQuADBert("validation", model_name)
    # print(squad_train[1:2])
    # print(squad_val[1:2])

    # model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained("./weights/albert_1_epoch")
    optimizer = AdamW(model.parameters(), lr=2e-5)
    accelerator = Accelerator()

    train_model(
        model,
        train_set=squad_train,
        val_set=val_set,
        accelerator=accelerator,
        optimizer=optimizer,
    )
