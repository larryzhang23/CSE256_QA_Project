import trainer
import torch
from data.dataset import SQuADBert
from transformers import AutoModelForQuestionAnswering
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import default_data_collator, get_scheduler
from accelerate import Accelerator
from tqdm import tqdm

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


def evaluate(model, dataloader):
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


squad_train = SQuADBert("train[100:200]", "distilbert-base-uncased")
squad_val = SQuADBert("validation[10:50]", "distilbert-base-uncased")
print(squad_train[1:2])
print(squad_val[1:2])

model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
optimizer = AdamW(model.parameters(), lr=2e-5)
accelerator = Accelerator()

train_model(model, train_set=squad_train, accelerator=accelerator, optimizer=optimizer)