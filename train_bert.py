import trainer
import torch
from data.dataset import SQuADBert
from torch.utils.data import DataLoader
from transformers import AutoModelForQuestionAnswering

squad_train = SQuADBert("train[100:110]", "distilbert-base-uncased")
print(squad_train[1:3])

with torch.inference_mode():
    model = AutoModelForQuestionAnswering.from_pretrained("distilbert-base-uncased")
    # print(squad_train[0])
    pred = model(**squad_train[0:1])
    print(pred)