import trainer
from data.dataset import SQuADBert
from torch.utils.data import DataLoader

squad_train = SQuADBert("train[100:110]", "distilbert-base-uncased")
print(squad_train[1:3])