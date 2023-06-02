import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from model import InputEmbedClf, EmbedEncClf, MACQClf, TFCQClf
from dataset import SQuADQANet
from trainer import trainer, lr_scheduler_func


def main():
    datasetVersion = "v1"
    glove_dim = 300
    char_dim = 200
    dim = 128
    batch_size = 32
    glove_version = "6B"
    lr = 1e-3
    squadTrain = SQuADQANet("train", version=datasetVersion, glove_version=glove_version, glove_dim=glove_dim)
    squadVal = SQuADQANet("validation", version=datasetVersion, glove_version=glove_version, glove_dim=glove_dim)
    subsetTrain = squadTrain
    subsetVal = squadVal
    # subsetTrain = Subset(squadTrain, [i for i in range(32)])
    # subsetVal = Subset(squadVal, [i for i in range(32)])
    trainLoader = DataLoader(subsetTrain, batch_size=batch_size, shuffle=True)
    valLoader = DataLoader(subsetVal, batch_size=batch_size, shuffle=False)
  
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # model = QANet(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, freeze=True)
    # model = InputEmbedClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim)
    # model = EmbedEncClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim, with_mask=False, version=datasetVersion)
    # model = CQClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim)
    # model = MACQClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim, with_mask=True)
    model = TFCQClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim, with_mask=True, version=datasetVersion)
    
    print(f"Model parameters: {model.count_params()}")
    model.to(device)

    optimizer = optim.Adam(
        model.parameters(),
        betas=(0.8, 0.999),
        eps=1e-7,
        lr=lr,
    )

    # exponential moving average
    ema = None
    # ema = EMA(0.9999)
    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         ema.register(name, param.data)
    
    # lr_scheduler = None
    warm_up_iters = 1000
    lr_func = lr_scheduler_func(warm_up_iters=warm_up_iters)
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_func)
    criterion = nn.CrossEntropyLoss()

    log = wandb.init(
        project="qanet",
        notes="Implementation of qanet",
        config={
            "dataset_version": datasetVersion,
            "glove_version": glove_version,
            "glove_dim": glove_dim, 
            "char_dim": char_dim, 
            "dim": dim, 
            "batch_size": batch_size, 
            "warm_up_iters": warm_up_iters, 
            "base_lr": lr
        }
    )
    trainer(30, trainLoader, valLoader, model, criterion, optimizer, lr_scheduler, device, ema, log)

    