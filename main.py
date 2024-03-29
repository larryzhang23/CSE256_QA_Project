import random
import copy
import numpy as np
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
import torch.optim as optim
from model import InputEmbedClf, EmbedEncClf, MACQClf, TFCQClf, QANet, EMA
from dataset import SQuADQANet
from trainer import trainer, lr_scheduler_func


def main():
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    questionLen = 50
    exp_name = "TFNET"
    datasetVersion = "v1"
    glove_dim = 300
    char_dim = 200
    dim = 128
    batch_size = 32
    glove_version = "42B"
    lr = 1e-3
    dropout = 0.0
    squadTrain = SQuADQANet(version=datasetVersion, glove_version=glove_version, glove_dim=glove_dim)
    print(f"Training samples: {len(squadTrain)}")
    squadVal = copy.deepcopy(squadTrain)
    squadVal.setSplit("validation")
    print(f"Validation samples: {len(squadVal)}")
    del squadVal.train_dataset
    del squadVal.train_spans
    del squadVal.train_index
    del squadVal.trainLegalDataIdx
    del squadTrain.val_dataset
    del squadTrain.val_spans
    del squadTrain.val_index
    del squadTrain.valLegalDataIdx
    
    trainLoader = DataLoader(squadTrain, batch_size=batch_size, shuffle=True, pin_memory=True)
    valLoader = DataLoader(squadVal, batch_size=batch_size, shuffle=False)
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    # model = InputEmbedClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim)
    # model = EmbedEncClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim, with_mask=False, version=datasetVersion)
    # model = CQClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim)
    # model = MACQClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim, with_mask=True, gloveVersion=glove_version, dropout=dropout)
    # model = TFCQClf(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, dim=dim, with_mask=True, version=datasetVersion, gloveVersion=glove_version, dropout=dropout, questionMaxLen=questionLen)
    model = QANet(numChar=squadTrain.charSetSize, dimChar=char_dim, dimGlove=glove_dim, freeze=True, gloveVersion=glove_version, dropout=dropout, with_mask=True)
    
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

    with wandb.init(
        project="qanet",
        name=exp_name,
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
    ) as run:
        trainer(30, trainLoader, valLoader, model, criterion, optimizer, lr_scheduler, device, ema)


if __name__ == "__main__":
    main()