import math
import torch
from dataset import SQuADQANet

def get_accuracy(pred_start, target_start, pred_end, target_end):
    pred_start_idx = torch.argmax(pred_start.detach(), dim=1)
    pred_end_idx = torch.argmax(pred_end.detach(), dim=1)
    correct_start = pred_start_idx == target_start
    correct_end = pred_end_idx == target_end
    correct = torch.logical_and(correct_start, correct_end)
    acc = torch.sum(correct).item() / len(pred_start_idx)
    return acc


def train_one_epoch(epoch, trainLoader, model, lossFunc, optimizer, lr_scheduler, device):
    avg_acc, avg_loss = 0, 0
    model.train()
    for it, (contextDict, questionDict, target) in enumerate(trainLoader):
        target_start = target[:, 0].to(device, non_blocking=True)
        target_end = target[:, 1].to(device, non_blocking=True)
        contextDict["wordIdx"] = contextDict["wordIdx"].to(device, non_blocking=True)
        contextDict["charIdx"] = contextDict["charIdx"].to(device, non_blocking=True)
        questionDict["wordIdx"] = questionDict["wordIdx"].to(device, non_blocking=True)
        questionDict["charIdx"] = questionDict["charIdx"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred_start, pred_end = model(contextDict, questionDict)
        try:
            loss_start = lossFunc(pred_start, target_start)
            loss_end = lossFunc(pred_end, target_end)
        except:
            print(target_start)
            print(pred_start)
            return
        loss = loss_start + loss_end
        loss.backward()
        optimizer.step()
        acc = get_accuracy(pred_start, target_start, pred_end, target_end)
        avg_loss += loss.item()
        avg_acc += acc
        print(f"[Epoch:{epoch}/{it}] -- loss: {loss.item():.4f} -- acc: {(acc * 100):.2f}%")
        lr_scheduler.step()
    avg_acc /= len(trainLoader)
    avg_loss /= len(trainLoader)
    print("=================")
    print(f"[Epoch:{epoch}] -- avg loss: {avg_loss:.4f} -- avg acc: {(avg_acc * 100):.2f}% --lr {lr_scheduler.get_last_lr()}")
    return {"avg_loss": avg_loss, "avg_acc": avg_acc}

def trainer(epochs, trainLoader, model, lossFunc, optimizer, lr_scheduler, device):
    for epoch in range(epochs):
        stats = train_one_epoch(epoch, trainLoader, model, lossFunc, optimizer, lr_scheduler,device)

def lr_scheduler_func(warm_up_iters=1000):
    maxVal = 1 / math.log(warm_up_iters)
    func = lambda iters: maxVal * math.log(iters + 1) if iters < warm_up_iters else 1.0
    return func
