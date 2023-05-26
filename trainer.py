import torch
from data.dataset import SQuADQANet

def get_accuracy(pred_start, target_start, pred_end, target_end):
    """
    Work for cpu
    """
    pred_start_idx = torch.argmax(pred_start.detach(), dim=1)
    pred_end_idx = torch.argmax(pred_end.detach(), dim=1)
    correct_start = pred_start_idx == target_start 
    correct_end = pred_end_idx == target_end 
    correct = torch.logical_and(correct_start, correct_end)
    acc = torch.sum(correct).item() / len(pred_start_idx)
    return acc


def train_one_epoch(epoch, trainLoader, model, lossFunc, optimizer):
    avg_acc, avg_loss = 0, 0
    for it, (contextDict, questionDict, target) in enumerate(trainLoader):
        target_start = target[:, 0]
        target_end = target[:, 1]
        optimizer.zero_grad()
        pred_start, pred_end = model(contextDict, questionDict)
        loss_start = lossFunc(pred_start, target_start)
        loss_end = lossFunc(pred_end, target_end)
        loss = loss_start + loss_end
        loss.backward()
        optimizer.step()
        acc = get_accuracy(pred_start, target_start, pred_end, target_end)
        avg_loss += loss.item()
        avg_acc += acc
        print(f"[Epoch:{epoch}/{it}] -- loss: {loss.item():4f} -- acc: {acc:4f}")
    avg_acc /= len(trainLoader)
    avg_loss /= len(trainLoader)
    print("=================")
    print(f"[Epoch:{epoch}] -- avg loss: {avg_loss:4f} -- avg acc: {avg_acc:4f}")
    return {"avg_loss": avg_loss, "avg_acc": avg_acc}

def trainer(epochs, trainLoader, model, lossFunc, optimizer):
    for epoch in range(epochs):
        stats = train_one_epoch(epoch, trainLoader, model, lossFunc, optimizer)
    