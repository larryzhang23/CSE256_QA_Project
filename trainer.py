import math
import torch
import wandb
from eval import get_em, get_em_max, get_f1_score, get_f1_score_max
from model import predict

def train_one_epoch(epoch, trainLoader, model, lossFunc, optimizer, lr_scheduler, device, ema=None):
    avg_acc, avg_f1, avg_loss = 0, 0, 0
    model.train()

    total_steps = epoch * len(trainLoader)
    for it, (contextDict, questionDict, target) in enumerate(trainLoader):
        target_start = target[:, 0].to(device, non_blocking=True)
        target_end = target[:, 1].to(device, non_blocking=True)
        contextDict["wordIdx"] = contextDict["wordIdx"].to(device, non_blocking=True)
        contextDict["charIdx"] = contextDict["charIdx"].to(device, non_blocking=True)
        questionDict["wordIdx"] = questionDict["wordIdx"].to(device, non_blocking=True)
        questionDict["charIdx"] = questionDict["charIdx"].to(device, non_blocking=True)

        optimizer.zero_grad()
        pred_start, pred_end = model(contextDict, questionDict)
        loss_start = lossFunc(pred_start, target_start)
        loss_end = lossFunc(pred_end, target_end)
        
        loss = (loss_start + loss_end) / 2
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)
        optimizer.step()
        best_pred_start, best_pred_end = predict(pred_start, pred_end)
        acc = get_em(best_pred_start, target_start, best_pred_end, target_end)
        f1 = get_f1_score(best_pred_start, target_start, best_pred_end, target_end)
        avg_loss += loss.item()
        avg_acc += acc
        avg_f1 += f1
        lr = get_lr(optimizer)
        wandb.log({"train_loss": loss.item(), "train_em_acc": acc, "train_f1_score": f1, "lr": lr})
        if it > 0 and it % 20 == 0:
            print(f"[Epoch:{epoch}/{it}] -- loss: {loss.item():.4f} -- EM acc: {(acc * 100):.2f}% -- F1 score: {f1:.3f} -- lr: {lr:.4f}")

        if lr_scheduler is not None:
            lr_scheduler.step()

        if ema is not None:
            ema(model, total_steps)
        total_steps += 1


    avg_acc /= len(trainLoader)
    avg_f1 /= len(trainLoader)
    avg_loss /= len(trainLoader)
    print("=================")
    print(f"[Epoch:{epoch}] -- Train avg loss: {avg_loss:.4f} -- Train avg EM acc: {(avg_acc * 100):.2f}% -- Train avg F1 score: {avg_f1:.3f}")

    ### debug ###
    if epoch > 0 and epoch % 5 == 0:
        for it, (contextDict, questionDict, target) in enumerate(trainLoader):
            contextDict["wordIdx"] = contextDict["wordIdx"].to(device, non_blocking=True)
            contextDict["charIdx"] = contextDict["charIdx"].to(device, non_blocking=True)
            questionDict["wordIdx"] = questionDict["wordIdx"].to(device, non_blocking=True)
            questionDict["charIdx"] = questionDict["charIdx"].to(device, non_blocking=True)
            with torch.no_grad():
                pred_start, pred_end = model(contextDict, questionDict)
                pred_start = torch.argmax(pred_start, dim=1)
                pred_end = torch.argmax(pred_end, dim=1)
                pred_start = pred_start.cpu().numpy().tolist()
                pred_end = pred_end.cpu().numpy().tolist()
                print("pred: ", list(zip(pred_start, pred_end)))
                print("target: ", target.numpy().tolist())
            if it >= 0:
                break
                
    train_dict = {"train_avg_loss": avg_loss, "train_avg_em_acc": avg_acc, "train_avg_f1_score": avg_f1}
    return train_dict

def validate(epoch, valLoader, model, device, ema=None):
    avg_acc, avg_f1 = 0, 0
    model.eval()
    print(f"======== Begin Validation at Epoch {epoch} ===========")
    with torch.no_grad():
        for it, (contextDict, questionDict, targets) in enumerate(valLoader):
            targets = targets.to(device, non_blocking=True)
            contextDict["wordIdx"] = contextDict["wordIdx"].to(device, non_blocking=True)
            contextDict["charIdx"] = contextDict["charIdx"].to(device, non_blocking=True)
            questionDict["wordIdx"] = questionDict["wordIdx"].to(device, non_blocking=True)
            questionDict["charIdx"] = questionDict["charIdx"].to(device, non_blocking=True)

            pred_start, pred_end = model(contextDict, questionDict)
            pred_start, pred_end = predict(pred_start, pred_end)
            acc = get_em_max(pred_start, pred_end, targets)
            f1 = get_f1_score_max(pred_start, pred_end, targets)
            avg_acc += acc
            avg_f1 += f1
            # print(f"[Epoch:{epoch}/{it}] --acc: {(acc * 100):.2f}% --f1 score: {f1:.3f}")

            ## debug
            if it == 0:
                pred_start, pred_end = model(contextDict, questionDict)
                pred_start, pred_end = predict(pred_start, pred_end)
                pred_start = pred_start.cpu().numpy().tolist()
                pred_end = pred_end.cpu().numpy().tolist()
                print("pred: ", list(zip(pred_start, pred_end)))
                print("target: ", targets.cpu().numpy().tolist())
            

    avg_acc /= len(valLoader)
    avg_f1 /= len(valLoader)
    print(f"[Epoch:{epoch}] -- Val avg EM acc: {(avg_acc * 100):.2f}% -- Val avg F1 score: {avg_f1:.3f}")
    val_dict = {"val_avg_em_acc": avg_acc, "val_avg_f1_score": avg_f1}
    wandb.log(val_dict)
    return val_dict

def trainer(epochs, trainLoader, valLoader, model, lossFunc, optimizer, lr_scheduler, device, ema=None):
    for epoch in range(epochs):
        val_stats = validate(epoch, valLoader, model, device)
        train_stats = train_one_epoch(epoch, trainLoader, model, lossFunc, optimizer, lr_scheduler, device, ema)
        
        

def lr_scheduler_func(warm_up_iters=1000):
    maxVal = 1 / math.log(warm_up_iters)
    func = lambda iters: maxVal * math.log(iters + 1) if iters < warm_up_iters else 1.0
    return func

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']