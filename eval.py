import torch

def get_em(pred_start, target_start, pred_end, target_end):
    pred_start_idx = torch.argmax(pred_start.detach(), dim=1)
    pred_end_idx = torch.argmax(pred_end.detach(), dim=1)
    correct_start = pred_start_idx == target_start
    correct_end = pred_end_idx == target_end
    correct = torch.logical_and(correct_start, correct_end)
    acc = torch.sum(correct).item() / len(pred_start_idx)
    return acc


def get_em_max(pred_start, pred_end, targets):
    acc = 0
    for i in range(6):
        target = targets[:, i]
        acc = max(acc, get_em(pred_start, target[:, 0], pred_end, target[:, 1]))
    return acc
        

def get_f1_score(pred_start, target_start, pred_end, target_end):
    pred_start = torch.argmax(pred_start.detach(), dim=1)
    pred_end = torch.argmax(pred_end.detach(), dim=1)
    sample_num = len(pred_start)
    start_max = torch.stack([pred_start, target_start], dim=1).max(dim=1).values
    end_min = torch.stack([pred_end, target_end], dim=1).min(dim=1).values
    overlap_len = end_min - start_max
    gt_len = target_end - target_start 
    pred_len = pred_end - pred_start
    # handle zero length pred_len or zero length gt_len
    unanswerable = torch.logical_or(gt_len == 0, pred_len == 0)
    pred_start_una, pred_end_una = pred_start[unanswerable], pred_end[unanswerable]
    target_start_una, target_end_una = target_start[unanswerable], target_end[unanswerable]
    f1_una = torch.logical_and(pred_start_una == target_start_una, pred_end_una == target_end_una).sum().item()
    # handle normal cases
    answerable = torch.logical_and(gt_len != 0, pred_len != 0)
    gt_len, pred_len, overlap_len = gt_len[answerable], pred_len[answerable], overlap_len[answerable]
    overlap_len = overlap_len.masked_fill(overlap_len < 0, 0)
    # ignore zero overlapped length cases since f1 only zero
    overlap_mask = overlap_len != 0
    gt_len, pred_len, overlap_len = gt_len[overlap_mask], pred_len[overlap_mask], overlap_len[overlap_mask]
    precision = overlap_len / pred_len
    recall = overlap_len / gt_len
    f1_normal = ((2 * precision * recall) / (precision + recall)).sum().item()
    return (f1_normal + f1_una) / sample_num


def get_f1_score_max(pred_start, pred_end, targets):
    f1 = 0
    for i in range(6):
        target = targets[:, i]
        f1 = max(f1, get_f1_score(pred_start, target[:, 0], pred_end, target[:, 1]))
    return f1
    
