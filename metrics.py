import torch


def calc_metrics(pred_mask, true_mask):
    "pred_mask is the predicted mask, true_mask is ground truth"
    # Calculate the intersection and union between the two masks
    intersection = torch.sum(pred_mask * true_mask)
    union = torch.sum(pred_mask) + torch.sum(true_mask) - intersection

    iou = intersection / union
    dice = (2 * intersection) / (union + intersection)

    precision = intersection / torch.sum(pred_mask)
    recall = intersection / torch.sum(true_mask)

    return iou, dice, precision, recall
