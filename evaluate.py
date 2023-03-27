import torch
import numpy as np
import torchvision


def evaluate(model, data_loader, iou_threshold=0.5, dice_threshold=0.5):
    model.eval()
    with torch.no_grad():
        iou_scores = []
        dice_scores = []
        all_boxes = []
        all_scores = []
        all_labels = []

        for images, annotations in data_loader:
            images = list(image.to(device) for image in images)

            targets = []
            for i in range(len(images)):
                d = {}
                d["boxes"] = annotations["boxes"][i]
                d["labels"] = annotations["labels"][i]
                targets.append(d)

            # Forward pass
            outputs = model(images)

            for i, output in enumerate(outputs):
                boxes = output["boxes"].detach().cpu().numpy()
                scores = output["scores"].detach().cpu().numpy()
                labels = output["labels"].detach().cpu().numpy()

                # Calculate IoU
                iou = calculate_iou(targets[i]["boxes"], boxes)
                iou_scores.append(iou)

                # Calculate Dice score
                dice = calculate_dice(targets[i]["boxes"], boxes, dice_threshold)
                dice_scores.append(dice)

                all_boxes.append(boxes)
                all_scores.append(scores)
                all_labels.append(labels)

        # Calculate mAP@50 and mAP@90
        all_boxes = np.concatenate(all_boxes)
        all_scores = np.concatenate(all_scores)
        all_labels = np.concatenate(all_labels)

        mAP50 = calculate_mAP(all_boxes, all_scores, all_labels, iou_threshold=0.5)
        mAP90 = calculate_mAP(all_boxes, all_scores, all_labels, iou_threshold=0.9)

        return mAP50, mAP90, np.mean(iou_scores), np.mean(dice_scores)


def calculate_iou(gt_box, pred_box):
    # Calculate intersection area
    x1 = np.maximum(gt_box[0], pred_box[:, 0])
    y1 = np.maximum(gt_box[1], pred_box[:, 1])
    x2 = np.minimum(gt_box[2], pred_box[:, 2])
    y2 = np.minimum(gt_box[3], pred_box[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate union area
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
    union = gt_area + pred_area - intersection

    # Calculate IoU
    iou = intersection / union
    return np.max(iou)


def calculate_dice(gt_box, pred_box, threshold):
    # Calculate intersection area
    x1 = np.maximum(gt_box[0], pred_box[:, 0])
    y1 = np.maximum(gt_box[1], pred_box[:, 1])
    x2 = np.minimum(gt_box[2], pred_box[:, 2])
    y2 = np.minimum(gt_box[3], pred_box[:, 3])
    intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    # Calculate Dice score
    gt_area = (gt_box[2] - gt_box[0]) * (gt_box[3] - gt_box[1])
    pred_area = (pred_box[:, 2] - pred_box[:, 0]) * (pred_box[:, 3] - pred_box[:, 1])
    union = gt_area + pred_area
    dice = 2 * intersection / (union + 1e-8)

    return np.max(dice) if len(dice) > 0 and np.max(dice) >= threshold else 0.0
