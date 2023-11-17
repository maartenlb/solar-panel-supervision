import torch


preds = torch.tensor(
    [[0.2, 0.3, 0.0, 0.7, 0.5], [0.1, 0.2, 0.3, 0.4, 0.5], [0.5, 0.2, 0.1, 0.8, 0.2]]
)
targets = torch.tensor([3, 4, 1], dtype=torch.long)

hot_targets = torch.nn.functional.one_hot(targets, num_classes=5).to(dtype=torch.float)

weights = torch.tensor([1, 0.75, 0.2, 1, 0.1], dtype=torch.float)

loss_function = torch.nn.functional.cross_entropy

num = loss_function(preds, targets)

print(f"Loss for no weights normal: {num.item()}")

num = loss_function(preds, hot_targets)

print(f"Loss for no weights one hot: {num.item()}")

num = loss_function(preds, targets, weights)

print(f"Loss for weighted normal: {num.item()}")

num = loss_function(preds, hot_targets, weights)

print(f"Loss for weighted one hot: {num.item()}")
