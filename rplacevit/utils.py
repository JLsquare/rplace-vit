import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_loss(train_loss: list, val_loss: list, log_every: int):
    plt.figure(figsize=(12, 6))
    steps = range(log_every, len(train_loss) * log_every + 1, log_every)
    plt.plot(steps, train_loss, label='Training loss')
    plt.plot(steps, val_loss, label='Validation loss')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Loss per step')
    plt.show()

def plot_accuracy(train_accuracy: list, val_accuracy: list, log_every: int):
    plt.figure(figsize=(12, 6))
    steps = range(log_every, len(train_accuracy) * log_every + 1, log_every)
    plt.plot(steps, train_accuracy, label='Training accuracy')
    plt.plot(steps, val_accuracy, label='Validation accuracy')
    plt.legend()
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title('Accuracy per step')
    plt.show()

def test_accuracy(model: torch.nn.Module, testLoader: torch.utils.data.DataLoader, device: str) -> float:
    model.eval()
    correct = 0
    total = 0
    test_pbar = tqdm(testLoader, desc="Testing", leave=False)
    with torch.no_grad():
        for image, label in test_pbar:
            image, label = image.to(device), label.to(device)
            pred = model(image)
            _, predicted = torch.max(pred.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

    accuracy = correct / total
    return accuracy

def trainable_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in filter(lambda p: p.requires_grad, model.parameters()))