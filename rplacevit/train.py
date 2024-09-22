import os
import time
import torch
from tqdm import tqdm
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from collections import deque

def train_epoch(
    model: torch.nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: str,
    scaler: GradScaler,
    epoch: int,
    num_epochs: int,
    global_step: int,
    recent_losses: deque,
    save_every: int,
    log_every: int,
    model_path: str,
    losses: list
) -> tuple:
    """
    Train the model for one epoch
    
    Args:
        model (torch.nn.Module): The model to train
        train_loader (torch.utils.data.DataLoader): The training data
        criterion (torch.nn.Module): The loss function
        optimizer (torch.optim.Optimizer): The optimizer
        device (str): The device to train on
        scaler (GradScaler): The gradient scaler
        epoch (int): The current epoch
        num_epochs (int): The total number of epochs
        global_step (int): The current global step
        recent_losses (deque): A deque to store the recent losses
        save_every (int): Save a checkpoint every n steps
        log_every (int): Log the loss every n steps
        model_path (str): The path to save the model checkpoints
        losses (list): A list to store the losses
    """
    model.train()
    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
    
    for image, label in pbar:
        image, label = image.to(device), label.to(device)
        optimizer.zero_grad()

        with autocast('cuda'):
            pred = model(image)
            loss = criterion(pred, label)

        scaler.scale(loss).backward()
        clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        global_step += 1
        recent_losses.append(loss.item())
        losses.append(loss.item())
        mean_loss = sum(recent_losses) / len(recent_losses)

        pb_str = f"Loss: {loss.item():.4f}, Mean loss: {mean_loss:.4f}"
        pbar.set_postfix_str(pb_str)

        if global_step % log_every == 0:
            print(f"Epoch {epoch+1}, Step {global_step}: Loss: {loss.item():.4f}, Mean loss: {mean_loss:.4f}")

        if save_every != 0 and global_step % save_every == 0:
            save_checkpoint(model, optimizer, scaler, epoch, global_step, recent_losses, model_path, losses)

    return global_step, recent_losses, losses

def save_checkpoint(model, optimizer, scaler, epoch, global_step, recent_losses, model_path, losses):
    """
    Save a checkpoint of the model
    
    Args:
        model (torch.nn.Module): The model to save
        optimizer (torch.optim.Optimizer): The optimizer to save
        scaler (GradScaler): The gradient scaler to save
        epoch (int): The current epoch
        global_step (int): The current global step
        recent_losses (deque): A deque to store the recent losses
        model_path (str): The path to save the model
        losses (list): A list to store the losses
    """
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'global_step': global_step,
        'recent_losses': list(recent_losses),
        'losses': losses
    }
    torch.save(checkpoint, os.path.join(model_path, f"checkpoint_epoch_{epoch+1}_step_{global_step}.pt"))
    print(f"Checkpoint saved at epoch {epoch+1}, step {global_step}! Mean loss over last {len(recent_losses)} steps: {sum(recent_losses)/len(recent_losses):.4f}")

def train_model(
        model: torch.nn.Module,
        train_loader: torch.utils.data.DataLoader,
        criterion: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        model_path: str,
        num_epochs: int,
        save_every: int = 0,
        log_every: int = 100,
        checkpoint_path: str | None = None,
) -> tuple:
    """
    Train a model

    Args:
        model (torch.nn.Module): The model to train
        train_loader (torch.utils.data.DataLoader): The training data
        criterion (torch.nn.Module): The loss function
        optimizer (torch.optim.Optimizer): The optimizer
        device (str): The device to train on
        model_path (str): The path to save the model
        num_epochs (int): The number of epochs to train for
        save_every (int): Save a checkpoint every n steps
        log_every (int): Log the loss every n steps
        checkpoint_path (str): The path to a checkpoint to resume training from
    """
    train_losses = []
    start_time = time.time()
    global_step = 0
    start_epoch = 0
    scaler = GradScaler()
    recent_losses = deque(maxlen=1000)
    losses = []

    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler'])
        global_step = checkpoint['global_step']
        recent_losses = deque(checkpoint['recent_losses'], maxlen=1000)
        start_epoch = checkpoint['epoch'] + 1
        losses = checkpoint.get('losses', [])
        
        print(f"Resuming training from epoch {start_epoch}, step {global_step}")
    elif checkpoint_path:
        print(f"Checkpoint not found at {checkpoint_path}, starting from scratch")

    for epoch in range(start_epoch, num_epochs):
        global_step, recent_losses, losses = train_epoch(
            model, train_loader, criterion, optimizer, device, scaler,
            epoch, num_epochs, global_step, recent_losses, save_every, log_every, model_path, losses
        )
        epoch_loss = sum(recent_losses) / len(recent_losses)
        train_losses.append(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs} completed. Average loss: {epoch_loss:.4f}")

        if save_every != 0:
            save_checkpoint(model, optimizer, scaler, epoch, global_step, recent_losses, model_path, losses)

    total_time = time.time() - start_time
    return train_losses, total_time, losses