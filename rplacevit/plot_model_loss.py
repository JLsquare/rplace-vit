import argparse
import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_losses(checkpoint_path, avg_window=100):
    """
    Plot the training losses from a checkpoint file.
    
    Args:
        checkpoint_path (str): Path to the checkpoint file
        avg_window (int): Number of steps to average over (default: 100)
    """
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    
    losses = checkpoint.get('losses', [])
    
    if not losses:
        print("No loss data found in the checkpoint.")
        return
    
    avg_losses = []
    for i in range(0, len(losses), avg_window):
        avg_losses.append(np.mean(losses[i:i+avg_window]))
    
    steps = list(range(0, len(losses), avg_window))
    
    plt.figure(figsize=(12, 6))
    plt.plot(steps, avg_losses)
    plt.title(f'Training Loss Over Time (Averaged every {avg_window} steps)')
    plt.xlabel('Step')
    plt.ylabel('Average Loss')
    plt.grid(True)
    
    output_file = 'loss_plot.png'
    plt.savefig(output_file)
    print(f"Plot saved as {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot training losses from a checkpoint file.")
    parser.add_argument("checkpoint_path", type=str, help="Path to the checkpoint file")
    parser.add_argument("--avg_window", type=int, default=100, help="Number of steps to average over (default: 100)")
    args = parser.parse_args()
    
    plot_losses(args.checkpoint_path, args.avg_window)