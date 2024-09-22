import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss, MSELoss
from torch.optim import AdamW
from torch.nn.parallel import DataParallel
from model import RPlaceTimeTransformer, RPlaceColorTransformerV2
from dataset import RPlaceColorDataset, RPlaceTimeDataset
from train import train_model
from utils import trainable_parameters

def setup_argparse():
    """
    Setup the argparse parser for training R/Place models.
    
    Returns:
        parser: The argparse parser
    """
    parser = argparse.ArgumentParser(description="Train R/Place models")
    parser.add_argument("--model", type=str, choices=["color", "time"], required=True, help="Model type to train")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=4, help="Number of epochs to train")
    parser.add_argument("--d_model", type=int, default=512, help="Dimension of the model")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--num_blocks", type=int, default=12, help="Number of transformer blocks")
    parser.add_argument("--d_ff", type=int, default=2048, help="Dimension of the feed-forward network")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout rate")
    parser.add_argument("--center_width", type=int, default=16, help="Width of the center view")
    parser.add_argument("--peripheral_width", type=int, default=64, help="Width of the peripheral view")
    parser.add_argument("--use_peripheral", action="store_true", help="Use peripheral view")
    parser.add_argument("--output_strategy", type=str, default="cls_token", choices=["cls_token", "avg", "center"], help="Output strategy for color model")
    parser.add_argument("--single_partition", type=int, default=None, help="Single partition to use")
    parser.add_argument("--use_user_features", action="store_true", help="Use user features")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Path to checkpoint to resume training")
    parser.add_argument("--min_x", type=int, default=0, help="Minimum x-coordinate of the region of interest")
    parser.add_argument("--min_y", type=int, default=0, help="Minimum y-coordinate of the region of interest")
    parser.add_argument("--max_x", type=int, default=2048, help="Maximum x-coordinate of the region of interest")
    parser.add_argument("--max_y", type=int, default=2048, help="Maximum y-coordinate of the region of interest")
    
    return parser

def setup_training(args):
    """
    Setup the training environment for the model.
    
    Args:
        args: Parsed arguments from argparse
        
    Returns:
        model: The model to train
        dataloader: DataLoader for the training dataset
        criterion: Loss function
        optimizer: Optimizer for training
        device: Device to use for training
        model_path: Path to save model checkpoints
        save_every: Number of steps to save a checkpoint
        log_every: Number of steps to log training information
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        num_gpus = torch.cuda.device_count()
        print(f"Using {num_gpus} GPUs")
    else:
        device = torch.device("cpu")
        num_gpus = 1
        print("CUDA is not available. Using CPU")

    if args.model == "color":
        dataset = RPlaceColorDataset(
            single_partition=args.single_partition,
            use_user_features=args.use_user_features,
            view_width=args.center_width,
            min_x=args.min_x,
            min_y=args.min_y,
            max_x=args.max_x,
            max_y=args.max_y
        )
        criterion = CrossEntropyLoss()
    else:
        dataset = RPlaceTimeDataset(
            single_partition=args.single_partition,
            use_user_features=args.use_user_features,
            view_width=args.center_width,
            min_x=args.min_x,
            min_y=args.min_y,
            max_x=args.max_x,
            max_y=args.max_y
        )
        criterion = MSELoss()

    dataset.load_end_states()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    model_params = {
        "in_channels": 32,
        "d_model": args.d_model,
        "num_heads": args.num_heads,
        "num_blocks": args.num_blocks,
        "d_ff": args.d_ff,
        "dropout": args.dropout,
        "center_width": args.center_width,
        "peripheral_width": args.peripheral_width,
        "use_peripheral": args.use_peripheral,
    }

    if args.model == "color":
        model = RPlaceColorTransformerV2(**model_params, output_strategy=args.output_strategy)
    else:
        model = RPlaceTimeTransformer(**model_params)

    model.to(device)
    model.init_weights()

    if num_gpus > 1:
        model = DataParallel(model)

    print(f"Trainable parameters: {trainable_parameters(model)}")

    optimizer = AdamW(model.parameters(), lr=args.learning_rate)

    model_path = f"./{args.model}_model_checkpoints"
    save_every = 1000
    log_every = 100

    return model, dataloader, criterion, optimizer, device, model_path, save_every, log_every

def main():
    parser = setup_argparse()
    args = parser.parse_args()

    model, dataloader, criterion, optimizer, device, model_path, save_every, log_every = setup_training(args)

    train_loss, total_time = train_model(
        model=model,
        train_loader=dataloader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        model_path=model_path,
        num_epochs=args.num_epochs,
        save_every=save_every,
        log_every=log_every,
        checkpoint_path=args.checkpoint_path,
    )

    print(f"Training completed in {total_time:.2f} seconds")
    return model, train_loss

if __name__ == "__main__":
    trained_model, losses = main()
