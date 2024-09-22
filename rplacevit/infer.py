import torch
import numpy as np
import cv2
import argparse
from PIL import Image
from tqdm import tqdm
from palette import rplace_palette
from model import RPlaceColorTransformerV2, RPlaceTimeTransformer
from utils import trainable_parameters

PALETTE_ARRAY = np.array([color for color in rplace_palette])

def find_closest_color(rgba_color: np.ndarray) -> int:
    """
    Find the closest color in the R/Place palette to the given RGBA color.
    
    Args:
        rgba_color (np.ndarray): RGBA color to find the closest color for
        
    Returns:
        int: Index of the closest color in the R/Place palette
    """
    distances = np.sqrt(np.sum((np.array([color for color in rplace_palette]) - rgba_color) ** 2, axis=1))
    return np.argmin(distances)

def load_model(model_class: torch.nn.Module, checkpoint_path: str, args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    """
    Load a model from a checkpoint file.
    
    Args:
        model_class (torch.nn.Module): Model class to load
        checkpoint_path (str): Path to the checkpoint file
        args (argparse.Namespace): Command-line arguments
        device (torch.device): Device to move the model to
        
    Returns:
        torch.nn.Module: Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    if model_class == RPlaceColorTransformerV2:
        model = model_class(
            in_channels=args.color_in_channels,
            d_model=args.color_d_model,
            num_heads=args.color_num_heads,
            num_blocks=args.color_num_blocks,
            d_ff=args.color_d_ff,
            dropout=args.color_dropout,
            center_width=args.color_center_width,
            peripheral_width=args.color_peripheral_width,
            use_peripheral=args.color_use_peripheral,
            output_strategy=args.color_output_strategy
        )
    else:
        model = RPlaceTimeTransformer(
            in_channels=args.time_in_channels,
            d_model=args.time_d_model,
            num_heads=args.time_num_heads,
            num_blocks=args.time_num_blocks,
            d_ff=args.time_d_ff,
            dropout=args.time_dropout,
            center_width=args.time_center_width,
            peripheral_width=args.time_peripheral_width,
            use_peripheral=args.time_use_peripheral
        )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"{model_class.__name__} has {trainable_parameters(model):,} trainable parameters")
    return model.to(device)

def initialize_canvas(initial_image_path: str, canvas_size: tuple) -> np.ndarray:
    """
    Initialize the canvas with the given initial image or a blank canvas.
    
    Args:
        initial_image_path (str): Path to the initial image
        canvas_size (tuple): Size of the canvas (height, width)
        
    Returns:
        np.ndarray: Initialized canvas
    """
    if initial_image_path:
        img = Image.open(initial_image_path).convert('RGBA').crop((0, 0, canvas_size[1], canvas_size[0]))
        canvas = np.zeros(canvas_size, dtype=np.uint8)
        img_array = np.array(img)
        for y in range(canvas_size[0]):
            for x in range(canvas_size[1]):
                color = img_array[y, x]
                color_id = find_closest_color(color)
                canvas[y, x] = color_id
    else:
        canvas = np.full(canvas_size, 31, dtype=np.uint8)
    return canvas

def get_view(canvas: np.ndarray, view_size: tuple, offset_y: int, offset_x: int) -> np.ndarray:
    """
    Get a view from the canvas with the given size and offset.
    
    Args:
        canvas (np.ndarray): Canvas to get the view from
        view_size (tuple): Size of the view (height, width)
        offset_y (int): Y offset of the view
        offset_x (int): X offset of the view
        
    Returns:
        np.ndarray: View from the canvas
    """
    return canvas[offset_y:offset_y+view_size[0], offset_x:offset_x+view_size[1]]

@torch.no_grad()
def predict(model: torch.nn.Module, view: np.ndarray, device: torch.device) -> np.ndarray:
    """
    Predict the next color / time grid for the given view using the model.
    
    Args:
        model (torch.nn.Module): Model to use for prediction
        view (np.ndarray): View to predict the next color / time for
        device (torch.device): Device to move the input tensor to
        
    Returns:
        np.ndarray: Predicted color / time
    """
    one_hot_view = np.eye(32)[view]
    input_tensor = torch.from_numpy(one_hot_view).float().permute(2, 0, 1).unsqueeze(0).to(device)
    return model(input_tensor).squeeze().cpu().numpy()

def choose_pixel_to_change(time_prediction: np.ndarray) -> tuple:
    """
    Choose a pixel to change based on the time prediction.
    
    Args:
        time_prediction (np.ndarray): Time prediction
        
    Returns:
        tuple: Pixel to change
    """
    inverted_prediction = 1 - time_prediction
    combined_prediction = inverted_prediction + np.random.uniform(0, 0.1, inverted_prediction.shape)
    return np.unravel_index(np.argmax(combined_prediction), combined_prediction.shape)

def choose_random_pixel(view_size: tuple) -> tuple:
    """
    Choose a random pixel from the view.
    
    Args:
        view_size (tuple): Size of the view (height, width)
        
    Returns:
        tuple: Random pixel
    """
    return (np.random.randint(0, view_size[0]), np.random.randint(0, view_size[1]))

def choose_color(color_prediction: np.ndarray) -> tuple:
    """
    Choose a color based on the color prediction.   
    
    Args:
        color_prediction (np.ndarray): Color prediction
        
    Returns:
        tuple: Chosen color and probability
    """
    chosen_color = np.argmax(color_prediction)
    probability = color_prediction[chosen_color]
    return chosen_color, probability

def main():
    parser = argparse.ArgumentParser(description='Generate R/Place video with optional initial image.')
    parser.add_argument('--initial_image', type=str, help='Path to initial PNG image')
    parser.add_argument('--random_view', action='store_true', help='Use random view selection instead of time transformer')
    parser.add_argument('--color_model_path', type=str, required=True, help='Path to color model checkpoint')
    parser.add_argument('--time_model_path', type=str, help='Path to time model checkpoint')
    parser.add_argument('--canvas_size', type=int, nargs=2, default=[256, 256], help='Canvas size (height width)')
    parser.add_argument('--num_iterations', type=int, default=65536, help='Number of iterations')
    parser.add_argument('--fps', type=int, default=30, help='Frames per second in output video')
    parser.add_argument('--frame_interval', type=int, default=32, help='Number of iterations between frames')
    
    parser.add_argument('--color_in_channels', type=int, default=32, help='Number of input channels for color model')
    parser.add_argument('--color_d_model', type=int, default=512, help='Dimension of the color model')
    parser.add_argument('--color_num_heads', type=int, default=8, help='Number of attention heads for color model')
    parser.add_argument('--color_num_blocks', type=int, default=6, help='Number of transformer blocks for color model')
    parser.add_argument('--color_d_ff', type=int, default=2048, help='Dimension of the feed-forward network for color model')
    parser.add_argument('--color_dropout', type=float, default=0.1, help='Dropout rate for color model')
    parser.add_argument('--color_center_width', type=int, default=16, help='Width of the center view for color model')
    parser.add_argument('--color_peripheral_width', type=int, default=64, help='Width of the peripheral view for color model')
    parser.add_argument('--color_use_peripheral', action='store_true', help='Use peripheral view for color model')
    parser.add_argument('--color_output_strategy', type=str, default='cls_token', choices=['cls_token', 'avg', 'center'], help='Output strategy for color model')
    
    parser.add_argument('--time_in_channels', type=int, default=32, help='Number of input channels for time model')
    parser.add_argument('--time_d_model', type=int, default=256, help='Dimension of the time model')
    parser.add_argument('--time_num_heads', type=int, default=4, help='Number of attention heads for time model')
    parser.add_argument('--time_num_blocks', type=int, default=4, help='Number of transformer blocks for time model')
    parser.add_argument('--time_d_ff', type=int, default=1024, help='Dimension of the feed-forward network for time model')
    parser.add_argument('--time_dropout', type=float, default=0.1, help='Dropout rate for time model')
    parser.add_argument('--time_center_width', type=int, default=16, help='Width of the center view for time model')
    parser.add_argument('--time_peripheral_width', type=int, default=64, help='Width of the peripheral view for time model')
    parser.add_argument('--time_use_peripheral', action='store_true', help='Use peripheral view for time model')
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    color_model = load_model(RPlaceColorTransformerV2, args.color_model_path, args, device)
    
    if not args.random_view:
        if args.time_model_path is None:
            raise ValueError("Time model path must be provided when not using random view selection")
        time_model = load_model(RPlaceTimeTransformer, args.time_model_path, args, device)

    canvas_size = tuple(args.canvas_size)
    color_view_size = (args.color_center_width, args.color_center_width)
    time_view_size = (args.time_center_width, args.time_center_width)

    canvas = initialize_canvas(args.initial_image, canvas_size)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter('rplace_generation.mp4', fourcc, args.fps, canvas_size)

    frames = []
    for iteration in tqdm(range(args.num_iterations)):
        max_offset_x = canvas_size[1] - max(color_view_size[1], time_view_size[1])
        max_offset_y = canvas_size[0] - max(color_view_size[0], time_view_size[0])
        offset_x = np.random.randint(0, max_offset_x + 1)
        offset_y = np.random.randint(0, max_offset_y + 1)
        
        if args.random_view:
            relative_y, relative_x = choose_random_pixel(color_view_size)
            pixel_y = offset_y + relative_y
            pixel_x = offset_x + relative_x
        else:
            time_view = get_view(canvas, time_view_size, offset_y, offset_x)
            time_prediction = predict(time_model, time_view, device)
            relative_y, relative_x = choose_pixel_to_change(time_prediction)
            
            pixel_y = offset_y + relative_y
            pixel_x = offset_x + relative_x
        
        new_offset_y = max(0, min(pixel_y - color_view_size[0] // 2, canvas_size[0] - color_view_size[0]))
        new_offset_x = max(0, min(pixel_x - color_view_size[1] // 2, canvas_size[1] - color_view_size[1]))
        
        color_view = get_view(canvas, color_view_size, new_offset_y, new_offset_x)
        color_prediction = predict(color_model, color_view, device)
        
        chosen_color, probability = choose_color(color_prediction)

        canvas[pixel_y, pixel_x] = chosen_color
        
        if iteration % args.frame_interval == 0:
            frame = PALETTE_ARRAY[canvas, :3].astype(np.uint8)
            video.write(frame[..., ::-1])
            frames.append(frame)

    video.release()

    images = [Image.fromarray(frame) for frame in frames]
    images[0].save("rplace_generation.gif", save_all=True, append_images=images[1:], duration=1000/args.fps, loop=0)

if __name__ == "__main__":
    main()