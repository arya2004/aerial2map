# utils/image_processing.py

import os
from PIL import Image, UnidentifiedImageError
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F

def crop_interpolate_and_save(image_path, target_shape):
    """
    Loads an image, crops the left half as the 'condition', interpolates it to the target shape, 
    and saves it back to the same path.

    Args:
    - image_path (str): Path to the image file.
    - target_shape (tuple): Target shape as (height, width) for the interpolation.

    Returns:
    - None
    """
    # Validate input arguments
    if not isinstance(image_path, str) or not os.path.isfile(image_path):
        raise ValueError(f"Invalid image path: {image_path}")
    if not isinstance(target_shape, tuple) or len(target_shape) != 2:
        raise ValueError("Target shape must be a tuple of (height, width).")

    try:
        # Load and transform the image to a tensor
        transform = transforms.ToTensor()
        with Image.open(image_path) as img:
            image = transform(img).unsqueeze(0)  # Add batch dimension

        # Validate image dimensions
        _, _, image_height, image_width = image.shape
        if image_width < 2:
            raise ValueError("Image width must be at least 2 pixels for cropping.")

        # Crop the left half of the image
        condition = image[:, :, :, :image_width // 2]

        # Interpolate to the target shape
        condition = F.interpolate(condition, size=target_shape, mode='bilinear', align_corners=False)

        # Convert the tensor back to a PIL image and save
        save_image = transforms.ToPILImage()(condition.squeeze(0))  # Remove batch dimension
        save_image.save(image_path)

        print(f"Processed and saved image: {image_path}")
    
    except UnidentifiedImageError:
        raise ValueError(f"File at {image_path} is not a valid image.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while processing the image: {e}")



image_path = "./data/sample/24.jpg"
target_shape = (256, 256)

# Run the function
crop_interpolate_and_save(image_path, target_shape)