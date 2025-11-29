"""
Utility functions for rendering and saving images
"""

import numpy as np
from PIL import Image
import torch
import cv2

def save_img_u8(img_array, save_path):
    """
    Save image array as 8-bit unsigned integer format
    
    Args:
        img_array: numpy array of shape (H, W, C) or (H, W) with values in [0, 1] range
        save_path: path to save the image
    """
    # Ensure the array is numpy
    if isinstance(img_array, torch.Tensor):
        img_array = img_array.detach().cpu().numpy()
    
    # Clip values to [0, 1] range
    img_array = np.clip(img_array, 0.0, 1.0)
    
    # Convert to 8-bit unsigned integer [0, 255]
    img_u8 = (img_array * 255).astype(np.uint8)
    
    # Handle different image formats
    if len(img_u8.shape) == 3:
        # Color image (H, W, C)
        if img_u8.shape[2] == 3:
            # RGB image
            img_pil = Image.fromarray(img_u8, 'RGB')
        elif img_u8.shape[2] == 4:
            # RGBA image
            img_pil = Image.fromarray(img_u8, 'RGBA')
        else:
            raise ValueError(f"Unsupported number of channels: {img_u8.shape[2]}")
    elif len(img_u8.shape) == 2:
        # Grayscale image (H, W)
        img_pil = Image.fromarray(img_u8, 'L')
    else:
        raise ValueError(f"Unsupported image shape: {img_u8.shape}")
    
    # Save the image
    img_pil.save(save_path)

def save_img_f32(img_array, save_path):
    """
    Save image array as 32-bit floating point format (typically for HDR images)
    
    Args:
        img_array: numpy array of shape (H, W, C) or (H, W) with floating point values
        save_path: path to save the image (should have .exr, .hdr, or .tiff extension)
    """
    # Ensure the array is numpy
    if isinstance(img_array, torch.Tensor):
        img_array = img_array.detach().cpu().numpy()
    
    # Convert to float32
    img_f32 = img_array.astype(np.float32)
    
    # Use OpenCV to save HDR formats
    if save_path.lower().endswith(('.exr', '.hdr')):
        # OpenCV expects BGR format for color images
        if len(img_f32.shape) == 3 and img_f32.shape[2] == 3:
            img_f32 = cv2.cvtColor(img_f32, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img_f32)
    elif save_path.lower().endswith('.tiff') or save_path.lower().endswith('.tif'):
        # Use PIL for TIFF
        if len(img_f32.shape) == 3:
            img_pil = Image.fromarray(img_f32, 'RGB')
        else:
            img_pil = Image.fromarray(img_f32, 'L')
        img_pil.save(save_path)
    else:
        # Fallback: convert to 8-bit and save
        print(f"Warning: {save_path} extension not recognized for float32, converting to 8-bit")
        save_img_u8(img_array, save_path)

def tensor_to_numpy(tensor):
    """
    Convert tensor to numpy array, handling device and gradient issues
    
    Args:
        tensor: PyTorch tensor
    
    Returns:
        numpy array
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor

def normalize_image(img_array, min_val=None, max_val=None):
    """
    Normalize image array to [0, 1] range
    
    Args:
        img_array: input image array
        min_val: minimum value for normalization (if None, use array min)
        max_val: maximum value for normalization (if None, use array max)
    
    Returns:
        normalized image array in [0, 1] range
    """
    if min_val is None:
        min_val = img_array.min()
    if max_val is None:
        max_val = img_array.max()
    
    if max_val - min_val == 0:
        return np.zeros_like(img_array)
    
    return (img_array - min_val) / (max_val - min_val)
