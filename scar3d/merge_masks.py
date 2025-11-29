def merge_mask(post_mask_path, pre_mask_path, output_path):
    """
    Merge post and pre change masks into a single mask.
    
    Args:
        post_mask_path (str): Path to the post change mask.
        pre_mask_path (str): Path to the pre change mask.
        output_path (str): Path to save the merged mask.
    """
    import numpy as np
    import cv2

    # Load the post and pre masks
    post_mask = cv2.imread(post_mask_path, cv2.IMREAD_GRAYSCALE)
    pre_mask = cv2.imread(pre_mask_path, cv2.IMREAD_GRAYSCALE)

    if post_mask is None or pre_mask is None:
        raise ValueError("One of the masks could not be loaded. Check the paths.")

    # Ensure both masks are of the same size
    if post_mask.shape != pre_mask.shape:
        raise ValueError("Post and pre masks must have the same dimensions.")

    # Merge masks: combine them using bitwise OR operation
    merged_mask = cv2.bitwise_or(post_mask, pre_mask)

    # Apply a threshold to ensure binary masks
    _, merged_mask = cv2.threshold(merged_mask, 1, 255, cv2.THRESH_BINARY)

    # Save the merged mask
    cv2.imwrite(output_path, merged_mask)


def merge_masks(post_mask_dir, pre_mask_dir, output_dir):
    """
    Merge all post and pre change masks in the specified directories.
    
    Args:
        post_mask_dir (str): Directory containing post change masks.
        pre_mask_dir (str): Directory containing pre change masks.
        output_dir (str): Directory to save the merged masks.
    """
    import os

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    post_masks = [f for f in os.listdir(post_mask_dir) if f.endswith('.png')]
    pre_masks = [f for f in os.listdir(pre_mask_dir) if f.endswith('.png')]

    for post_mask in post_masks:
        pre_mask = post_mask

        if pre_mask in pre_masks:
            post_mask_path = os.path.join(post_mask_dir, post_mask)
            pre_mask_path = os.path.join(pre_mask_dir, pre_mask)
            output_path = os.path.join(output_dir, post_mask)

            merge_mask(post_mask_path, pre_mask_path, output_path)
            print(f"Merged {post_mask} and {pre_mask} into {output_path}")
        else:
            print(f"Pre mask for {post_mask} not found.")