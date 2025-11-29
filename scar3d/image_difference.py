'''
Compute Signed Distance
'''

import os
import json
from PIL import Image
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import zipfile
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import yaml
import shutil
import cv2
from sklearn.decomposition import PCA 
from typing import Dict
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits

def collect_vectors_for_pca(embeddings_dict):
    """Collect all embedding vectors for PCA training"""
    all_vectors = []
    for emb in embeddings_dict.values():
        # Convert shape from [256, 64, 64] to [4096, 256]
        vectors = emb.permute(1, 2, 0).reshape(-1, 256).cpu().numpy()
        all_vectors.append(vectors)
    return np.concatenate(all_vectors, axis=0)

def compute_signed_distance_pixelwise(embeddings1, embeddings2, v_pca, device):
    """Compute signed projection distance"""
    similarities = {}
    # Convert reference vector to PyTorch tensor and send to device
    v_pca_tensor = torch.tensor(v_pca, dtype=torch.float32, device=device)
    
    for fname1, embed1 in tqdm(embeddings1.items(), desc="Computing signed distance"):
        for fname2, embed2 in embeddings2.items():
            if fname2 == fname1:
                # Compute vector difference (A-B)
                diff = embed1 - embed2  # Shape [256, 64, 64]
                
                # Adjust dimensions to [64, 64, 256]
                diff_permuted = diff.permute(1, 2, 0)
                
                # Compute projection (dot product)
                projection = torch.einsum('ijk,k->ij', diff_permuted, v_pca_tensor)
                
                similarities[fname1] = projection.cpu().numpy()
                break
    return similarities


def load_models(weights_dir):
    """
    Load all required models and extract weight files.
    
    Args:
        weights_dir (str): Directory containing weight files.
    
    Returns:
        dict: Dictionary containing all models.
    """
    models = {}
    
    # Build EfficientSAM-Ti model
    models['efficient_sam_vitt'] = build_efficient_sam_vitt()
    
    # Extract EfficientSAM-ViTS weight file
    vits_zip_path = os.path.join(weights_dir, "efficient_sam_vits.pt.zip")
    vits_extracted_path = os.path.join(weights_dir, "efficient_sam_vits.pt")
    if not os.path.exists(vits_extracted_path):
        with zipfile.ZipFile(vits_zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)
    # Build EfficientSAM-ViTS model
    models['efficient_sam_vits'] = build_efficient_sam_vits()
    
    return models

def prepare_model(model, weights_path, device):
    model.to(device)
    model.eval()
    return model

def load_images_from_directory(directory, preprocess, device):
    """
    Load all images from the specified directory and return a dictionary
    Keys are filenames, values are image tensors
    """
    images = {}
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    for filename in os.listdir(directory):
        if filename.lower().endswith(supported_formats):
            path = os.path.join(directory, filename)
            try:
                image = Image.open(path).convert('RGB')
                image_tensor = preprocess(image).to(device)
                images[filename] = image_tensor
            except Exception as e:
                print(f"Failed to load image {path}: {e}")
    height, width = image_tensor.shape[-2:]
    return images, height, width

def compute_embeddings(images, model, device):
    embeddings = {}
    with torch.no_grad():
        for filename, image_tensor in tqdm(images.items(), desc="compute embeddings"):
            # Add batch dimension
            image_tensor = image_tensor.unsqueeze(0)  # Shape [1, C, H, W]
            
            # Call model forward pass, input_points and input_labels set to None
            # Assuming model's forward method has been modified to return embeddings
            # image_embedding = model(
            #     image_tensor,
            #     None,
            #     None,
            #     img_embedding_only=True
            # )  # Expected shape [1, 256, 64, 64]

            print(f"image_tensor shape: {image_tensor.shape}")
            image_embedding = model.get_image_embeddings(image_tensor)
            
            # Ensure returning a single tensor
            if isinstance(image_embedding, tuple):
                image_embedding = image_embedding[0]
            
            # Remove batch dimension and normalize
            embedding = image_embedding.squeeze(0)  # Shape [256, 64, 64]
            # embedding = F.normalize(embedding.view(256, -1), dim=1)  # Shape [256, 4096]
            embeddings[filename] = embedding # Shape [256, 64, 64]
    return embeddings

def compute_cosine_similarity(embeddings1, embeddings2):
   
    similarities = {}
    for fname1, embed1 in tqdm(embeddings1.items(), desc="Computing similarity"):
        for fname2, embed2 in embeddings2.items():
            # Compute cosine similarity
            # embed1 and embed2 shapes are both [256, 4096]
            if fname2 == fname1:
                cos_sim = F.cosine_similarity(embed1, embed2, dim=1).mean().item()
                similarities[fname1] = cos_sim
                break
    return similarities


def compute_cosine_similarity_pixelwise(embeddings1, embeddings2):
    similarities = {}
    
    for fname1, embed1 in tqdm(embeddings1.items(), desc="cos sim"):
        for fname2, embed2 in embeddings2.items():
            # Only compute similarity between files with the same name
            if fname2 == fname1:
                # Assume embed1 and embed2 are 256 x 64 x 64 shapes
                cos_sim_map = np.zeros(embed1.shape[1:])  # Initialize a 64 x 64 matrix to store similarity results
                
                for i in range(embed1.shape[1]):  # Iterate through each pixel of the 64x64 feature map
                    for j in range(embed1.shape[2]):
                        # Extract vector for each pixel, compute cosine similarity
                        pixel_embed1 = embed1[:, i, j].reshape(1, -1)  # (1, 256)
                        pixel_embed2 = embed2[:, i, j].reshape(1, -1)  # (1, 256)

                        if isinstance(pixel_embed1, torch.Tensor):
                            pixel_embed1 = pixel_embed1.cpu().numpy()
                        if isinstance(pixel_embed2, torch.Tensor):
                            pixel_embed2 = pixel_embed2.cpu().numpy()

                        cos_sim = cosine_similarity(pixel_embed1, pixel_embed2)[0][0]
                        cos_sim_map[i, j] = cos_sim

                # cos_sim_map = cos_sim_map.astype(np.float64)
                # cos_sim_map = torch.tensor(cos_sim_map, dtype=torch.float64)
                # cos_sim_map = cos_sim_map.tolist()
                
                similarities[fname1] = cos_sim_map

    return similarities

# Signed distance
def plot_signed_distance_heatmaps(similarities, save_path):
    os.makedirs(save_path, exist_ok=True)
    for fname, cos_sim_map in similarities.items():
        # Create heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(cos_sim_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Signed Distance: {fname}')
        fname = fname.split('.')[0]
        # Save heatmap
        plt.savefig(f'{save_path}/{fname}_signed_dist.png')
        plt.close()
        print(f"Saved heatmap: {save_path}/{fname}_signed_dist.png")
    
    print("Heatmaps saved successfully")


def plot_cosine_similarity_heatmaps(similarities, save_path):
    os.makedirs(save_path, exist_ok=True)
    for fname, cos_sim_map in similarities.items():
        # 创建热图
        plt.figure(figsize=(8, 6))
        plt.imshow(cos_sim_map, cmap='hot', interpolation='nearest')
        plt.colorbar()
        plt.title(f'Cosine Similarity Heatmap: {fname}')
        fname = fname.split('.')[0]
        # 保存热图
        plt.savefig(f'{save_path}/{fname}_cos_sim_heatmap.png')
        plt.close()
        print(f"保存热图: {save_path}/{fname}_cos_sim_heatmap.png")
    
    print("热图保存完毕")
    

def resize_similarities(similarities, target_height, target_width):
    resized_similarities = {}
    
    for fname, cos_sim_map in similarities.items():
        # Convert NumPy array to PyTorch tensor
        cos_sim_map_tensor = torch.tensor(cos_sim_map, dtype=torch.float32)
        
        # Use F.interpolate for upsampling (interpolation)
        resized_map = F.interpolate(cos_sim_map_tensor.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions
                                    size=(target_height, target_width),  # Target size
                                    mode='bilinear',  # Interpolation mode
                                    align_corners=False)  # Align corners (prevent offset during interpolation)
        
        # Remove extra dimensions, restore to [H, W] size
        resized_map = resized_map.squeeze(0).squeeze(0).cpu().numpy()  # Convert back to NumPy array
        
        # Store resized similarity map
        resized_similarities[fname] = resized_map
    
    return resized_similarities


def get_difference_map(similarities, save_path, threshold, inv=False):
    """
    Generate difference map based on cosine similarity.
    By default, regions with similarity below threshold are considered differences.
    If inv=True, regions with similarity above threshold are considered differences.

    Args:
        similarities (dict): Cosine similarity dictionary, keys are filenames, values are similarity matrices (2D arrays).
        save_path (str): Directory to save difference maps.
        threshold (float): Similarity threshold for difference judgment, below this value is considered difference.
    """
    os.makedirs(save_path, exist_ok=True)
    
    for fname, cos_sim_map in similarities.items():
        # Create a binarized difference map
        if inv:
            diff_map = cos_sim_map > threshold
        else:
            diff_map = cos_sim_map < threshold  # If similarity is below threshold, it is True (difference)
        
        # Convert difference map from boolean to 0 or 1 numeric type
        diff_map = diff_map.astype(np.uint8)  # Convert to 0 and 1 binary map
        
        # Save difference map as image
        diff_map_image = Image.fromarray(diff_map * 255)  # Map 0 and 1 to 0 and 255 for saving as image
        diff_map_image = diff_map_image.convert('L')  # Convert to grayscale
        
        # Save image
        diff_map_image.save(f"{save_path}/{fname}")
        print(f"Saved difference map: {save_path}/{fname}")


def plot_similarity_distributions(similarities: Dict[str, np.ndarray],
                                 save_path: str,
                                 bins: int = 100,
                                 range_min: float = -1.0,
                                 range_max: float = 1.0,
                                 individual_alpha: float = 0.3):
    """
    Visualize pixel value distribution of all similarity maps and their average distribution
    
    Args:
        similarities: Dictionary, keys are filenames, values are similarity map numpy arrays
        save_path: Path to save result image
        bins: Number of histogram bins (default 100)
        range_min: Minimum value range (default 0.0)
        range_max: Maximum value range (default 1.0)
        individual_alpha: Transparency of individual distribution curves (default 0.3)
    """
    # Initialize storage structure
    all_distributions = []
    accumulated = None
    bin_edges = None
    total_images = 0
    
    for fname, sim_map in similarities.items():
        # Compute histogram for current similarity map
        hist, edges = np.histogram(sim_map, 
                                 bins=bins, 
                                 range=(range_min, range_max))
        
        # Convert to frequency distribution
        hist_freq = hist / sim_map.size
        
        # Store individual distribution
        all_distributions.append(hist_freq)
        
        # Initialize accumulation array
        if accumulated is None:
            accumulated = np.zeros_like(hist_freq)
            bin_edges = edges  # Save unified bin edges
        
        # Accumulate frequency distribution
        accumulated += hist_freq
        total_images += 1

    # Compute average distribution
    avg_distribution = accumulated / total_images
    
    # Create visualization chart
    plt.figure(figsize=(14, 8))
    
    # Plot all individual distributions
    for dist in all_distributions:
        plt.step(bin_edges[:-1], dist, 
                where='post', 
                alpha=individual_alpha,
                color='gray')
    
    # Plot average distribution
    plt.step(bin_edges[:-1], avg_distribution, 
            where='post', 
            linewidth=3,
            color='red',
            label=f'Average of {total_images} images')
    
    # Compute dynamic thresholds
    left, right = find_peak_thresholds(avg_distribution, range_min=range_min, range_max=range_max, bin=bins)
    print(f"Dynamic thresholds: {left}, {right}")

    # Plot dynamic threshold lines
    if left is not None:
        plt.axvline(x=left, color='blue', linestyle='--', label='Left Threshold')
    if right is not None:
        plt.axvline(x=right, color='green', linestyle='--', label='Right Threshold')
    
    # Set chart properties
    plt.xlabel("Cosine Similarity Value", fontsize=12)
    plt.ylabel("Normalized Frequency", fontsize=12)
    plt.title("Pixel Value Distribution of Similarity Maps", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)
    
    # Set axis ranges
    plt.xlim(range_min, range_max)
    plt.ylim(0, avg_distribution.max()*1.2)  # Automatically adjust Y-axis range
    
    # Save result
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()

    return avg_distribution, left, right


# dynamic thresholding
def find_peak_thresholds(signal, range_min, range_max, bin, sigma=1):
    """
    Input: 1D signal array, smoothing parameter sigma (Gaussian filter standard deviation)
    Output: Left and right threshold indices (left_threshold, right_threshold), None if no peak
    """
    # Step 1: Gaussian smoothing on signal to reduce noise
    smooth = gaussian_filter1d(signal, sigma=sigma)
    
    # Step 2: Find global main peak (maximum value point) of smoothed signal
    main_idx = np.argmax(smooth)
    
    # Step 3: Find secondary peak (local maximum) on the left of main peak
    left_peak_idx = None
    if main_idx > 0:
        left_slice = smooth[:main_idx]  # Sub-signal left of main peak
        left_peaks, _ = find_peaks(left_slice)
        if left_peaks.size > 0:
            # Select peak with largest amplitude on the left
            left_peak_rel = left_peaks[np.argmax(left_slice[left_peaks])]
            left_peak_idx = left_peak_rel
    
    # Find secondary peak (local maximum) on the right of main peak
    right_peak_idx = None
    if main_idx < len(smooth) - 1:
        right_slice = smooth[main_idx+1:]  # Sub-signal right of main peak
        right_peaks, _ = find_peaks(right_slice)
        if right_peaks.size > 0:
            # Select peak with largest amplitude on the right (remember to add offset)
            right_peak_rel = right_peaks[np.argmax(right_slice[right_peaks])]
            right_peak_idx = main_idx + 1 + right_peak_rel
    
    # Step 4: Find valley position between main peak and secondary peak as threshold
    left_threshold = None
    if left_peak_idx is not None:
        # Find minimum value position in interval from left secondary peak to main peak
        segment = smooth[left_peak_idx: main_idx+1]
        min_rel = np.argmin(segment)
        left_threshold = left_peak_idx + min_rel
    
    right_threshold = None
    if right_peak_idx is not None:
        # Find minimum value position in interval from main peak to right secondary peak
        segment = smooth[main_idx: right_peak_idx+1]
        min_rel = np.argmin(segment)
        right_threshold = main_idx + min_rel

    step = (range_max - range_min) / bin
    if left_threshold is not None:
        left_threshold = (left_threshold - bin / 2) * (range_max - range_min) / bin + step / 2
    if right_threshold is not None:
        right_threshold = (right_threshold - bin / 2) * (range_max - range_min) / bin + step / 2
    
    return left_threshold, right_threshold


def detect_overlap(dir_prompt, config, model, difmap_prompt, difmap_add, difmap_temp):
    '''
    0. For right (bright, removed)
    1. For each dif-right, find points, prompt efsam (pre), get mask
    2. mask(defmap_temp) minus dif-right(difmap_prompt) is the supplement for dif-left(difmap_add)
    3. Read dif-left, add to mask, save
    4. Also, can complete the incomplete mask of right: add mask to dif-right(difmap_prompt) and save to difmap_prompt (but this is the very initial method)
    5. left similarly
    '''
    from scar3d.segmentation import get_prompted_mask_difference
    bin_temp_dir = os.path.join(difmap_temp, "temp")

    # The first one is the prompt mask dir
    masks = get_prompted_mask_difference(difmap_prompt, config, dir_prompt, difmap_temp, bin_temp_dir)

    # Delete temp directory
    shutil.rmtree(bin_temp_dir)

    names_prompt = os.listdir(difmap_prompt)
    names_add = os.listdir(difmap_add)

    for name in tqdm(names_prompt, desc="detect overlap"):

        if name not in masks.keys() or name not in names_add:
            print(f"Warning: {name} not found in masks or difmap_add, skipping.")
            continue
        
        dif_prompt_path = os.path.join(difmap_prompt, name)
        dif_prompt = Image.open(dif_prompt_path).convert('L')  # Convert to grayscale
        dif_prompt_array = np.array(dif_prompt)

        dif_add_path = os.path.join(difmap_add, name)
        dif_add = Image.open(dif_add_path).convert('L')  # Convert to grayscale
        dif_add_array = np.array(dif_add)

        mask = masks[name]  # uint8 format
        if len(mask.shape) == 3:
            # If mask is color, convert to grayscale
            mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        else:
            mask_gray = mask

        # Calculate dif_prompt white area minus mask remaining white area, and merge this part with dif_add, then save to corresponding dif_add path
        
        
        # Binarize mask, ensure only 0 and 255
        _, mask_binary = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        # Calculate dif_prompt white area minus mask remaining white area
        # 1. dif_prompt white area (assume white area is pixels > 127)
        dif_prompt_white = (dif_prompt_array > 1).astype(np.uint8) * 255
        
        # 2. mask white area
        mask_white = (mask_binary > 1).astype(np.uint8) * 255
        
        # 3. Remaining part of dif_prompt white area minus mask white area
        # Use bitwise operation: dif_prompt_white AND NOT mask_white
        remaining_white = cv2.bitwise_and(mask_white, cv2.bitwise_not(dif_prompt_white))
        
        # 4. Merge remaining white area with dif_add
        # Use bitwise OR operation to merge
        merged_result = cv2.bitwise_or(dif_add_array, remaining_white)
        
        # 5. Save to dif_add path
        merged_image = Image.fromarray(merged_result)
        merged_image.save(dif_add_path)


        # Add mask to dif-right(difmap_prompt) and save to difmap_prompt
        # Here merge mask white area with dif_prompt white area
        enhanced_dif_prompt = cv2.bitwise_or(dif_prompt_array, mask_white)
        
        # Save enhanced dif_prompt
        enhanced_image_prompt = Image.fromarray(enhanced_dif_prompt)
        enhanced_image_prompt.save(dif_prompt_path)
        
        
        print(f"Processed and saved: {dif_add_path}")


def signed_difference(dir_pre, dir_post, config, model, heatmap, difmap):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Using device: {device}")

    # Load configuration file
    # try:
    with open(config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        left_threshold = config['mask']['difference_map_threshold_left']
        right_threshold = config['mask']['difference_map_threshold_right']
        if config['mask']["use_dynamic_thre"] == False:
            use_dynamic_thre = False
        else:
            use_dynamic_thre = True
        print(f"load config file: {config}")
    # except:
    # threshold = 0.27
    # use_dynamic_thre = True
    # print(f"load config file failed, use dynamic threshold")
    
    # Define image preprocessing
    preprocess = transforms.Compose([
        # transforms.Resize((256, 256)),  
        transforms.ToTensor(),
    ])
    
    # Load images from directory 1
    print("Loading images from directory 1...")
    images1, height, width = load_images_from_directory(dir_pre, preprocess, device)
    print(f"Loaded {len(images1)} images from {dir_pre}")
    print(f"Image size: {height} x {width}")
    
    # Load images from directory 2
    print("Loading images from directory 2...")
    images2, _, _ = load_images_from_directory(dir_post, preprocess, device)
    print(f"Loaded {len(images2)} images from {dir_post}")

    
    # Compute embeddings for directory 1
    print("Computing embeddings for directory 1...")
    embeddings1 = compute_embeddings(images1, model, device)
    
    # Compute embeddings for directory 2
    print("Computing embeddings for directory 2...")
    embeddings2 = compute_embeddings(images2, model, device)


    # ========== New PCA processing part ==========
    print("Preparing PCA training data...")
    # Merge embeddings from both directories
    all_embeddings = {**embeddings1, **embeddings2}
    pca_vectors = collect_vectors_for_pca(all_embeddings)
    
    print("Training PCA model...")
    pca = PCA(n_components=1)
    pca.fit(pca_vectors)
    v_pca = pca.components_[0]  # Get first principal component direction
    print(f"Principal component direction shape: {v_pca.shape}, Norm: {np.linalg.norm(v_pca):.4f}")

    # ========== Modified similarity computation part ==========
    print("Computing signed projection distance...")
    similarities = compute_signed_distance_pixelwise(
        embeddings1, embeddings2, v_pca, device
    )


    # Subsequent processing remains unchanged (resize, generate heatmap and difference map)
    similarities = resize_similarities(similarities, height, width)
    plot_signed_distance_heatmaps(similarities, heatmap)

    range_min = -1.0
    range_max = 1.0
    bins = 50


    difmap_left_base = os.path.join(difmap, "left")
    difmap_right_base = os.path.join(difmap, "right")
    os.makedirs(difmap_left_base, exist_ok=True)
    os.makedirs(difmap_right_base, exist_ok=True)



    # Find a left and right threshold for each image
    if use_dynamic_thre:
        print("Using dynamic thresholds for each image.")
        for fname, sim_map in tqdm(similarities.items(), desc="Generating difference maps (dynamic per image)"):
            # Calculate distribution for the current image's similarity map
            hist, _ = np.histogram(sim_map, bins=bins, range=(range_min, range_max))
            if sim_map.size == 0: # Avoid division by zero for empty maps
                hist_freq = np.zeros_like(hist, dtype=float)
            else:
                hist_freq = hist / sim_map.size
            
            # Find thresholds for the current image
            # Ensure find_peak_thresholds can handle all-zero hist_freq if that's possible
            img_left_threshold, img_right_threshold = find_peak_thresholds(
                hist_freq, 
                range_min=range_min, 
                range_max=range_max, 
                bin=bins  
            )
            
            if img_left_threshold is not None:
                print(f"Image {fname}: Using dynamic left threshold: {img_left_threshold:.4f}")
                get_difference_map({fname: sim_map}, difmap_left_base, img_left_threshold, inv=False)
            else:
                print(f"Image {fname}: No dynamic left threshold found or applicable. ")
                get_difference_map({fname: sim_map}, difmap_left_base, left_threshold, inv=False) 
            
            if img_right_threshold is not None:
                print(f"Image {fname}: Using dynamic right threshold: {img_right_threshold:.4f}")
                get_difference_map({fname: sim_map}, difmap_right_base, img_right_threshold, inv=True)
            else:
                print(f"Image {fname}: No dynamic right threshold found or applicable.")
                get_difference_map({fname: sim_map}, difmap_right_base, right_threshold, inv=True)


    else:
        print(f"Left={left_threshold}, Right={right_threshold}")
        get_difference_map(similarities, difmap_left_base, left_threshold, inv=False)
        get_difference_map(similarities, difmap_right_base, right_threshold, inv=True)



    if config['mask']['detect_overlap']:
        difmap_temp_left = os.path.join(difmap, "temp_left")
        difmap_temp_right = os.path.join(difmap, "temp_right")
        os.makedirs(difmap_temp_left, exist_ok=True)
        os.makedirs(difmap_temp_right, exist_ok=True)
        detect_overlap(dir_pre, config, model, difmap_right_base, difmap_left_base, difmap_temp_left) # pre temp叫left是加到left上面的意思
        detect_overlap(dir_post, config, model, difmap_left_base, difmap_right_base, difmap_temp_right) # post
    
    print("Done")



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compute cosine similarity between images in two directories and save as JSON")
    parser.add_argument('--dir1', type=str, help='First directory path')
    parser.add_argument('--dir2', type=str, help='Second directory path')
    parser.add_argument('--heatmap', type=str, help='Path to save heatmaps')
    # parser.add_argument('--output', type=str, help='Output JSON file path')
    parser.add_argument('--difmap', type=str, help='Output difference map file path')
    # parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--weights_dir', type=str, default='submodules/EfficientSAM/weights', help='Directory containing model weights')
    parser.add_argument('--model', type=str, default='efficient_sam_vitt', help='Name of the model to use')
    parser.add_argument('--sim_stat', type=str, default=None, help='Directory to save similarity statistics plot')
    
    args = parser.parse_args()
    
    # main(args)
    signed_difference(args.dir1, args.dir2, args.config, args.model, args.weights_dir, args.heatmap, args.difmap, args.sim_stat)
