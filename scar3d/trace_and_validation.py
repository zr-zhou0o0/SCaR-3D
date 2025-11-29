
import os
import sys
import argparse
import zipfile
import yaml
import shutil
from PIL import Image
from tqdm import tqdm
import cv2

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

import torch
from scene import Scene
from scene.gaussian_model import GaussianModel # true GaussianEditor version
from gaussian_renderer import render_mean2D 

from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args


def load_binary_mask(mask_path, device, width=1600, height=1200):
    mask_image = Image.open(mask_path).convert('L')  # Convert to grayscale
    mask_image = mask_image.resize((width, height))
    # print("mask_image resize: ", mask_image.size)
    mask_np = np.array(mask_image)
    id_mask = (mask_np > 120).astype(np.int32)
    id_mask_tensor = torch.from_numpy(id_mask).to(torch.int).to(device)

    # mask_image_output_path = os.path.splitext(mask_path)[0] + "_output.png"
    # mask_image_output = Image.fromarray((id_mask_tensor.cpu().numpy() * 255).astype(np.uint8))
    # mask_image_output.save(mask_image_output_path)
    # print(f"Mask image saved to {mask_image_output_path}")

    return id_mask_tensor


def print_pt(pt_path):
    data = torch.load(pt_path, weights_only=True)
    # Randomly select 10 rows to print
    if data.ndim == 2 and data.shape[0] > 10:
        num_cols = data.shape[0]
        # Randomly select 10 column indices
        # Ensure selected_indices are sorted for potentially more readable output, though not strictly necessary
        selected_indices = torch.randperm(num_cols)[:10].sort().values
        print(f"Randomly selected 10 columns (indices: {selected_indices.tolist()}):")
        print(data[selected_indices])
    elif data.ndim == 1:
        print("Tensor is 1D, printing all elements:")
        print(data)
    else: # ndim == 2 and data.shape[1] <= 10, or other ndim cases
        print("Tensor has 10 or fewer columns, or is not 2D. Printing all data:")

    # print(data)

def statistic_pt(pt_path):
    data = torch.load(pt_path, weights_only=True)
    print("shape: ", data.shape)
    print("dtype: ", data.dtype)
    # print("device: ", data.device)
    print("mean: ", data.mean())
    # print("std: ", data.std())
    print("min: ", data.min()) 
    print("max: ", data.max())
    # print("sum: ", data.sum())

    mean_first_col = data[:, 0].mean().item()
    print("mean of first column: ", mean_first_col)

    mean_second_col = data[:, 1].mean().item()
    print("mean of second column: ", mean_second_col)

    non_zero_rows = (data != 0).any(dim=1).sum().item()
    print(f"non-zero rows / total: {non_zero_rows}/{data.shape[0]}")


def statistic_xy(points_xy_image):
    # points_xy_image: [3484164, 2]
    print("points_xy_image shape:", len(points_xy_image))
    print("points_xy_image dtype:", points_xy_image.dtype)
    print("points_xy_image mean:", points_xy_image.mean(dim=0))
    print("points_xy_image min:", points_xy_image.min(dim=0))
    print("points_xy_image max:", points_xy_image.max(dim=0))
    # print("points_xy_image std:", points_xy_image.std(dim=0))
    # print("points_xy_image sum:", points_xy_image.sum(dim=0))
    
    # randomly sample 10 points
    sample_indices = torch.randint(0, points_xy_image.shape[0], (10,))
    print("sampled points_xy_image:", points_xy_image[sample_indices])


def prune(gaussians, scene, config, dataset : ModelParams, iteration : int, pipeline : PipelineParams, color_raw : torch.tensor, masks_dir : str, separate_sh: bool):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    error_views = config["mask"]["error_views"]

    with torch.no_grad():

        # gaussians = GaussianModel(dataset.sh_degree)
        # scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        import time
        start_time = time.time()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda") # Move to GPU 

        model_path = dataset.model_path
        views = scene.getTrainCameras()
        iteration = scene.loaded_iter
        train_test_exp = dataset.train_test_exp

        # save_color_path = os.path.join(model_path, save_color_name+".pt")
      
        # color = torch.load(color_path, weights_only=True).to(device) 
        color = color_raw.repeat(1, 3)
        # traced_gaussians is 1.0 for non-black, 0.0 for black
        traced_gaussians_active_mask = (color.abs().sum(dim=1) > 1e-6).to(dtype=torch.bool) # Boolean mask on GPU
        # Initialize pruned_gaussians mask on GPU (1.0 means prune, 0.0 means keep)
        pruned_gaussians_accumulated_mask = torch.zeros_like(traced_gaussians_active_mask, dtype=torch.bool, device=device)
        # pruned_gaussians_accumulated_mask = torch.zeros_like(traced_gaussians_active_mask, dtype=torch.int16, device=device)
       
        

        for idx, view in enumerate(tqdm(views, desc="Pruning")):

            print("view:", view.image_name)
            uid = int(view.image_name.split(".")[0].split("_")[-1]) # e.g. whole_23.png
            mask_path = os.path.join(masks_dir, f"{uid:05d}.png")

            mask_name = f"{uid:05d}.png"
            if mask_name in error_views:
                print(f"Skipping view {view.image_name} due to error view.")
                continue

            try:
                masknp = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)
                print("mask shape:", masknp.shape)
            except:
                print(f"Error reading mask image: {mask_path}")
                continue

            # binary mask
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            # True if larger than 128, False otherwise
            masknp = masknp > 128
            # bool type
            masknp = masknp.astype(np.bool_)

            width = masknp.shape[1]
            height = masknp.shape[0]

            image_width = view.image_width
            image_height = view.image_height
            assert width == image_width
            assert height == image_height

            masked_num = np.sum(masknp)
            print("width, height:", width, height)
            print("masked num:", masked_num)
        

            # export CUDA_LAUNCH_BLOCKING=1
            # export TORCH_USE_CUDA_DSA=1

            # render_all = render_original(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh) # to test

            render_all = render_mean2D(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)

            # rendering = render_all["render"]
            # depth = render_all["depth"]
            mean2D = render_all["means2D"]
            # print("mean2D: ", mean2D[0])
            # statistic_xy(mean2D)

            mask_gpu = torch.from_numpy(masknp).to(device) # Shape (H, W), boolean, on GPU

            x_coords = mean2D[:, 0]
            y_coords = mean2D[:, 1]

            # Identify Gaussians out of image bounds
            out_of_bounds = (x_coords < 0) | (x_coords >= width) | \
                            (y_coords < 0) | (y_coords >= height)

            # Prune if initially active AND out of bounds
            prune_due_to_oob_this_view = traced_gaussians_active_mask & out_of_bounds
            
            # Consider Gaussians that are initially active AND in bounds for mask checking
            active_and_in_bounds = traced_gaussians_active_mask & ~out_of_bounds
            
            # Initialize mask for pruning due to being in a masked-out region (for this view)
            prune_due_to_mask_this_view = torch.zeros_like(traced_gaussians_active_mask, dtype=torch.bool, device=device)
            # prune_due_to_mask_this_view = torch.zeros_like(traced_gaussians_active_mask, dtype=torch.int16, device=device)

            indices_active_in_bounds = torch.where(active_and_in_bounds)[0] # Because we only prune Gaussians that were traced before

            if indices_active_in_bounds.numel() > 0:
                # Get integer coordinates for indexing the mask_gpu
                x_coords_idx = x_coords[indices_active_in_bounds].long()
                y_coords_idx = y_coords[indices_active_in_bounds].long()

                # Clamp coordinates to be within mask dimensions (0 to H-1, 0 to W-1)
                x_coords_idx.clamp_(0, width - 1)
                y_coords_idx.clamp_(0, height - 1)
                
                # Check mask values: mask_gpu is True for foreground (keep), False for background (prune)
                # We want to prune if mask_gpu value is False (i.e., background)
                is_in_background_region = (mask_gpu[y_coords_idx, x_coords_idx] == False)
                
                # Identify the original indices of Gaussians that are active, in_bounds, AND in a background region
                gaussians_to_prune_in_background = indices_active_in_bounds[is_in_background_region]
                
                if gaussians_to_prune_in_background.numel() > 0:
                    prune_due_to_mask_this_view[gaussians_to_prune_in_background] = True
            
            # Combine pruning reasons for this view: OOB or in background region of the mask
            # pruned_in_current_view_mask = prune_due_to_oob_this_view | prune_due_to_mask_this_view
            # The correct handling for OOB is that in this view, nothing can be deleted, otherwise if it doesn't appear in the view, it will be deleted entirely
            pruned_in_current_view_mask = prune_due_to_mask_this_view
            
            # Accumulate to the global pruned_gaussians_accumulated_mask
            # If a Gaussian is marked for pruning in ANY view, it stays pruned.
            
            # hard prune
            pruned_gaussians_accumulated_mask |= pruned_in_current_view_mask

            # soft prune
            # pruned_gaussians_accumulated_mask += pruned_in_current_view_mask

        # thresholded soft prune
        # thre = len(views) / 2
        # print(f"Threshold for pruning: {thre}")
        # print(f"Pruned Gaussians Accumulated Mask max: {pruned_gaussians_accumulated_mask.max().item()}")
        # pruned_gaussians_accumulated_mask = pruned_gaussians_accumulated_mask > thre

        # Actually, we can also have a strategy to directly subtract the negative color of prune from color, and then turn all negative values into 0, which is softer

        # pruned number / total valid gaussian number:
        pruned_gaussians_count = pruned_gaussians_accumulated_mask.sum().item()
        total_valid_gaussians_count = traced_gaussians_active_mask.sum().item()
        if total_valid_gaussians_count == 0:
            print("No valid Gaussians found to prune.")
            return color_raw, 0.0

        print(f"All gaussians: {traced_gaussians_active_mask.shape[0]}")
        print(f"Pruned Gaussians: {pruned_gaussians_count} / {total_valid_gaussians_count} = {pruned_gaussians_count / total_valid_gaussians_count:.2%}")
        print(f"Remained Gaussians: {total_valid_gaussians_count - pruned_gaussians_count} / {total_valid_gaussians_count} = {(total_valid_gaussians_count - pruned_gaussians_count) / total_valid_gaussians_count:.2%}")

        remain_rate = (total_valid_gaussians_count - pruned_gaussians_count) / total_valid_gaussians_count

        # average value
        if traced_gaussians_active_mask.any():
            active_colors = color[traced_gaussians_active_mask]
            print("color shape:", color.shape)
            print("active colors shape:", active_colors.shape)
            average_brightness = active_colors.mean()
            print(f"Average brightness of non-zero colors before pruning: {average_brightness.item():.4f}")
        else:
            print("No non-zero colors found before pruning.")


        # Create the new color tensor
        color_new = color.clone() # color is already on GPU
        # Set color to red for all Gaussians marked for pruning
        # color_new[pruned_gaussians_accumulated_mask] = torch.tensor([1, 0, 0], dtype=torch.float32, device=device) # red
        print("color new shape:", color_new.shape)
        print("pruned_gaussians_accumulated_mask shape:", pruned_gaussians_accumulated_mask.shape)
        color_new[pruned_gaussians_accumulated_mask] = torch.tensor([0, 0, 0], dtype=torch.float32, device=device) # black

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time for pruning: {elapsed_time:.2f} seconds")

        # save color new
        # torch.save(color_new.cpu(), save_color_path) # Move to CPU before saving if needed
        # print("save color new to:", save_color_path)
        return color_new, remain_rate

        # One is that prune points are best implemented on Gaussian (GPU)
        # The other is what about its xy image coordinate system?
        # In the test example 1007 * 755 = 1762
        # minx = -453.65   max x = 1346.3055
        # miny = -437.28   max y = 1417.4528
        # Is it because it didn't add image box restrictions?
        # Since it is called xy, the first idx should be x and the second should be y


            # logic：
            # for gaussian_idx in range(traced_gaussians.shape[0]):
            #     if mean2D[0] < 0 or mean2D[0] >= mask.shape[1] or mean2D[1] < 0 or mean2D[1] >= mask.shape[0]:
            #         pruned_gaussians[gaussian_idx] = 1
            #     else:
            #         # check if the center is inside the mask
            #         if mask[mean2D[1], mean2D[0]] == 0:
            #             pruned_gaussians[gaussian_idx] = 1
            # color_new = color.clone()
            # color_new[pruned_gaussians == 1] = torch.tensor([1, 0, 0], dtype=torch.float32, device="cuda") # red




def save_bicolor_gaussians(gaussians, selected_mask, cumulative_weights, output_path):
    # print("gaussian feature_dc shape:", gaussians._features_dc.shape)
    # print("gaussian feature_rest shape:", gaussians._features_rest.shape)

    device = gaussians._features_dc.device  
    
    selected_mask = selected_mask.to(device) 
    
    binary_colors = torch.zeros_like(gaussians._features_dc, device=device) 
    # binary_colors_rest = torch.zeros_like(gaussians._features_rest[:, :3, :], device=device)  
    binary_colors_rest = torch.zeros_like(gaussians._features_rest, device=device)  
    
    binary_colors[selected_mask] = 1.0 
    binary_colors_rest[selected_mask] = 1.0  

    print("binary_colors_shape:", binary_colors_rest.shape)
    
    with torch.no_grad():
        gaussians._features_dc = binary_colors
        gaussians._features_rest = binary_colors_rest
        # gaussians._features_rest = None # This doesn't work, save and load ply will fail, can only change render?

    gaussians.save_ply(output_path)
    print(f"Bicolor Gaussians saved to {output_path}")

    # NOTE Note that this one without weight looks very wrong
    # torch.save(binary_colors, output_path.replace(".ply", "_color.pt"))

    weight_colors = cumulative_weights.repeat(1, 3)
    torch.save(weight_colors, output_path.replace(".ply", "_weight_color.pt"))




def trace(config, dataset, opt, pipe, iteration, mask_folder, output_ply, threshold, width=1600, height=1200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("dataset.sh degree", dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree, 
                              anchor_weight_init_g0=1.0,
                              anchor_weight_init=0.1,
                              anchor_weight_multiplier=2,
                              )  

    scene = Scene(dataset, gaussians, load_iteration=iteration) 
    gaussians.training_setup(opt)
    train_cameras = scene.getTrainCameras()

    import time
    start_time = time.time()
    print("start_time: ", start_time)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    num_gaussians = gaussians.get_scaling.shape[0]
    cumulative_weights = torch.zeros_like(gaussians._opacity, dtype=torch.float32, device=device) # cumulative weights type: tensor([n, 1])

    print("cumulative_weight shape: ", cumulative_weights.shape)

    mask_files = sorted(os.listdir(mask_folder))
    num_masks = len(mask_files)
    print(f"找到 {num_masks} 个 mask 文件.")

    error_views = config["mask"]["error_views"]
    if_prune = config["mask"]["prune"]

    num_view = 0
    num_views = np.zeros_like(cumulative_weights.to('cpu'), dtype=np.int32) # [n, 1]
    num_views = torch.from_numpy(num_views).to(device)  # 将 NumPy 数组转换为 PyTorch 张量
    for mask_file in tqdm(mask_files, desc="loading masks..."):
        # print("mask_file:", mask_file)
        if mask_file in error_views:
            print("ignore error view")
            continue
        num_view += 1
        mask_path = os.path.join(mask_folder, mask_file)
        # print(f"\n mask: {mask_path}")
        id_mask = load_binary_mask(mask_path, device, width=width, height=height)
        id_mask = id_mask.float()
        id_mask = id_mask.unsqueeze(0)
        mask_name = os.path.basename(mask_path)
        # img_name_str = mask_name.split('.')[0].split('_')[0]
        img_name_str = mask_name.split('.')[0]

        try:
            img_uid = int(img_name_str)
        except:
            img_uid = int(img_name_str[:5])

        # XXX 3dgscd dataset uses arbitrary bits for image names 
        # img_name = 'whole_'+ str(img_uid) + ".png"
        # XXX Ours blender dataset use 5 bits images namedataset
        img_name = f"whole_{img_uid:05d}.png"  # e.g. whole_00023.png 
        # XXX PASLCD dataset
        # img_name = f"Inst_1_test_IMG_{img_uid:04d}.jpg" 

        viewpoint_camera = None
        for camera in train_cameras:
            if camera.image_name == img_name:
                # print("find camera: ", camera.image_name)
                viewpoint_camera = camera
                break

        if viewpoint_camera is None:
            print(f"未找到与图像{img_name}对应的相机。跳过此 mask。")
            return 
        
        # print("viewpoint cam: ",viewpoint_camera.image_name)

        weights = torch.zeros_like(gaussians._opacity)
        weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)

        gaussians.apply_weights(viewpoint_camera, weights, weights_cnt, id_mask)

        train_test_exp = dataset.train_test_exp

        render_all = render_mean2D(viewpoint_camera, gaussians, pipe, background, use_trained_exp=train_test_exp, separate_sh=False)
        mean2D = render_all["means2D"]
        x_coords = mean2D[:, 0]
        y_coords = mean2D[:, 1]
        out_of_bounds = (x_coords < 0) | (x_coords >= width) | \
                        (y_coords < 0) | (y_coords >= height)
        in_bounds = ~out_of_bounds
        # 确保 in_bounds 的形状与 num_views 匹配 [n, 1]
        in_bounds = in_bounds.unsqueeze(1)  # 从 [n] 变为 [n, 1]
        num_views += in_bounds.to(dtype=torch.int32)



        weights /= weights_cnt + 1e-7
        # print("weights shape: ", weights.shape)
        # print("weight max: ", weights.max())
        # print("weight min: ", weights.min())
        # print("weight_cnt shape: ", weights_cnt.shape)
        # print("weight_cnt max: ", weights_cnt.max()) # 这个数值表示某个Gaussian在当前视角下被采样了320094次
        # print("weight_cnt min: ", weights_cnt.min())

        # weights shape:  torch.Size([1119604, 1])
        # weight max:  tensor(1., device='cuda:0')
        # weight min:  tensor(0., device='cuda:0')
        # weight_cnt shape:  torch.Size([1119604, 1])
        # weight_cnt max:  tensor(320094, device='cuda:0', dtype=torch.int32)
        # weight_cnt min:  tensor(0, device='cuda:0', dtype=torch.int32)
        # Found 31650 non-zero weights. Randomly selecting up to 10 to print:
        # mask_image resize:  (1600, 899)
        # find camera:  Inst_1_test_IMG_2870.jpg

        # visualize weights
        non_zero_indices = torch.where(weights.squeeze() != 0)[0]
        num_non_zero = non_zero_indices.shape[0]
        if num_non_zero > 0:
            print(f"Found {num_non_zero} non-zero weights. Randomly selecting up to 10 to print:")
            num_to_select = min(10, num_non_zero)

        # Random sample 1 (original index 839997): 1.0
        # Random sample 2 (original index 2461128): 1.0
        # Random sample 3 (original index 2915355): 1.0
        # Random sample 4 (original index 1194257): 1.0
        # Random sample 5 (original index 38004): 0.4000000059604645
        # Random sample 6 (original index 1900549): 1.0
        # Random sample 7 (original index 931510): 1.0
        # Random sample 8 (original index 20591): 0.94669508934021
        # Random sample 9 (original index 1858351): 1.0
        # Random sample 10 (original index 392264): 0.190476194024086


        cumulative_weights += weights
    
    print("max in num_views", num_views.max())
    print("min in num_views", num_views.min())

    # cumulative_weights /= num_masks + 1e-7
    # TODO (!) consider the dropped views
    # 避免 num_views 中的 0 导致除以 0 的错误
    safe_num_views = num_views.clone()
    safe_num_views[safe_num_views == 0] = 1 
    cumulative_weights /= safe_num_views + 1e-7 

    # cumulative_weights /= num_view + 1e-7 

    print("cumulative_weights shape: ", cumulative_weights.shape)
    print("cumulative_weights max: ", cumulative_weights.max())
    print("cumulative_weights min: ", cumulative_weights.min())

    # selected_mask = cumulative_weights > threshold 
    # selected_mask = selected_mask[:, 0]

    # print("selected_mask shape: ", selected_mask.shape)
    # print("selected_mask max: ", selected_mask.max())
    # print("selected_mask min: ", selected_mask.min())

    # # 7021 / 1119604
    # print(f"\n Total Gaussians number: {selected_mask.sum().item()} / {selected_mask.numel()}")

    end_time = time.time()
    print("end_time: ", end_time)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
    print("Done")

    return cumulative_weights.repeat(1,3), 0.0 # remain_rate




def trace_and_prune(config, dataset, opt, pipe, iteration, mask_folder, output_ply, threshold, width=1600, height=1200, zero_fill=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("dataset.sh degree", dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree)  

    scene = Scene(dataset, gaussians, load_iteration=iteration) 
    gaussians.training_setup(opt)
    train_cameras = scene.getTrainCameras()

    import time
    start_time = time.time()
    print("start_time: ", start_time)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    num_gaussians = gaussians.get_scaling.shape[0]
    cumulative_weights = torch.zeros_like(gaussians._opacity, dtype=torch.float32, device=device) # cumulative weights type: tensor([n, 1])

    print("cumulative_weight shape: ", cumulative_weights.shape)

    mask_files = sorted(os.listdir(mask_folder))
    num_masks = len(mask_files)
    print(f"Found {num_masks} mask files.")

    error_views = config["mask"]["error_views"]
    if_prune = config["mask"]["prune"]

    num_view = 0
    for mask_file in tqdm(mask_files, desc="loading masks..."):
        # print("mask_file:", mask_file)
        if mask_file in error_views:
            print("ignore error view")
            continue
        num_view += 1
        mask_path = os.path.join(mask_folder, mask_file)
        print(f"\n mask: {mask_path}")
        id_mask = load_binary_mask(mask_path, device, width=width, height=height)
        id_mask = id_mask.float()
        id_mask = id_mask.unsqueeze(0)
        mask_name = os.path.basename(mask_path)
        # img_name_str = mask_name.split('.')[0].split('_')[0]
        img_name_str = mask_name.split('.')[0]

        try:
            img_uid = int(img_name_str)
        except:
            img_uid = int(img_name_str[:5])

        # XXX 3dgscd dataset uses arbitrary bits for image names, but blender dataset use 5 bits images name
        if zero_fill:
            img_name = f"whole_{img_uid:05d}.png"  # e.g. whole_00023.png # XXX Ours blender dataset
        else:
            img_name = 'whole_'+ str(img_uid) + ".png"

        # raw_image_names = sorted(os.listdir(os.path.join(dataset.source_path, dataset.images)))
        # # pre or post
        # # print("raw_image_names:", raw_image_names)

        # try:
        #     img_name_int = int(img_name_str)
        #     img_name = raw_image_names[img_name_int]
        #     print("img_name_int:", img_name_int)
        #     print("img_name:", img_name)
        # except (ValueError, IndexError) as e:
        #     print(f"Cannot parse mask filename '{mask_name}' to get corresponding image name. Skipping this mask.")
        #     continue

        viewpoint_camera = None
        for camera in train_cameras:
            if camera.image_name == img_name:
                # print("find camera: ", camera.image_name)
                viewpoint_camera = camera
                break

        if viewpoint_camera is None:
            print(f"Camera corresponding to image {img_name} not found. Skipping this mask.")
            return 
        
        # print("viewpoint cam: ",viewpoint_camera.image_name)

        weights = torch.zeros_like(gaussians._opacity)
        weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)

        gaussians.apply_weights(viewpoint_camera, weights, weights_cnt, id_mask)

        weights /= weights_cnt + 1e-7
        # print("weights shape: ", weights.shape)
        # print("weight max: ", weights.max())
        # print("weight min: ", weights.min())

        non_zero_indices = torch.where(weights.squeeze() != 0)[0]
        num_non_zero = non_zero_indices.shape[0]

        # if num_non_zero > 0:
        #     print(f"Found {num_non_zero} non-zero weights. Randomly selecting up to 10 to print:")
        #     num_to_select = min(10, num_non_zero)
            
        #     # Ensure random_indices are within the bounds of non_zero_indices
        #     if num_non_zero > 0: # Check if there are any non-zero elements
        #         perm = torch.randperm(num_non_zero)
        #         selected_indices_in_non_zero = perm[:num_to_select]
        #         random_actual_indices = non_zero_indices[selected_indices_in_non_zero]
            
        #         for i, idx in enumerate(random_actual_indices):
        #             print(f"  Random sample {i+1} (original index {idx.item()}): {weights[idx].item()}")
        #     else:
        #         print("No non-zero weights to sample from.")
        # else:
        #     print("No non-zero weights found in this view.")

        # Random sample 1 (original index 839997): 1.0
        # Random sample 2 (original index 2461128): 1.0
        # Random sample 3 (original index 2915355): 1.0
        # Random sample 4 (original index 1194257): 1.0
        # Random sample 5 (original index 38004): 0.4000000059604645
        # Random sample 6 (original index 1900549): 1.0
        # Random sample 7 (original index 931510): 1.0
        # Random sample 8 (original index 20591): 0.94669508934021
        # Random sample 9 (original index 1858351): 1.0
        # Random sample 10 (original index 392264): 0.190476194024086


        cumulative_weights += weights
    
    # cumulative_weights /= num_masks + 1e-7
    cumulative_weights /= num_view + 1e-7 # consider the dropped views

    print("cumulative_weights shape: ", cumulative_weights.shape)
    print("cumulative_weights max: ", cumulative_weights.max())
    print("cumulative_weights min: ", cumulative_weights.min())

    selected_mask = cumulative_weights > threshold 
    selected_mask = selected_mask[:, 0]

    # print("selected_mask shape: ", selected_mask.shape)
    # print("selected_mask max: ", selected_mask.max())
    # print("selected_mask min: ", selected_mask.min())


    # print(f"\n Total Gaussians number: {selected_mask.sum().item()} / {selected_mask.numel()}")

    end_time = time.time()
    print("end_time: ", end_time)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    pruned_color, remain_rate = prune(gaussians, scene, config, dataset, iteration, pipe, cumulative_weights, mask_folder, False)

    print("Done")
    
    if if_prune:
        return pruned_color, remain_rate

    else:
        return cumulative_weights.repeat(1,3), remain_rate



def trace_and_remove(config, dataset, opt, pipe, iteration, mask_folder, output_ply, threshold, width=1600, height=1200):
    '''
    - Save target object
    - ply
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("dataset.sh degree", dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree, 
                              anchor_weight_init_g0=1.0,
                              anchor_weight_init=0.1,
                              anchor_weight_multiplier=2,
                              )  

    # scene = Scene(dataset, gaussians, load_iteration=30000)  
    scene = Scene(dataset, gaussians, load_iteration=iteration) # XXX trace in incre scene!  
    gaussians.training_setup(opt)
    train_cameras = scene.getTrainCameras()

    import time
    start_time = time.time()
    print("start_time: ", start_time)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=device)

    num_gaussians = gaussians.get_scaling.shape[0]
    cumulative_weights = torch.zeros_like(gaussians._opacity, dtype=torch.float32, device=device) # cumulative weights type: tensor([n, 1])

    print("cumulative_weight shape: ", cumulative_weights.shape)

    mask_files = sorted(os.listdir(mask_folder))
    num_masks = len(mask_files)
    print(f"Found {num_masks} mask files.")

    error_views = config["mask"]["error_views"]

    num_view = 0
    for mask_file in tqdm(mask_files, desc="loading masks..."):
        # print("mask_file:", mask_file)
        if mask_file in error_views:
            print("ignore error view")
            continue
        num_view += 1
        mask_path = os.path.join(mask_folder, mask_file)
        # print(f"\n mask: {mask_path}")
        id_mask = load_binary_mask(mask_path, device, width=width, height=height)
        id_mask = id_mask.float()
        id_mask = id_mask.unsqueeze(0)
        mask_name = os.path.basename(mask_path)
        # img_name_str = mask_name.split('.')[0].split('_')[0]
        img_name_str = mask_name.split('.')[0]

        try:
            img_uid = int(img_name_str)
        except:
            img_uid = int(img_name_str[:5])

        # XXX 3dgscd dataset uses arbitrary bits for image names, but blender dataset use 5 bits images name
        # img_name = 'whole_'+ str(img_uid) + ".png"
        img_name = f"whole_{img_uid:05d}.png"  # e.g. whole_00023.png

        viewpoint_camera = None
        for camera in train_cameras:
            if camera.image_name == img_name:
                # print("find camera: ", camera.image_name)
                viewpoint_camera = camera
                break

        if viewpoint_camera is None:
            print(f"Camera corresponding to image {img_name} not found. Skipping this mask.")
            return None, 0.0
        
        # print("viewpoint cam: ",viewpoint_camera.image_name)

        weights = torch.zeros_like(gaussians._opacity)
        weights_cnt = torch.zeros_like(gaussians._opacity, dtype=torch.int32)

        gaussians.apply_weights(viewpoint_camera, weights, weights_cnt, id_mask)

        weights /= weights_cnt + 1e-7
        # print("weights shape: ", weights.shape)

        cumulative_weights += weights
    
    # cumulative_weights /= num_masks + 1e-7
    cumulative_weights /= num_view + 1e-7 # consider the dropped views

    print("cumulative_weights shape: ", cumulative_weights.shape)
    print("cumulative_weights max: ", cumulative_weights.max())
    print("cumulative_weights min: ", cumulative_weights.min())

    traced_mask = cumulative_weights > threshold 
    traced_mask = traced_mask[:, 0]  # Convert to 1D

    print("traced_mask shape: ", traced_mask.shape)
    print("traced_mask max: ", traced_mask.max())
    print("traced_mask min: ", traced_mask.min())

    traced_count = traced_mask.sum().item()
    total_count = traced_mask.numel()
    remaining_count = traced_count

    print(f"\nTotal Gaussians: {total_count}")
    print(f"Traced (to be removed) Gaussians: {traced_count}")
    print(f"Remaining Gaussians: {remaining_count}")
    print(f"Removal ratio: {traced_count / total_count:.2%}")
    print(f"Remaining ratio: {remaining_count / total_count:.2%}")

    # Remove traced parts from Gaussians
    if traced_count > 0:
        print("Removing traced Gaussians...")
        
        # Use prune_points method to remove traced Gaussians
        # prune_points accepts a deletion mask (True means delete)
        remove_mask = torch.logical_not(traced_mask)  
        gaussians.prune_points(remove_mask)
        
        print(f"After removal - Remaining Gaussians: {gaussians.get_xyz.shape[0]}")
    else:
        print("No Gaussians to remove (all weights below threshold).")

    # If output_ply exists, delete old ply file
    if os.path.exists(output_ply):
        os.remove(output_ply)

    # Save remaining Gaussians
    gaussians.save_ply(output_ply)
    print(f"Remaining Gaussians saved to: {output_ply}")

    end_time = time.time()
    print("end_time: ", end_time)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")

    print("Done")
    
    remain_rate = remaining_count / total_count if total_count > 0 else 0.0
    return output_ply, remain_rate
    


def trace_remove_post(config, dataset, opt, pipe, iteration, mask_folder, output_ply, threshold, boundary_a, width=1600, height=1200, reverse=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("dataset.sh degree", dataset.sh_degree)
    gaussians = GaussianModel(dataset.sh_degree, 
                              anchor_weight_init_g0=1.0,
                              anchor_weight_init=0.1,
                              anchor_weight_multiplier=2,
                              )  

    # Load scene and setup
    scene = Scene(dataset, gaussians, load_iteration=iteration)
    gaussians.training_setup(opt)
    train_cameras = scene.getTrainCameras()

    # Create color_raw with correct shape [N, 1] to work with repeat(1, 3) in trace_prune
    color_raw = torch.zeros_like(gaussians.get_scaling)[:, 0:1]  # Shape [N, 1]
    # idx 大于 boundary a 的是1
    color_raw[:] = 1.0

    # 8.16之前版本
    # color_new, _ = trace_prune_nobound(config, dataset, iteration, pipe, color_raw, mask_folder, False) # shape n,3
    # prune的是0 其它是1

    color_new, _ = trace(config, dataset, opt, pipe, iteration, mask_folder, None, 0.1, width, height) # shape n,3


    traced_gs = color_new[:,0] > 0.01  # Convert to boolean mask # 改这个没用
    print("sum of traced_gs:", traced_gs.sum().item())
    if reverse:
        prune_mask = traced_gs
    else: 
        prune_mask = ~traced_gs  # Gaussians that were NOT traced

    traced_count = prune_mask.sum(0)

    if traced_count > 0:
        print("Removing traced Gaussians...")
        
        # 使用prune_points方法移除被trace到的Gaussians
        # prune_points接受的是删除mask（True表示删除）
        gaussians.prune_points(prune_mask)
        
        print(f"After removal - Remaining Gaussians: {gaussians.get_xyz.shape[0]}")
    else:
        print("No Gaussians to remove (all weights below threshold).")

    # 如果output_ply存在，删除旧的ply文件
    if os.path.exists(output_ply):
        os.remove(output_ply)

    # 保存剩余的Gaussians
    gaussians.save_ply(output_ply)
    print(f"Remaining Gaussians saved to: {output_ply}")



    print(f"\nTracing Statistics:")
    print(f"Total Gaussians: {traced_gs.shape[0]}")
    print(f"Traced Gaussians: {traced_gs.sum().item()}")

    # return output_ply


