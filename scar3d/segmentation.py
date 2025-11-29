import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from PIL import Image
import io
import os
import cv2
from scipy import ndimage

from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from torchvision.ops.boxes import batched_nms, box_area

def process_small_region(rles, device):
    """
    Avoid noise and small holes in the mask.
    """
    new_masks = []
    scores = []
    min_area = 100
    nms_thresh = 0.7

    for rle in rles:
        mask = torch.from_numpy(rle_to_mask(rle[0]).astype(np.uint8)).to(device)

        # Convert mask to NumPy for compatibility with remove_small_regions
        mask_np = mask.cpu().numpy()

        mask_np, changed = remove_small_regions(mask_np, min_area, mode="holes")
        unchanged = not changed
        mask_np, changed = remove_small_regions(mask_np, min_area, mode="islands")
        unchanged = unchanged and not changed

        # Convert back to PyTorch Tensor
        mask = torch.from_numpy(mask_np).to(device)

        new_masks.append(mask.unsqueeze(0))
        # Give score=0 to changed masks and score=1 to unchanged masks
        scores.append(float(unchanged))

    if new_masks == []:
        return []
    masks = torch.cat(new_masks, dim=0)
    boxes = batched_mask_to_box(masks)
    keep_by_nms = batched_nms(
        boxes.float(),
        torch.as_tensor(scores, device=device),
        torch.zeros_like(boxes[:, 0], device=device),  # categories
        iou_threshold=nms_thresh,
    )

    for i_mask in keep_by_nms:
        if scores[i_mask] == 0.0:
            mask_torch = masks[i_mask].unsqueeze(0)
            rles[i_mask] = mask_to_rle_pytorch(mask_torch)
    masks = [rle_to_mask(rles[i][0]) for i in keep_by_nms]
    return masks

def get_predictions_given_embeddings_and_queries(img, points, point_labels, model):
    print("shape of img", img.shape)  # (3, 1200, 1600)
    print("shape of points", points.shape)  # (1, 1024, 1, 2)
    print("shape of point_labels", point_labels.shape)  # (1, 1024, 1)

    predicted_masks, predicted_iou = model(img[None, ...], points, point_labels)

    print("shape of predicted_iou", predicted_iou.shape)  # (1, 1024, 3)
    print("shape of predicted_masks", predicted_masks.shape)  # (1, 1024, 3, 1200, 1600)

    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None], dim=2)

    predicted_masks = predicted_masks[0]

    iou = predicted_iou_scores[0, :, 0]
    index_iou = iou > 0.7
    iou_ = iou[index_iou]
    masks = predicted_masks[index_iou]

    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]

    masks = torch.ge(masks, 0.0)
    print("shape of masks", masks.shape)  # (n, 3, 1200, 1600)
    print("shape of iou_", iou_.shape)  # (963,)

    return masks, iou_

def process_small_region_bool(masks, thre=10):
    
    from scipy import ndimage
    
    # Convert to numpy if it's a tensor
    if isinstance(masks, torch.Tensor):
        masks_np = masks.cpu().numpy().astype(bool)
    else:
        masks_np = masks.astype(bool)
    
    # Find connected components
    labeled_mask, num_components = ndimage.label(masks_np)
    
    # Create a new mask to store the result
    filtered_mask = np.zeros_like(masks_np, dtype=bool)
    
    # Process each connected component
    for component_id in range(1, num_components + 1):  # component IDs start from 1
        component_mask = (labeled_mask == component_id)
        component_area = np.sum(component_mask)
        
        # Keep components with area >= threshold
        if component_area >= thre:
            filtered_mask |= component_mask
    
    return filtered_mask

    


def validate_mask(predicted_masks, mask_bool, thre=0.01):
    '''
    1. split mask_bool into connected components
    2. for each predicted masks, find out the most similar connected component in mask_bool
    3. calculate the intersection over union (IoU) between the predicted mask and the connected component
    4. print the IoU
    5. return the predicted masks that have IoU >= thre
    '''
    from scipy import ndimage
    
    # Step 1: Split mask_bool into connected components
    labeled_mask, num_components = ndimage.label(mask_bool)
    print(f"Found {num_components} connected components in ground truth mask")
    
    valid_masks = []
    valid_ious = []
    
    # Process each predicted mask
    for i, pred_mask in enumerate(predicted_masks):
        if isinstance(pred_mask, torch.Tensor):
            pred_mask_np = pred_mask.cpu().numpy().astype(bool)
        else:
            pred_mask_np = pred_mask.astype(bool)
        
        best_iou = 0.0
        best_component_id = -1
        
        # Step 2: Find the most similar connected component for this predicted mask
        for component_id in range(1, num_components + 1):  # component IDs start from 1
            component_mask = (labeled_mask == component_id)
            
            # Step 3: Calculate IoU between predicted mask and this component
            intersection = np.logical_and(pred_mask_np, component_mask).sum()
            union = np.logical_or(pred_mask_np, component_mask).sum()
            
            if union > 0:
                iou = intersection / union
                if iou > best_iou:
                    best_iou = iou
                    best_component_id = component_id
        
        # Step 4: Print the IoU
        print(f"Predicted mask {i}: Best IoU = {best_iou:.4f} with component {best_component_id}")
        
        # Step 5: Keep masks with IoU >= threshold
        if best_iou >= thre:
            valid_masks.append(predicted_masks[i])
            valid_ious.append(best_iou)
    
    print(f"Kept {len(valid_masks)} out of {len(predicted_masks)} masks with IoU >= {thre}")
    
    return valid_masks



def run_everything_ours(no, img_path, model, device, mask_path, traced_bin_path, rend2bin, IoU_thre, GRID_SIZE, batch_size=8):
    model = model.to(device)

    # 加载mask
    image_mask = Image.open(mask_path).convert("L")
    mask_bool = np.array(image_mask) > rend2bin
    mask_h = mask_bool.shape[0]
    mask_w = mask_bool.shape[1]

    # 加载原图
    print(f"Loading image from {img_path}")
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = np.array(image)
    image = cv2.resize(image, (mask_w, mask_h)) # 和mask形状相同

    img_tensor = ToTensor()(image).to(device)
    # _, original_image_h, original_image_w = img_tensor.shape

    
    mask_bool = process_small_region_bool(mask_bool)

    print(f"Saving traced binary mask to {traced_bin_path}")
    cv2.imwrite(traced_bin_path, mask_bool.astype(np.uint8) * 255)

    # HERE resize1
    image_mask = np.array(image_mask)
    # image = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4))

    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * mask_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * mask_h
            # 只添加在mask_bool区域内的点
            if mask_bool[int(curr_y), int(curr_x)]:
                xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy)).to(device)

    points = xy
    num_pts = xy.shape[0]
    point_labels = torch.ones(num_pts, 1, device=device)

    if num_pts == 0:
        print("No valid points found in the mask!")
        return []

    # 写一段代码 把所有points用matplotlib画在图片上 可视化一下
    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), s=1)
    # plt.savefig(f"data/test_imgs/points_{no}.png")

    # 3. 分批处理预测
    all_predicted_masks = []
    all_predicted_iou = []
    
    num_batches = (num_pts + batch_size - 1) // batch_size  # 向上取整
    print(f"Processing {num_pts} points in {num_batches} batches")
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, num_pts)
            batch_points = points[start_idx:end_idx]
            batch_size_actual = batch_points.shape[0]
            
            print(f"Processing batch {batch_idx + 1}/{num_batches}: points {start_idx} to {end_idx-1} ({batch_size_actual} points)")
            
            # 为当前批次创建point_labels
            batch_point_labels = torch.ones(batch_size_actual, 1, device=device)
            
            # 4. 对当前批次返回预测的mask
            try:
                batch_predicted_masks, batch_predicted_iou = get_predictions_given_embeddings_and_queries(
                    img_tensor,
                    batch_points.reshape(1, batch_size_actual, 1, 2),
                    batch_point_labels.reshape(1, batch_size_actual, 1),
                    model
                )
                
                # 收集批次结果
                all_predicted_masks.append(batch_predicted_masks)
                all_predicted_iou.append(batch_predicted_iou)
                
                print(f"Batch {batch_idx + 1} completed: {batch_predicted_masks.shape[0]} masks")
                
            except Exception as e:
                print(f"Error processing batch {batch_idx + 1}: {e}")
                continue
    
    # 5. 合并所有批次的结果
    if len(all_predicted_masks) == 0:
        print("No valid predictions from any batch!")
        return []
    
    # 沿着第一个维度拼接所有批次的结果
    predicted_masks = torch.cat(all_predicted_masks, dim=0)
    predicted_iou = torch.cat(all_predicted_iou, dim=0)  
    
    print(f"Merged results: predicted_masks.shape = {predicted_masks.shape}")
    print(f"Merged results: predicted_iou.shape = {predicted_iou.shape}")

    rle = [mask_to_rle_pytorch(m[0:1]) for m in predicted_masks]
    predicted_masks = process_small_region(rle, device)
    if len(predicted_masks) == 0:
        print("No valid masks found after processing small regions!")
        return []
    # print("predicted_masks: ", predicted_masks)
    print("len(predicted_masks): ", len(predicted_masks))
    print("predicted_masks[0].shape: ", predicted_masks[0].shape)

    predicted_masks = validate_mask(predicted_masks, mask_bool, IoU_thre)


    return predicted_masks


def show_anns_ours(mask, ax):
    ax.set_autoscale_on(False)
    img = np.ones((mask[0].shape[0], mask[0].shape[1], 4))
    img[:, :, 3] = 0
    for ann in mask:
        m = ann
        color_mask = np.concatenate([np.random.random(3), [0.5]])
        img[m] = color_mask
    ax.imshow(img)


def get_prompted_mask(render_dir, config, image_dir, masks_dir, traced_bin_dir, view_pre, no_img_prefix=False, zero_fill=True):
    from efficient_sam.build_efficient_sam import build_efficient_sam_vits
    # from efficient_sam.build_efficient_sam import build_efficient_sam_vitt

    import time
    start_time = time.time()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    efficient_sam_vits_model = build_efficient_sam_vits().to(device)
    # efficient_sam_vits_model = build_efficient_sam_vitt().to(device)

    efficient_sam_vits_model.eval()

    masks_name = os.listdir(render_dir)
    masks_name.sort()
    
    skip_name = []

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(traced_bin_dir, exist_ok=True)

    # read config in yaml file
    import yaml
    with open(config, 'r') as f:
        config_data = yaml.safe_load(f)
        rend2bin = config_data["mask"]["render_to_bin"]
        IoU_thre = config_data["mask"]["IoU_threshold"]
        grid_size = config_data["mask"]["grid_size"]

    for name in masks_name:
        no = int(name.split(".")[0])
        if zero_fill:
            image_path = os.path.join(image_dir, "whole_" + f"{no:05d}" + ".png")
        else: 
            image_path = os.path.join(image_dir, "whole_" + str(no)+ ".png")
            # train_origin
            # image_path = os.path.join(image_dir, f"{no:05d}" + ".png")
        print("image_path: ", image_path)

        print("no_img_prefix: ", no_img_prefix)
        if no_img_prefix:
            image_path = os.path.join(image_dir, f"{no:05d}" + ".png")
            print("image_path with no_img_prefix: ", image_path)

        print("Processing image: ", image_path)
        
        mask_path = os.path.join(render_dir, name)
        bin_path = os.path.join(traced_bin_dir, name)

        all_masks = []
        image_shape = None

        # grid size 是一个数组 
        for GRID_SIZE in grid_size:
            print(f"Processing with grid size: {GRID_SIZE}")

            # try:
            mask_efficient_sam_vits = run_everything_ours(no, image_path, efficient_sam_vits_model, device, mask_path, bin_path, rend2bin, IoU_thre, GRID_SIZE, 64)
                # mask_efficient_sam_vits = run_everything_center(no, image_path, efficient_sam_vits_model, device, mask_path, bin_path, rend2bin, IoU_thre)
                # mask_efficient_sam_vits_cent = run_everything_center(no, image_path, efficient_sam_vits_model, device, mask_path, bin_path, rend2bin, IoU_thre, 64)
            # except Exception as e:
            #     print(f"Error processing {name}: {e}")
            #     skip_name.append(name)
            #     continue

            # 收集当前grid_size的所有masks
            all_masks.extend(mask_efficient_sam_vits)
            print(f"Added {len(mask_efficient_sam_vits)} masks from grid size {GRID_SIZE}")

            # all_masks.extend(mask_efficient_sam_vits_cent)
            # print(f"Added {len(mask_efficient_sam_vits_cent)} masks from center points")

            # if len(mask_efficient_sam_vits) == 0:
            #     print(f"No valid masks found for {name}, skipping...")
            #     skip_name.append(name)
            #     continue

            # validate_mask(mask_efficient_sam_vits, mask_path)

            # show_anns_ours(mask_efficient_sam_vits, ax[1])
            # ax[1].title.set_text("EfficientSAM")
            # ax[1].axis('off')
            # plt.savefig("data/test_imgs/efficient_sam.png")
        if len(all_masks) == 0:
            print(f"No valid masks found for {name} from any grid size, skipping...")
            skip_name.append(name)
            continue

        if image_shape is None:
            image_shape = all_masks[0].shape

        # 合并所有masks (取并集)
        print(f"Merging {len(all_masks)} masks for {name}")

          # 创建合并后的mask图像
        merged_mask = np.zeros((image_shape[0], image_shape[1]), dtype=bool)
        
        # 对所有masks取并集
        for mask in all_masks:
            if isinstance(mask, torch.Tensor):
                mask_np = mask.cpu().numpy().astype(bool)
            else:
                mask_np = mask.astype(bool)
            merged_mask = np.logical_or(merged_mask, mask_np)

        # 创建彩色可视化图像
        mask_only_img = np.zeros((image_shape[0], image_shape[1], 4))
        
        # 为合并后的mask分配颜色
        if np.any(merged_mask):
            color_mask = np.concatenate([np.random.random(3), [0.8]])  # 稍微增加透明度
            mask_only_img[merged_mask] = color_mask

        # 保存合并后的mask
        save_path = os.path.join(masks_dir, name)
        mask_only_img = (mask_only_img * 255).astype(np.uint8)
        cv2.imwrite(save_path, mask_only_img[:, :, :3])
        print(f"Saved merged mask to {save_path}")

    # mask = mask_efficient_sam_vits
    # fig, ax = plt.subplots(figsize=(15, 15))
    # mask_only_img = np.zeros((mask[0].shape[0], mask[0].shape[1], 4))

    # for ann in mask:
    #     m = ann
    #     color_mask = np.concatenate([np.random.random(3), [0.5]])
    #     mask_only_img[m] = color_mask

    # # ax.imshow(mask_only_img)
    # # ax.axis('off')
    # # plt.show()
    # plt.savefig("data/test_imgs/mask_only.png")

    # save_path = os.path.join(masks_dir, name)
    # # 怎么保存mask_only_img？不用plt？
    # mask_only_img = (mask_only_img * 255).astype(np.uint8)
    # cv2.imwrite(save_path, mask_only_img[:, :, :3])
    # print(f"Saved mask to {save_path}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total processing time: {elapsed_time:.2f} seconds")
    


def get_prompted_mask_difference(render_dir, config, image_dir, masks_dir, traced_bin_dir):
    from efficient_sam.build_efficient_sam import build_efficient_sam_vits

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    efficient_sam_vits_model = build_efficient_sam_vits().to(device)

    efficient_sam_vits_model.eval()

    masks_name = os.listdir(render_dir)
    masks_name.sort()
    print("render_dir: ", render_dir)
    print("masks_name: ", masks_name)
    
    skip_name = []

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(traced_bin_dir, exist_ok=True)

    masks = {}

    for name in masks_name:
        no = int(name.split(".")[0])
        image_path = os.path.join(image_dir, name)
        print("Processing image: ", image_path)

        # fig, ax = plt.subplots(1, 2, figsize=(30, 30))
       
        
        rend2bin = 128
        GRAD_SIZE = config["mask"]["overlap_grid_size"]
        IoU_thre = config["mask"]["overlap_IoU_threshold"]
        
        
        mask_path = os.path.join(render_dir, name)
        bin_path = os.path.join(traced_bin_dir, name)

        try:
            mask_efficient_sam_vits = run_everything_ours(no, image_path, efficient_sam_vits_model, device, mask_path, bin_path, rend2bin, IoU_thre, GRAD_SIZE)
        except Exception as e:
            print(f"Error processing {name}: {e}")
            skip_name.append(name)
            continue

        if len(mask_efficient_sam_vits) == 0:
            print(f"No valid masks found for {name}, skipping...")
            skip_name.append(name)
            continue

        # validate_mask(mask_efficient_sam_vits, mask_path)

        # HERE resize2
        # mask_efficient_sam_vits = [cv2.resize(m.cpu().numpy(), (original_image_w, original_image_h)) for m in mask_efficient_sam_vits]


        # show_anns_ours(mask_efficient_sam_vits, ax[1])
        # ax[1].title.set_text("EfficientSAM")
        # ax[1].axis('off')
        # plt.savefig("data/test_imgs/efficient_sam.png")

        mask = mask_efficient_sam_vits
        fig, ax = plt.subplots(figsize=(15, 15))
        mask_only_img = np.zeros((mask[0].shape[0], mask[0].shape[1], 4))

        for ann in mask:
            m = ann
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            mask_only_img[m] = color_mask

        # ax.imshow(mask_only_img)
        # ax.axis('off')
        # plt.show()
        # plt.savefig("data/test_imgs/mask_only.png")

        save_path = os.path.join(masks_dir, name)
        # 怎么保存mask_only_img？不用plt？
        mask_only_img = (mask_only_img * 255).astype(np.uint8)
        cv2.imwrite(save_path, mask_only_img[:, :, :3])
        print(f"Saved mask to {save_path}")

        # 转换为二值图

        masks[name] = mask_only_img
    
    return masks
    # debug 完了后 把输入输出交互全注释掉 会慢


# def generate_patches(image_height, image_width, patch_size, overlap):
#         """生成patch的坐标列表"""
#         patch_w, patch_h = patch_size
#         patches = []
        
#         y_start = 0
#         while y_start < image_height:
#             y_end = min(y_start + patch_h, image_height)
            
#             x_start = 0
#             while x_start < image_width:
#                 print(f"Generating patch at x_start: {x_start}, y_start: {y_start}")
#                 x_end = min(x_start + patch_w, image_width)
                
#                 patches.append({
#                     'x_start': x_start,
#                     'y_start': y_start,
#                     'x_end': x_end,
#                     'y_end': y_end,
#                     'width': x_end - x_start,
#                     'height': y_end - y_start
#                 })
                
#                 if x_end >= image_width:
#                     break
#                 x_start = x_end - overlap
            
#             if y_end >= image_height:
#                 break
#             y_start = y_end - overlap
        
#         return patches

# def extract_patch(image, patch_info):
#     """从图像中提取patch"""
#     return image[patch_info['y_start']:patch_info['y_end'], 
#                 patch_info['x_start']:patch_info['x_end']]

# def merge_patches(patches_results, original_shape, patches_info, overlap):
#     """将多个patch的结果合并成原始大小的图像"""
#     original_h, original_w = original_shape
#     merged_mask = np.zeros((original_h, original_w), dtype=bool)
#     weight_map = np.zeros((original_h, original_w), dtype=np.float32)
    
#     for patch_masks, patch_info in zip(patches_results, patches_info):
#         if len(patch_masks) == 0:
#             continue
            
#         # 为当前patch创建合并的mask
#         patch_h, patch_w = patch_info['height'], patch_info['width']
#         patch_merged = np.zeros((patch_h, patch_w), dtype=bool)
        
#         # 合并当前patch的所有masks
#         for mask in patch_masks:
#             if isinstance(mask, torch.Tensor):
#                 mask_np = mask.cpu().numpy().astype(bool)
#             else:
#                 mask_np = mask.astype(bool)
#             patch_merged = np.logical_or(patch_merged, mask_np)
        
#         # 创建权重图（边缘权重较低，中心权重较高）
#         patch_weight = np.ones((patch_h, patch_w), dtype=np.float32)
#         if overlap > 0:
#             # 在重叠区域使用渐变权重
#             fade_pixels = min(overlap // 2, min(patch_h, patch_w) // 4)
#             for i in range(fade_pixels):
#                 weight = (i + 1) / fade_pixels
#                 # 上边缘
#                 if patch_info['y_start'] > 0:
#                     patch_weight[i, :] *= weight
#                 # 下边缘
#                 if patch_info['y_end'] < original_h:
#                     patch_weight[-(i+1), :] *= weight
#                 # 左边缘
#                 if patch_info['x_start'] > 0:
#                     patch_weight[:, i] *= weight
#                 # 右边缘
#                 if patch_info['x_end'] < original_w:
#                     patch_weight[:, -(i+1)] *= weight
        
#         # 将patch结果放回原始位置
#         y_start, y_end = patch_info['y_start'], patch_info['y_end']
#         x_start, x_end = patch_info['x_start'], patch_info['x_end']
        
#         # 加权合并
#         current_weight = weight_map[y_start:y_end, x_start:x_end]
#         current_mask = merged_mask[y_start:y_end, x_start:x_end].astype(np.float32)
        
#         new_weight = current_weight + patch_weight
#         new_mask = (current_mask * current_weight + patch_merged.astype(np.float32) * patch_weight) / (new_weight + 1e-8)
        
#         merged_mask[y_start:y_end, x_start:x_end] = new_mask > 0.5
#         weight_map[y_start:y_end, x_start:x_end] = new_weight
    
#     return merged_mask




# def get_prompted_mask_patched(render_dir, config, image_dir, masks_dir, traced_bin_dir, view_pre, patch_size=(400, 400), overlap=50, no_img_prefix=False):
#     from efficient_sam.build_efficient_sam import build_efficient_sam_vits
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     efficient_sam_vits_model = build_efficient_sam_vits().to(device)
#     efficient_sam_vits_model.eval()

#     masks_name = os.listdir(render_dir)
#     masks_name.sort()
    
#     skip_name = []
#     os.makedirs(masks_dir, exist_ok=True)
#     os.makedirs(traced_bin_dir, exist_ok=True)
        
#     patch_dir = os.path.dirname(traced_bin_dir)
#     patch_dir = os.path.join(patch_dir, "patches")
#     os.makedirs(patch_dir, exist_ok=True)

#     import yaml
#     with open(config, 'r') as f:
#         config_data = yaml.safe_load(f)
#         rend2bin = config_data["mask"]["render_to_bin"]
#         IoU_thre = config_data["mask"]["IoU_threshold"]
#         grid_size = config_data["mask"]["grid_size"]

#     for name in masks_name:
#         print(f"Processing {name}...")
        
#         no = int(name.split(".")[0])

#         # 为当前图像创建专门的patch目录
#         image_patch_dir = os.path.join(patch_dir, f"image_{no:05d}")
#         os.makedirs(image_patch_dir, exist_ok=True)
        
#         # In train_origin, 1600 * 899
#         if no_img_prefix:
#             image_path = os.path.join(image_dir, f"{no:05d}.png")
#         else:
#             image_path = os.path.join(image_dir, f"{no:05d}.png") # ???
        
#         print(f"Processing image: {image_path}")
        
#         # output
#         mask_path = os.path.join(render_dir, name)
#         bin_path = os.path.join(traced_bin_dir, name)
        
#         # 读取图像
#         image = cv2.imread(image_path)
#         print(f"Loaded image from {image_path}, shape: {image.shape}")
#         image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
#         # 读取mask
#         mask_image = Image.open(mask_path).convert("L")
#         print(f"Loaded mask from {mask_path}, size: {mask_image.size}")
#         mask_array = np.array(mask_image)

#         if image.shape[:2] != mask_array.shape:
#             print(f"Warning: Image and mask dimensions do not match for {name}. ")
#             print(f"Mask resize.")
#             mask_array = cv2.resize(mask_array, (image.shape[1], image.shape[0]))

#         # 根据rend2bin阈值生成二值mask
#         mask_bool = mask_array > rend2bin

#         # 可选：处理小区域噪声
#         mask_bool = process_small_region_bool(mask_bool)

#         # 保存二值图到 traced_bin_path（大于阈值的地方为白色255，其他为黑色0）
#         bin_save_path = os.path.join(traced_bin_dir, name)
#         cv2.imwrite(bin_save_path, mask_bool.astype(np.uint8) * 255)
#         print(f"Saved binary mask to {bin_save_path}")

#         original_height, original_width = image.shape[:2]
#         print(f"Original image size: {original_width}x{original_height}")
        
#         # 生成patches
#         patches_info = generate_patches(original_height, original_width, patch_size, overlap)
#         print(f"Generated {len(patches_info)} patches")
        
#         patches_results = []
        
#         # 处理每个patch
#         for i, patch_info in enumerate(patches_info):
            
#             # if i != 6:
#                 # continue

#             print(f"Processing patch {i}/{len(patches_info)}: "
#                     f"({patch_info['x_start']},{patch_info['y_start']}) to "
#                     f"({patch_info['x_end']},{patch_info['y_end']})")
            
#             image_patch = extract_patch(image, patch_info)
#             mask_patch = extract_patch(mask_array, patch_info)
            
#             patch_image_path = os.path.join(image_patch_dir, f"patch_{i:03d}_image.png")
#             patch_mask_path = os.path.join(image_patch_dir, f"patch_{i:03d}_mask.png")
#             patch_bin_path = os.path.join(image_patch_dir, f"patch_{i:03d}_bin.png")
#             patch_result_path = os.path.join(image_patch_dir, f"patch_{i:03d}_result.png")
            
#             # 保存patch的原始文件
#             cv2.imwrite(patch_image_path, cv2.cvtColor(image_patch, cv2.COLOR_RGB2BGR))
#             cv2.imwrite(patch_mask_path, mask_patch)

#             # 生成并保存patch的thresholded bin图片
#             patch_mask_bool = mask_patch > rend2bin
#             patch_mask_bool = process_small_region_bool(patch_mask_bool)
#             patch_bin_thresholded_path = os.path.join(image_patch_dir, f"patch_{i:03d}_bin_thresholded.png")
#             cv2.imwrite(patch_bin_thresholded_path, patch_mask_bool.astype(np.uint8) * 255)
#             print(f"Saved patch thresholded binary mask to {patch_bin_thresholded_path}")
            
#             # 保存patch信息到JSON文件
#             patch_info_path = os.path.join(image_patch_dir, f"patch_{i:03d}_info.json")
#             import json
#             with open(patch_info_path, 'w') as f:
#                 json.dump(patch_info, f, indent=2)
            
#             # 对patch运行EfficientSAM
#             try:
#                 patch_masks = []
#                 for GRID_SIZE in grid_size:
#                     patch_result = run_everything_ours(
#                         f"{no}_{i}", patch_image_path, efficient_sam_vits_model, 
#                         device, patch_mask_path, patch_bin_path, 
#                         rend2bin, IoU_thre, GRID_SIZE, batch_size=32
#                     )
#                     patch_masks.extend(patch_result)
                
#                 # 保存patch的结果mask
#                 if len(patch_masks) > 0:
#                     patch_h, patch_w = patch_info['height'], patch_info['width']
#                     patch_result_img = np.zeros((patch_h, patch_w, 4))
                    
#                     # 为每个mask分配不同的颜色
#                     for mask_idx, mask in enumerate(patch_masks):
#                         if isinstance(mask, torch.Tensor):
#                             mask_np = mask.cpu().numpy().astype(bool)
#                         else:
#                             mask_np = mask.astype(bool)
                        
#                         # 为每个mask生成随机颜色
#                         color_mask = np.concatenate([np.random.random(3), [0.8]])
#                         patch_result_img[mask_np] = color_mask
                    
#                     # 保存patch结果
#                     patch_result_img = (patch_result_img * 255).astype(np.uint8)
#                     cv2.imwrite(patch_result_path, patch_result_img[:, :, :3])
#                     print(f"Saved patch result to {patch_result_path}")
                
#                 patches_results.append(patch_masks)
#                 print(f"Patch {i} completed: {len(patch_masks)} masks found")
                
#             except Exception as e:
#                 print(f"Error processing patch {i+1}: {e}")
#                 patches_results.append([])
#                 empty_result = np.zeros((patch_info['height'], patch_info['width'], 3), dtype=np.uint8)
#                 cv2.imwrite(patch_result_path, empty_result)
            
#             # if i == 1:
#                 # break  # XXX 仅处理一个patch以便调试，去掉此行可处理所有patch
        
#         # 保存patch布局信息
#         layout_info = {
#             "original_size": [original_width, original_height],
#             "patch_size": patch_size,
#             "overlap": overlap,
#             "num_patches": len(patches_info),
#             "patches_info": patches_info
#         }
#         layout_path = os.path.join(image_patch_dir, "layout_info.json")
#         with open(layout_path, 'w') as f:
#             json.dump(layout_info, f, indent=2)
        
#         # 合并所有patch的结果
#         print("Merging patch results...")
#         merged_mask = merge_patches(patches_results, (original_height, original_width), patches_info, overlap)
        
#         # 创建输出图像
#         mask_only_img = np.zeros((original_height, original_width, 4))
#         if np.any(merged_mask):
#             color_mask = np.concatenate([np.random.random(3), [0.8]])
#             mask_only_img[merged_mask] = color_mask
        
#         # 保存最终结果
#         save_path = os.path.join(masks_dir, name)
#         mask_only_img = (mask_only_img * 255).astype(np.uint8)
#         cv2.imwrite(save_path, mask_only_img[:, :, :3])
#         print(f"Saved merged mask to {save_path}")
        
#         # 保存最终合并的二值mask
#         final_binary_path = os.path.join(image_patch_dir, "final_merged_binary.png")
#         cv2.imwrite(final_binary_path, merged_mask.astype(np.uint8) * 255)
        
#         print(f"All patches and results saved to: {image_patch_dir}")
            
        
#         # if name== "02865.png":
#             # print("Debugging complete, stopping after one image.")
#         break # XXX 仅处理一个图像以便调试，去掉此行可处理所有图像
    
#     print(f"Processing completed. All patches saved to: {patch_dir}")
#     print(f"Skipped {len(skip_name)} files: {skip_name}")
#     return skip_name





# if __name__ == "__main__":

#     from efficient_sam.build_efficient_sam import build_efficient_sam_vits

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     efficient_sam_vits_model = build_efficient_sam_vits().to(device)
#     # efficient_sam_vits_model = build_efficient_sam_vitt().to(device)

#     efficient_sam_vits_model.eval()
    
#     image_path = 'outputs/paslcd/001_Cantina_pre/train_origin/ours_30000/renders/02865.png'
#     mask_path = 'data/dataset_paslcd/PASLCD/Cantina/Instance_1/gt_mask/Inst_1_test_IMG_2865.png'
#     bin_path = 'data/test_imgs/bin.png'
#     save_path = 'data/test_imgs/change_mask.png'


#     all_masks = run_everything_ours(2865, image_path, efficient_sam_vits_model, device, mask_path, bin_path, 10, 0.0, 32, 32)
       

#     image_shape = all_masks[0].shape

#     # 合并所有masks (取并集)
#     print(f"Merging {len(all_masks)} masks ")

#     mask_only_img = np.zeros((image_shape[0], image_shape[1], 4))
        
#     # 为每个mask分配不同的颜色
#     for i, mask in enumerate(all_masks):
#         if isinstance(mask, torch.Tensor):
#             mask_np = mask.cpu().numpy().astype(bool)
#         else:
#             mask_np = mask.astype(bool)
        
#         # 为每个mask生成随机颜色
#         color_mask = np.concatenate([np.random.random(3), [0.8]])  # 随机RGB + 透明度
        
#         # 将当前mask的颜色叠加到图像上
#         mask_only_img[mask_np] = color_mask
        
#     # 计算合并后的mask用于统计
#     merged_mask = np.zeros((image_shape[0], image_shape[1]), dtype=bool)
    
#     # 对所有masks取并集
#     for mask in all_masks:
#         if isinstance(mask, torch.Tensor):
#             mask_np = mask.cpu().numpy().astype(bool)
#         else:
#             mask_np = mask.astype(bool)
#         merged_mask = np.logical_or(merged_mask, mask_np)


#     # 保存合并后的mask
#     mask_only_img = (mask_only_img * 255).astype(np.uint8)
#     cv2.imwrite(save_path, mask_only_img[:, :, :3])
#     print(f"Saved merged mask to {save_path}")
