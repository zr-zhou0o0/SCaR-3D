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
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import ToTensor
from torchvision.ops.boxes import batched_nms, box_area

from scene import Scene, GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams
from render import render_sets
from scar3d.merge_masks import merge_masks


from efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from segment_anything.utils.amg import (
    batched_mask_to_box,
    calculate_stability_score,
    mask_to_rle_pytorch,
    remove_small_regions,
    rle_to_mask,
)
from scar3d.segmentation import process_small_region, show_anns_ours

from scar3d.image_difference import signed_difference
from scar3d.trace_and_validation import trace
from scar3d.trace_and_validation import trace_and_prune 
from scar3d.segmentation import get_prompted_mask


def get_predictions_given_embeddings_and_queries(img, points, point_labels, model, device):

    # print("shape of img", img.shape) # (3, 1200, 1600)
    # print("shape of points", points.shape) # (1, 1024, 1, 2), where 1024 is the number of points
    # print("shape of point_labels", point_labels.shape) # (1, 1024, 1)

    predicted_masks, predicted_iou = model(img[None, ...].to(device), points.to(device), point_labels.to(device))
    # add a new dimension, like, (1, 3, 1200, 1600)

    # print("shape of predicted_iou", predicted_iou.shape) # (1, 1024, 3), where 3 is 3 candidate masks' iou score
    # print("shape of predicted_masks", predicted_masks.shape) # (1, 1024, 3, 1200, 1600)

    # this step is to get the max iou in 3 candidate for each point! so is still 1024
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True) # sorted ids, is index, like 2,0,1
    # then take the values from the index
    predicted_iou_scores = torch.take_along_dim(predicted_iou, sorted_ids, dim=2) # (1, 1024, 3)
    predicted_masks = torch.take_along_dim(predicted_masks, sorted_ids[..., None, None].expand_as(predicted_masks), dim=2) # (1, 1024, 3, 1200, 1600)
    # the none none is to match the dimension of the mask

    # because here we only process one image... so eliminate the first dimension
    predicted_masks = predicted_masks[0] # (1024, 3, 1200, 1600)

    iou = predicted_iou_scores[0, :, 0] # get the max iou score for each point, shape is (1024,)
    index_iou = iou > 0.7 # only keep the points with iou > 0.7
    iou_ = iou[index_iou] # (n,)
    masks = predicted_masks[index_iou] # (n, 3, 1200, 1600)

    score = calculate_stability_score(masks, 0.0, 1.0)
    score = score[:, 0]
    index = score > 0.9
    score_ = score[index]
    masks = masks[index]
    iou_ = iou_[index]

    masks = torch.ge(masks, 0.0) # to make it 0 or 1
    # print("shape of masks", masks.shape) # (n, 3, 1200, 1600), n is the number of points, here is 963(of 1024 primary points)
    # print("shape of iou_", iou_.shape) # (963,)

    return masks, iou_



def get_predictions_with_batching(img, points, point_labels, model, device, batch_size=100):
    """
    If the number of query points exceeds batch_size, split them into multiple batches,
    query the model separately and merge the results.

    Args:
        img (Tensor): Input image tensor.
        points (Tensor): Query points tensor with shape (1, N, 1, 2).
        point_labels (Tensor): Query point labels tensor with shape (1, N, 1).
        model (nn.Module): Model for prediction.
        device (torch.device): Computation device.
        batch_size (int): Maximum number of query points per batch.

    Returns:
        Tuple[Tensor, Tensor]: Merged mask tensor and IoU scores tensor.
    """
    num_pts = points.shape[1]
    if num_pts <= batch_size:
        return get_predictions_given_embeddings_and_queries(img, points, point_labels, model, device)
    
    # Initialize lists to store masks and IoU scores for each batch
    masks_list = []
    iou_list = []
    
    for start in range(0, num_pts, batch_size):
        end = start + batch_size
        batch_points = points[:, start:end, :, :]
        batch_labels = point_labels[:, start:end, :]
        
        masks, iou = get_predictions_given_embeddings_and_queries(img, batch_points, batch_labels, model, device)
        
        # To check if the batch returned any masks
        if masks.numel() > 0:
            masks_list.append(masks)
            iou_list.append(iou)

    if len(masks_list) == 0:
        # return empty tensors if no masks were found, ensure the output shape is consistent
        return torch.empty(0, device=device), torch.empty(0, device=device)
    
    # concatenate all masks and IoU scores
    combined_masks = torch.cat(masks_list, dim=0)
    combined_iou = torch.cat(iou_list, dim=0)
    
    return combined_masks, combined_iou


def get_thresholded_masks(predicted_masks, gray_mask, gray_threshold=50, area_threshold=0.5):
    """
    Filter predicted masks by overlap with a reference gray mask.
    
    Args:
        predicted_masks: List of masks, each a 2D array or tensor of shape (H,W), bool or {0,1}.
        gray_mask:       2D or 3D tensor of grayscale values (0–255) or float, shape (H,W) or (1,H,W).
        gray_threshold:  int, threshold to binarize gray_mask (> threshold => 1).
        area_threshold:  float in (0,1), minimum intersection/predicted_area to keep.
    
    Returns:
        dict[int, dict]: key = mask_id (0-based), value = {
            "mask": mask_tensor,        # BoolTensor shape (H,W)
            "precision": float          # intersection / predicted_area
        }
    """
    # prepare reference binary mask
    if gray_mask.dim() == 3 and gray_mask.shape[0] == 1:
        ref = gray_mask[0]
    else:
        ref = gray_mask
    ref_bin = (ref > gray_threshold).to(torch.bool)

    out = {}
    device = ref_bin.device
    for idx, m in enumerate(predicted_masks):
        # convert predicted mask to bool tensor on same device
        pm = torch.as_tensor(m, device=device)
        if pm.dim() == 3 and pm.shape[0] == 1:
            pm = pm[0]
        pm_bin = pm.bool()

        # compute areas
        pred_area = pm_bin.sum().item()
        if pred_area == 0:
            continue

        inter = (pm_bin & ref_bin).sum().item()
        precision = inter / pred_area
        print("precision: ", precision)

        if precision >= area_threshold:
            out[idx] = {
                "mask": pm_bin,
                "precision": precision
            }
    return out


# 本来这里要更换目录的，但是既然之前已经加载过了，那么理论上不需要
# 不过其实还是要单独指定train_1 好消息是train_1不会被挪走
def get_masks_from_render(render_dir, threshold, area_threshold, config_path, images_dir, masks_dir, save_binary_dir, original_image_w=1600, original_image_h=1200, batchsize=50):
    
    # 在不同视角下，同一个物体的feature有可能是不稳定的！这就是那篇文章做的工作。可以做个矩阵试试看，在filter之后算A视角和B视角下两组物体对应上的feat cos sim
    # 那是不是说可以用gaussian监督efficientsam来让它保持视角一致性？感觉大概率训不出来。
    # 如果这就是最后一步了，那就可以直接testset找了，但是分数恐怕不高。。。但是不用train post！
    # 如果再trace再找一次，时间会变长，而且就需要train post了。那这就是coarse2fine了。

    '''
    Input: render dir, with gray maps naming in uid

    1. for each gray map, if it is empty, skip it. if not, get a enabled grid whose point in difmap has value > threshold. save the query point list.
    2. for each group of query point, run everything ours and get the mask list.
    4. compute mask list and dif's IoU and get the best mask for each dif. (really? maybe i want to get the best 2 masks?)
    5. save the best 300 pre_masks and post_masks.

    Output: dictionary, key=uid_maskid, value=dictonary
    {
        00301：{
            01:{
                "mask": mask_tensor,
                "feature": average feature_tensor,
                "precision": , # intersection over mask region
            }
            02:{
                "mask": mask_tensor,
                "feature": average feature_tensor,
                "precision": , # intersection over mask region
            }
            ...
        }
        00302：{
            ...
        }
        ...
    }
    
    '''
    import time
    start_time = time.time()
    print("start_time: ", start_time)

    # because the new mask will add to the old mask instead of replace it
    if os.path.exists(masks_dir):
        # delete the directory
        shutil.rmtree(masks_dir)
        # raise ValueError(f"masks_dir {masks_dir} already exists, please delete it first")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    print(f"Using device: {device}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
        GRID_SIZE = config["mask"]["grid_size"]
        global min_area
        min_area = config["mask"]["area_threshold"]
        print("min_area: ", min_area)

    # efficient_sam_vitt_model = build_efficient_sam_vitt()
    efficient_sam_vitt_model = build_efficient_sam_vits()
    efficient_sam_vitt_model.eval()
    model = efficient_sam_vitt_model.to(device)

    xy = []
    for i in range(GRID_SIZE):
        curr_x = 0.5 + i / GRID_SIZE * original_image_w
        for j in range(GRID_SIZE):
            curr_y = 0.5 + j / GRID_SIZE * original_image_h
            xy.append([curr_x, curr_y])
    xy = torch.from_numpy(np.array(xy)).to(device)  # 将 xy 移动到 GPU
    print("xy.shape: ", xy.shape) # (1024, 2) if GRID_SIZE = 32
    len_xy = xy.shape[0]

    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # masks,_,_ = load_gray_images_from_directory(render_dir, device=device)
    masks_name = os.listdir(render_dir)
    masks_name.sort()
    
    skip_name = []

    os.makedirs(masks_dir, exist_ok=True)
    os.makedirs(save_binary_dir, exist_ok=True)
    
    masks_pre = {}

    def bytes_to_mb(bytes):
        return bytes / 1024**2
    print(f"[Before] Current GPU Memory: {bytes_to_mb(torch.cuda.memory_allocated())} MB") # 39.24MB

    OOM_counter = 0

    # for idx, name in enumerate(masks_name):

    # Tackle the OOM problem by using a queue
    # 初始化队列，每个元素为(start_id, name)元组
    from collections import deque
    mask_queue = deque([(0, name) for name in masks_name])

    while mask_queue:

        if OOM_counter > 10:
            print("OOM too many times, consider to change the grid size")
            break

        start_id, name = mask_queue.popleft()
        idx = masks_name.index(name)  # 获取原始索引

        mask = Image.open(os.path.join(render_dir, name)).convert('L')
        mask = ToTensor()(mask).to(device)              # shape [1, H, W]
        mask = mask.squeeze(0)  # (1200, 1600)
        # convert mask to binary, bool type
        mask = (mask > threshold).to(torch.bool) # (1200, 1600)
        # convert mask to binary image and save:
        mask_bin = mask.cpu().numpy()  # convert to numpy array
        mask_bin = (mask_bin * 255).astype(np.uint8)  # convert to uint8
        mask_bin = Image.fromarray(mask_bin)  # convert back to PIL image
        mask_bin.save(os.path.join(save_binary_dir, name))  # save the mask image

        print(f"Index: {idx}, Name: {name}, Tensor shape: {mask.shape}")

        print("maximum value in mask: ", mask.max())

        # TODO adaptive grid density for 40-50 points
        query_points = []
        for k in range(len_xy):
            x, y = xy[k].cpu().numpy()  
            if mask[int(y), int(x)]:
                query_points.append([x, y])
        if query_points == []:
            print(f"Skip {name} for empty query_points") # here it is, throw away the empty mask
            continue

        points = torch.from_numpy(np.array(query_points)).to(device)
        num_pts = points.shape[0]
        # print("query_points total len: ", num_pts) # (n, 2), n is the number of points in the dif map
        # point_labels = torch.ones(points.shape[0], 1).to(device) # (n, 1)

        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 10))
        plt.title(f"Query Points for {name}")
        plt.scatter(points[:, 0].cpu().numpy(), points[:, 1].cpu().numpy(), s=1)
        plt.savefig(f"data/test_imgs/points_{name}.png")


        end_id = min(start_id + batchsize, num_pts)
        current_points = points[start_id:end_id]
        actual_num_pts = current_points.shape[0]
        actual_point_labels = torch.ones(actual_num_pts, 1).to(device) # (n, 1)

        if end_id < num_pts:
            new_start = end_id
            mask_queue.appendleft((new_start, name))  # 插入队列前端优先处理
            print(f"Queued next batch: {name} from {new_start}")
        else:
            print(f"Finished processing {name}")

        
        print(f"======== Processing {name} [{start_id}:{end_id}]/{num_pts} points ========")
    
        try:
            image_name = "whole_" + str(int(name.split('.')[0])) + ".png"
            image_path = os.path.join(images_dir, image_name)
            image = Image.open(image_path).convert('RGB')

        except FileNotFoundError as e:
            print(f"Error: {e}, Skip {name}")
            continue

        if isinstance(image, torch.Tensor):
            img_tensor = image.to(device)
        else:
            img_tensor = ToTensor()(image).to(device)


        # label is 1 = positive points, 0 = negative points
        # all 1 will be used to generate masks, 0 will be used to generate background

        # print("pre_img_tensor shape: ", img_tensor.shape) # (3, 1200, 1600)

        print("[before] Current GPU Memory: ", bytes_to_mb(torch.cuda.memory_allocated()), "MB") # 68.57MB
        # 确实是50，而且在返回之前，即使分batch也没有用
        try:
            with torch.no_grad():
                predicted_masks, predicted_iou = get_predictions_with_batching(
                    img_tensor,
                    current_points.reshape(1, actual_num_pts, 1, 2),
                    actual_point_labels.reshape(1, actual_num_pts, 1),
                    model,
                    device,
                    batch_size=batchsize # 50 #100 oom  
                )
        except Exception as e:
            print(f"Error: {e}, Skip {name} pre")
            skip_name.append(name)
            OOM_counter += 1
            continue
            

        if predicted_masks == []:
            continue
        rle = [mask_to_rle_pytorch(m[0:1].to('cpu')) for m in predicted_masks] 
        # predicted mask: (963, 3, 1200, 1600)
        # m: (3, 1200, 1600)
        # m[0:1]: (1, 1200, 1600) the first mask
        # rle: compressed mask
        # print("rle shape: ", len(rle)) 
        if rle == []:
            continue
        predicted_masks = process_small_region(rle, device) 
        # print("predicted_masks: ", pre_predicted_masks) # list of bool matrix, list of masks
        print("len(predicted_masks): ", len(predicted_masks)) # len = 28
        # print("predicted_masks[0].shape: ", predicted_masks[0].shape) # (1200, 1600)
        if predicted_masks == []:
            continue
        pred_masks = predicted_masks


        # # TODO dynamic threshold
        # 0.4 is good for pre mask in post-003
        # 0.3 is good for post mask in post-003
        # area_threshold = 0.3

        pred_masks = get_thresholded_masks(predicted_masks, mask, threshold, area_threshold)
        if pred_masks == {}:
            print(f"Skip {name} for empty area thresholded masks")

        masks_pre[name] = pred_masks


        mask_only_img_pre = np.zeros((mask.shape[0], mask.shape[1], 4))  # RGBA
        mask_only_img_pre[:,:,3] = 1  # set the alpha channel to 1, not transparent

        # pre masks is dict!!!
        # to traverse the dict, use items()
        for k, v in pred_masks.items():
            ann = v["mask"]
            m = ann.cpu().numpy()
            print("ann true size: ", m.sum().item())
            # color_mask = [1,1,1,1]  # RGBA
            # color_mask_pre = list(np.random.random(3)) + [1.0]
            rgb = np.random.randint(100, 256, size=3)
            color_mask_pre = list(rgb / 255.0) + [1.0]
            mask_only_img_pre[m] = color_mask_pre  
            # concat_mask_pre[m] = 1

        # 如果同名图片不存在
        if not os.path.exists(os.path.join(masks_dir, name)):
            plt.imsave(os.path.join(masks_dir, name), mask_only_img_pre)
        
        else:
            # 读取图片，在原有mask的基础上继续添加mask
            # TODO
            existing_img = plt.imread(os.path.join(masks_dir, name))
            # Make sure the existing image has an alpha channel
            if existing_img.shape[2] == 3:
                # Convert RGB to RGBA by adding alpha channel
                alpha_channel = np.ones((existing_img.shape[0], existing_img.shape[1], 1))
                existing_img = np.concatenate([existing_img, alpha_channel], axis=2)
            
            valid_area = np.any(mask_only_img_pre[..., :3] != 0, axis=-1)
            existing_img[valid_area] = mask_only_img_pre[valid_area]
            
            # Save the updated image
            plt.imsave(os.path.join(masks_dir, name), existing_img)


    print("pre_skip: ", skip_name)

    end_time = time.time()
    print("end_time: ", end_time)
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")
   
    return masks_pre

