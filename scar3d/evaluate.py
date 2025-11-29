import os
import json 
import numpy as np
from PIL import Image
from sklearn.metrics import precision_score, recall_score, f1_score, jaccard_score
import cv2

def load_image(filepath, threshold=20, save_path=None):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    with Image.open(filepath) as img:
        img = img.convert('L')  # 转换为灰度图
        img_np = np.array(img)
        binary = (img_np > threshold).astype(np.uint8)
    
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        Image.fromarray(binary * 255).save(save_path)

    return binary

def compute_metrics(mask, gt):

    print("sum gt: ", np.sum(gt))
    print("gt size: ", gt.size)
    print("sum gt / gt.size: ", np.sum(gt) / gt.size)
    
    if np.sum(gt) == 0:
        return None
    
    if np.sum(gt) / gt.size < 0.0001:
        return None
    

    mask_flat = mask.flatten()
    gt_flat = gt.flatten()
    
    precision = precision_score(gt_flat, mask_flat, zero_division=0)
    recall = recall_score(gt_flat, mask_flat, zero_division=0)
    f1 = f1_score(gt_flat, mask_flat, zero_division=0)
    iou = jaccard_score(gt_flat, mask_flat, zero_division=0)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'iou': iou
    }

def main(mask_dir, gt_dir, output_json_path,save_path, threshold):  
   
    try: 
        mask_files = sorted([f for f in os.listdir(mask_dir) if (os.path.isfile(os.path.join(mask_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.JPG')))])
        gt_files = sorted([f for f in os.listdir(gt_dir) if (os.path.isfile(os.path.join(gt_dir, f)) and f.endswith(('.png', '.jpg', '.jpeg', '.JPG')))])
    except:
        print("No files found in the specified directories.")
        print("mask_dir:", mask_dir)
        print("gt_dir:", gt_dir)
        return
    
    
    num_images = len(mask_files)
    metrics_list = []
    
    results = {  # 新增: 创建一个字典来存储所有结果
        'per_image_metrics': [],
        'overall_metrics': {}
    }

    if save_path is not None:
        mask_save_dir = os.path.join(save_path, 'masks')
        os.makedirs(mask_save_dir, exist_ok=True)

        gt_save_dir = os.path.join(save_path, 'gt')
        os.makedirs(gt_save_dir, exist_ok=True)
    
    # for i in range(num_images):
    # for mask in mask_files:
    for gt in gt_files:
        uid = int(gt.split('.')[0].split('_')[-1])  # 文件名格式为 "gt_001.png"
        img_name = str(uid).zfill(5) + ".png"  # 生成对应的图像名称

        # uid = int(mask.split('.')[0])
        # # img_name = "whole_" + str(uid) + ".png"
        # img_name = "gt_" + str(uid).zfill(3) + ".png"

        # mask_path = os.path.join(mask_dir, mask)
        # gt_path = os.path.join(gt_dir, img_name)
        mask_path = os.path.join(mask_dir, img_name)
        gt_path = os.path.join(gt_dir, gt)


        if save_path is not None:
            mask_save_path = os.path.join(mask_save_dir, img_name)
            gt_save_path = os.path.join(gt_save_dir, img_name)
        
        # 加载和二值化图像
        mask = load_image(mask_path, threshold, mask_save_path if save_path is not None else None)
        gt = load_image(gt_path, threshold, gt_save_path if save_path is not None else None)

        # if mask is None or gt is None:
        #     print(f"Image {uid}: Mask or GT is None. Skip.\n")
        #     continue
        # # 如果mask 或者 gt 为纯黑色图像，跳过
        # if np.sum(mask) == 0 or np.sum(gt) == 0:
        #     print(f"Image {uid}: Mask or GT is empty. Skip.\n")
        #     continue
        # 如果gt 为纯黑色图像，跳过
        if np.sum(gt) == 0:
            print(f"Image {uid}: GT is empty. Skip.\n")
            continue
        if mask is None: # 所有指标都是0
            metrics = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'iou': 0.0
            }
            metrics_list.append(metrics)
        else:

            if mask.shape != gt.shape:
                target_height = max(mask.shape[0], gt.shape[0])
                target_width = max(mask.shape[1], gt.shape[1])

                mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                gt = cv2.resize(gt, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
                print(f"Resized mask and GT to ({target_width}, {target_height})")

            
            # 计算指标
            metrics = compute_metrics(mask, gt)
            metrics_list.append(metrics)
        
        print(f"Image {uid}:")
        # print(f"  Mask file: {mask_files[i]}")
        # print(f"  GT file:   {gt_files[i]}")
        print(f"  Precision: {metrics['precision']*100:.2f}%")
        print(f"  Recall:    {metrics['recall']*100:.2f}%")
        print(f"  F1-Score:  {metrics['f1']*100:.2f}%")
        print(f"  IoU:       {metrics['iou']*100:.2f}%\n")
        
        results['per_image_metrics'].append({
            'image_uid': uid,
            # 'mask_file': mask_files[i],
            # 'gt_file': gt_files[i],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'iou': metrics['iou']
        })
    
    precision_vals = np.array([m['precision'] for m in metrics_list])
    recall_vals = np.array([m['recall'] for m in metrics_list])
    f1_vals = np.array([m['f1'] for m in metrics_list])
    iou_vals = np.array([m['iou'] for m in metrics_list])
    
    metrics_mean = {
        'precision_mean': np.mean(precision_vals),
        'precision_std': np.std(precision_vals, ddof=1),
        'recall_mean': np.mean(recall_vals),
        'recall_std': np.std(recall_vals, ddof=1),
        'f1_mean': np.mean(f1_vals),
        'f1_std': np.std(f1_vals, ddof=1),
        'iou_mean': np.mean(iou_vals),
        'iou_std': np.std(iou_vals, ddof=1)
    }
    
    print("=== Result ===")
    print(f"Precision: {metrics_mean['precision_mean']*100:.2f}% ± {metrics_mean['precision_std']*100:.2f}%")
    print(f"Recall:    {metrics_mean['recall_mean']*100:.2f}% ± {metrics_mean['recall_std']*100:.2f}%")
    print(f"F1-Score:  {metrics_mean['f1_mean']*100:.2f}% ± {metrics_mean['f1_std']*100:.2f}%")
    print(f"IoU:       {metrics_mean['iou_mean']*100:.2f}% ± {metrics_mean['iou_std']*100:.2f}%")
    
    results['overall_metrics'] = metrics_mean

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    print(f"\nResult saved: {output_json_path}")

if __name__ == "__main__":
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description='Evaluate the segmentation results.')
    parser.add_argument("--mask_folder", type=str, required=True, help="Path to the folder containing the predicted masks.")
    parser.add_argument("--gt_folder", type=str, required=True, help="Path to the folder containing the ground truth masks.")
    parser.add_argument("--result_folder", type=str, default=None, help="Path to save the binarized images.")
    # parser.add_argument("--threshold", type=int, default=20, help="Threshold value for binarization.")
    parser.add_argument("--config", type=str, help="Path to the configuration file.")
    parser.add_argument("--seq", type=str, help="sequence")

    args = parser.parse_args()

    mask_folder = args.mask_folder
    gt_folder = args.gt_folder
    # threshold = args.threshold
    seq = args.seq

    os.makedirs(args.result_folder, exist_ok=True)

    output_json = os.path.join(args.result_folder, 'results.json')

    if args.config is not None:
        with open(args.config, 'r') as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
            assert seq == 'pre' or seq == 'post'
            try: 
                if seq == 'pre':
                    threshold = config["evaluation"]["mask_threshold_pre"]
                else:
                    threshold = config["evaluation"]["mask_threshold_post"]
            except:
                print("not specified sequence threshold")
                threshold = config["evaluation"]["mask_threshold"]

            print(f"Config: Threshold value set to {threshold}")


    main(mask_folder, gt_folder, output_json, args.result_folder, threshold)
