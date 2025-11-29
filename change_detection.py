import os
import sys
import argparse
import zipfile
import yaml
import shutil
from PIL import Image
from tqdm import tqdm
import time
import cv2
import numpy as np

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
from scar3d.image_difference import signed_difference, compute_cosine_similarity_pixelwise, plot_cosine_similarity_heatmaps, resize_similarities
from scar3d.trace_and_validation import trace, trace_and_prune, trace_and_remove, trace_remove_post
from scar3d.segmentation import get_prompted_mask, get_prompted_mask_difference
from scar3d.fine_masks import get_masks_from_render

def load_models(weights_dir):
    """
    load efficient sam models
    """
    models = {}
    models['efficient_sam_vitt'] = build_efficient_sam_vitt()
    
    vits_zip_path = os.path.join(weights_dir, "efficient_sam_vits.pt.zip")
    vits_extracted_path = os.path.join(weights_dir, "efficient_sam_vits.pt")
    if not os.path.exists(vits_extracted_path):
        with zipfile.ZipFile(vits_zip_path, 'r') as zip_ref:
            zip_ref.extractall(weights_dir)

    models['efficient_sam_vits'] = build_efficient_sam_vits()
    return models


def prepare_model(model, weights_path, device):
    model.to(device)
    model.eval()
    return model


def load_images_from_directory(directory, preprocess, device):
    """
    load images from a directory
    returned dict: {filename: image_tensor}
    """
    images = {}
    supported_formats = ('.png', '.jpg', '.jpeg')
    filenames = os.listdir(directory)
    filenames.sort()  # Sort filenames to ensure consistent order
    for filename in filenames:
        if filename.lower().endswith(supported_formats):
            path = os.path.join(directory, filename)
            # try:
            image = Image.open(path).convert('RGB')
            image_tensor = preprocess(image).to(device)
            images[filename] = image_tensor

    height, width = image_tensor.shape[-2:]
    return images, height, width


def load_gray_images_from_directory(directory, device):
    images = {}
    supported_formats = ('.png', '.jpg', '.jpeg')
    filenames = os.listdir(directory)
    filenames.sort()  # Sort filenames to ensure consistent order
    for filename in filenames:
        if not filename.lower().endswith(supported_formats):
            continue
        path = os.path.join(directory, filename)
        img = Image.open(path).convert('L')             
        image_tensor = ToTensor()(img).to(device)            
        images[filename] = image_tensor
    height, width = image_tensor.shape[-2:]
    return images, height, width


def get_embeddings(images, model, device):
    embeddings = {}
    with torch.no_grad():
        for filename, image_tensor in tqdm(images.items(), desc="compute embeddings"):
            # add batch dimension
            image_tensor = image_tensor.unsqueeze(0).to(device)  
            image_embedding = model.get_image_embeddings(image_tensor) # [1, 256, 64, 64]
            # remove batch dimension
            embedding = image_embedding.squeeze(0)  # [256, 64, 64]
            embeddings[filename] = embedding
    return embeddings


def get_difference_map(similarities, save_path, threshold):
    os.makedirs(save_path, exist_ok=True)
    for fname, cos_sim_map in similarities.items():
        diff_map = cos_sim_map < threshold  
        
        diff_map = diff_map.astype(np.uint8) 
        
        diff_map_image = Image.fromarray(diff_map * 255) 
        diff_map_image = diff_map_image.convert('L') 
        
        diff_map_image.save(f"{save_path}/{fname}")
        print(f"保存差异图: {save_path}/{fname}")


def compute_2d_difference(args, pre_render_dir, post_gt_dir, heatmap_dir, difmap_dir, threshold, if_dynamic_threshold=False, if_compute_cossim=True):
    """
    Loads the model, computes embeddings, and generates difference maps.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    models = load_models(args.weights_dir)
    if args.model not in models: raise ValueError
    model = prepare_model(models[args.model], os.path.join(args.weights_dir, f"{args.model}.pt"), device)
    print(f"loaded: {args.model}")
    
    preprocess = transforms.Compose([
        # transforms.Resize((256, 256)),  
        transforms.ToTensor(),
    ])
    
    images_pre, height, width = load_images_from_directory(pre_render_dir, preprocess, device)
    print(f"Loaded {len(images_pre)} images from {args.pre_path}")

    images_post, _, _ = load_images_from_directory(post_gt_dir, preprocess, device)
    print(f"Loaded {len(images_post)} images from {args.post_path}")

    print(f"Image size: {height} x {width}")

    embeddings_pre = get_embeddings(images_pre, model, device)
    embeddings_post = get_embeddings(images_post, model, device)
    
    if if_compute_cossim:
        print("Computing cosine similarity...")
        similarities = compute_cosine_similarity_pixelwise(embeddings_pre, embeddings_post)
        similarities = resize_similarities(similarities, height, width)

        plot_cosine_similarity_heatmaps(similarities, heatmap_dir)

        # TODO: dynamic threshold
        if if_dynamic_threshold:
            pass

        get_difference_map(similarities, difmap_dir, threshold)
        
        return images_pre, images_post, embeddings_pre, embeddings_post, similarities
    
    else:
        return images_pre, images_post, embeddings_pre, embeddings_post
    

def get_rank():
    # SLURM_PROCID can be set even if SLURM is not managing the multiprocessing,
    # therefore LOCAL_RANK needs to be checked first
    rank_keys = ("RANK", "LOCAL_RANK", "SLURM_PROCID", "JSM_NAMESPACE_RANK")
    for key in rank_keys:
        rank = os.environ.get(key)
        if rank is not None:
            return int(rank)
    return 0

def get_device():
    return torch.device(f"cuda:{get_rank()}")

def load_binary_mask(mask_path, device, width=1600, height=1200):
    mask_image = Image.open(mask_path).convert('L')  # 转为灰度图
    mask_image = mask_image.resize((width, height))
    print("mask_image resize: ", mask_image.size)
    mask_np = np.array(mask_image)
    id_mask = (mask_np > 120).astype(np.int32)
    id_mask_tensor = torch.from_numpy(id_mask).to(torch.int).to(device)

    # mask_image_output_path = os.path.splitext(mask_path)[0] + "_output.png"
    # mask_image_output = Image.fromarray((id_mask_tensor.cpu().numpy() * 255).astype(np.uint8))
    # mask_image_output.save(mask_image_output_path)
    # print(f"Mask image saved to {mask_image_output_path}")

    return id_mask_tensor

@torch.no_grad()
def update_mask(self, save_name="mask") -> None:
    print(f"Segment with prompt: {self.cfg.seg_prompt}")
    mask_cache_dir = os.path.join(
        self.cache_dir, self.cfg.seg_prompt + f"_{save_name}_{self.view_num}_view"
    )
    gs_mask_path = os.path.join(mask_cache_dir, "gs_mask.pt")
    if not os.path.exists(gs_mask_path) or self.cfg.cache_overwrite:
        if os.path.exists(mask_cache_dir):
            shutil.rmtree(mask_cache_dir)
        os.makedirs(mask_cache_dir)
        weights = torch.zeros_like(self.gaussian._opacity)
        weights_cnt = torch.zeros_like(self.gaussian._opacity, dtype=torch.int32)
        # threestudio.info(f"Segmentation with prompt: {self.cfg.seg_prompt}")
        for id in tqdm(self.view_list):
            cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
            cur_path_viz = os.path.join(
                mask_cache_dir, "viz_{:0>4d}.png".format(id)
            )

            cur_cam = self.trainer.datamodule.train_dataset.scene.cameras[id]

            mask = self.text_segmentor(self.origin_frames[id], self.cfg.seg_prompt)[
                0
            ].to(get_device())

            mask_to_save = (
                    mask[0]
                    .cpu()
                    .detach()[..., None]
                    .repeat(1, 1, 3)
                    .numpy()
                    .clip(0.0, 1.0)
                    * 255.0
            ).astype(np.uint8)
            cv2.imwrite(cur_path, mask_to_save)

            masked_image = self.origin_frames[id].detach().clone()[0]
            masked_image[mask[0].bool()] *= 0.3
            masked_image_to_save = (
                    masked_image.cpu().detach().numpy().clip(0.0, 1.0) * 255.0
            ).astype(np.uint8)
            masked_image_to_save = cv2.cvtColor(
                masked_image_to_save, cv2.COLOR_RGB2BGR
            )
            cv2.imwrite(cur_path_viz, masked_image_to_save)
            self.gaussian.apply_weights(cur_cam, weights, weights_cnt, mask) # HERE

        weights /= weights_cnt + 1e-7

        selected_mask = weights > self.cfg.mask_thres
        selected_mask = selected_mask[:, 0]
        torch.save(selected_mask, gs_mask_path)
    else:
        print("load cache")
        for id in tqdm(self.view_list):
            cur_path = os.path.join(mask_cache_dir, "{:0>4d}.png".format(id))
            cur_mask = cv2.imread(cur_path)
            cur_mask = torch.tensor(
                cur_mask / 255, device="cuda", dtype=torch.float32
            )[..., 0][None]
        selected_mask = torch.load(gs_mask_path)

    self.gaussian.set_mask(selected_mask)
    self.gaussian.apply_grad_mask(selected_mask)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="change detection 3d")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)

    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument('--mask_root', type=str, required=True, help='root dir to save masks, for one scene, e.g. /data/denoise_dif_map/blender_100_3d/post-001')
    parser.add_argument('--config', type=str, required=True, help='config file path')


    parser.add_argument('--weights_dir', type=str, default='weights', help='efficient sam weights dir')
    parser.add_argument('--model', type=str, default='efficient_sam_vits', help='efficient sam model name')

    parser.add_argument('--get_signed_difference', action='store_true', help='get 2d difference map')
    parser.add_argument('--get_trace', action='store_true', help='get trace')
    # parser.add_argument('--get_render', action='store_true', help='get render')
    parser.add_argument('--get_prompt', action='store_true', help='get prompt for efficient sam')
    parser.add_argument('--get_recon', action='store_true', help='reconstruction')
    # parser.add_argument('--get_related_efficientsam_mask', action='store_true', help='get related efficient sam mask')
    parser.add_argument('--get_pre', action='store_true', help='if get change mask under pre camera view')
    parser.add_argument('--get_post', action='store_true', help='if get change mask under post camera view')
    parser.add_argument('--get_test_pre', action='store_true', help='if get change mask under test-pre camera view')
    parser.add_argument('--get_test_post', action='store_true', help='if get change mask under test-post camera view')
    
    parser.add_argument('--view_pre', action='store_true', help='if get change mask at pre state under arbitrary view') 
    parser.add_argument('--patch_img', action='store_true')
    parser.add_argument('--removal', action='store_true', help='if removal the changed object')
    parser.add_argument('--zero_fill', action='store_true', help='if zero fill the image name')
    parser.add_argument('--prune_change_region', action='store_true', help='if prune the change region when reconstruction')

    args = parser.parse_args(sys.argv[1:])


    output_dir = lp.extract(args).model_path
    pre_render_dir = os.path.join(output_dir, 'pre_origin/ours_30000/renders')
    post_gt_dir = os.path.join(output_dir, 'pre_origin/ours_30000/gt')
    heatmap_dir = os.path.join(args.mask_root, 'heatmap')
    difmap_dir = os.path.join(args.mask_root, 'difmap')
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(difmap_dir, exist_ok=True)

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    imgnames = os.listdir(pre_render_dir)
    imgpath = os.path.join(pre_render_dir, imgnames[0])
    img = Image.open(imgpath)
    img_w = img.size[0]
    img_h = img.size[1]
    print("Image width: ", img_w)
    print("Image height: ", img_h)

    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        difference_map_threshold = config['mask']['difference_map_threshold']

    dif_3d_ply_left = os.path.join(args.mask_root, 'difmap_3d_left.ply')
    dif_3d_ply_right = os.path.join(args.mask_root, 'difmap_3d_right.ply')
    dif_left = os.path.join(difmap_dir, "left")
    dif_right = os.path.join(difmap_dir, "right")

    seqs = []
    if args.get_post:
        seq = 'post'
    if args.get_pre:
        seq = 'pre'
    if args.get_test_post:
        seq = 'test-post'
    if args.get_test_pre:
        seq = 'test-pre'


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"device: {device}")

    # Load all models
    print("loading model...")
    efficientsam_models = load_models(args.weights_dir)
    
    # Prepare specified model
    model = prepare_model(efficientsam_models[args.model], os.path.join(args.weights_dir, f"{args.model}.pt"), device)
    print(f"Loaded model: {args.model}")


    if args.get_signed_difference:
        signed_difference(pre_render_dir, post_gt_dir, args.config, model, heatmap_dir, difmap_dir) 


    # TODO Optimize：Reload gaussian model and save color are time-consuming
    if args.get_trace:
        color_left, left_remain_rate = trace_and_prune(config, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), iteration=args.iteration, mask_folder=dif_left, output_ply=dif_3d_ply_left, threshold=0.1, width=img_w, height=img_h, zero_fill=args.zero_fill)
        color_right, right_remain_rate = trace_and_prune(config, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args), iteration=args.iteration, mask_folder=dif_right, output_ply=dif_3d_ply_right, threshold=0.1, width=img_w, height=img_h, zero_fill=args.zero_fill)

        color_path_left = dif_3d_ply_left.replace(".ply", "_weight_color.pt")
        color_path_right = dif_3d_ply_right.replace(".ply", "_weight_color.pt")

        # TODO
        # mask_threshold = 0.1 # good for blender dataset
        # mask_threshold = 0.3
        # area_threshold = 0.3

        masks = {}

        args.change_state = seq 
        render_sets(f"{seq}_left",lp.extract(args), args.iteration, pp.extract(args), skip_train=False, skip_test=True, color=color_left, separate_sh=False)
        render_sets(f"{seq}_right",lp.extract(args), args.iteration, pp.extract(args), skip_train=False, skip_test=True, color=color_right, separate_sh=False)
        
    iteration = args.iteration
    render_traced_left = os.path.join(output_dir, f'{seq}_left/ours_{iteration}/renders')
    render_traced_right = os.path.join(output_dir, f'{seq}_right/ours_{iteration}/renders')

    pre_seq = 'right'
    post_seq = 'left'

    if seq == 'post' or seq == 'pre':
        image_dir = os.path.join(args.source_path, f'train-{seq}')
    elif seq == 'test-post' or seq == 'test-pre':
        image_dir = os.path.join(args.source_path, seq)
    else:
        raise ValueError("Invalid sequence for post state")

    traced_bin_dir = os.path.join(args.mask_root, f'{seq}_traced') 
    masks_dir = os.path.join(args.mask_root, f"{seq}_change_masks")


    if args.get_prompt:
        get_prompted_mask(render_traced_left, args.config, image_dir, masks_dir, traced_bin_dir, args.view_pre, no_img_prefix=False, zero_fill=args.zero_fill)






