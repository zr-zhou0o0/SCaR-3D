# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
# For inquiries contact  george.drettakis@inria.fr

'''
Train Original
'''

import os
import sys
import torch
import cv2
import uuid
from tqdm import tqdm
from random import randint
from argparse import ArgumentParser, Namespace

from utils.general_utils import safe_state, get_expon_lr_func
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim, create_gaussian_weights, create_bilinear_weights
# from gaussian_renderer import network_gui
from gaussian_renderer import render_point as render
from gaussian_renderer import render_mean2D
from scene import Scene
from scene.gaussian_model import GaussianModel
from arguments import ModelParams, PipelineParams, OptimizationParams

TENSORBOARD_FOUND = False
SPARSE_ADAM_AVAILABLE = False
from fused_ssim import fused_ssim
FUSED_SSIM_AVAILABLE = True

# Quick Setup for Debugging
# USE_WEIGHTED_CHANGE_MASK = False
# DEPTH_LOSS = False

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, start_checkpoint, debug_from, change_mask):

    # Initialization
    prepare_output_and_logger(dataset)

    first_iter = 0
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)

    print("start checkpoint: ", start_checkpoint)
    scene = Scene(dataset, gaussians, load_iteration=start_checkpoint) 
    # 因为原版用的Gaussianmodel和incre不同

    gaussians.training_setup(opt)
    gaussians.initialize_state_mask() 

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Create cuda events for timing
    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations) # return a function???

    all_train_cameras = scene.getTrainCameras()
    # viewpoint_indices = list(range(len(viewpoint_stack))) # a key list for viewpoint_stack
    
    change_state = dataset.change_state
    if change_state == 'mix':
        ratio = dataset.ratio
        split = dataset.split_pre_post # included in 'post'
        pre_cameras = [cam for cam in all_train_cameras 
                      if int(cam.image_name.split('.')[0].split('_')[1]) < split]
        post_cameras = [cam for cam in all_train_cameras 
                       if int(cam.image_name.split('.')[0].split('_')[1]) >= split]
        print(f"Total cameras: {len(all_train_cameras)}")
        print(f"Pre-change cameras: {len(pre_cameras)}")
        print(f"Post-change cameras: {len(post_cameras)}")
        
        pre_stack = pre_cameras.copy()
        post_stack = post_cameras.copy()
    
    else:
        viewpoint_stack = scene.getTrainCameras().copy() 

    
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    # Read change mask
    if change_mask is not None:
        print(f"Using change mask from {change_mask}")
        change_masks = {}
        for filename in os.listdir(change_mask):
            mask_path = os.path.join(change_mask, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = torch.tensor(mask > 0, dtype=torch.bool, device="cuda")
            change_masks[f"whole_{filename}"] = mask
        print("change_masks", change_masks.keys())
        print("camera names：", scene.getTrainCameras()[0].image_name, scene.getTrainCameras()[1].image_name)
    
    # Train
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        
        iter_start.record() # cuda timing
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0: # Increase the levels of SH up to a maximum degree
            gaussians.oneupSHdegree()

        # Pick a random camera
        if change_state == 'mix':
            use_pre = (iteration % ratio == 0)
            if use_pre:
                if not pre_stack:
                    pre_stack = pre_cameras.copy()
                    # print(f"[Iter {iteration}] Refilling pre_stack with {len(pre_stack)} cameras")
                rand_idx = randint(0, len(pre_stack) - 1)
                viewpoint_cam = pre_stack.pop(rand_idx)
                # print("[Iter {}] Using pre-change camera: {}".format(iteration, viewpoint_cam.image_name))
            else:
                if not post_stack:
                    post_stack = post_cameras.copy()
                    # print(f"[Iter {iteration}] Refilling post_stack with {len(post_stack)} cameras")
                rand_idx = randint(0, len(post_stack) - 1)
                viewpoint_cam = post_stack.pop(rand_idx)
                # print("[Iter {}] Using post-change camera: {}".format(iteration, viewpoint_cam.image_name))
        else:
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
            rand_idx = randint(0, len(viewpoint_indices) - 1)
            viewpoint_cam = viewpoint_stack.pop(rand_idx)
            # vind = viewpoint_indices.pop(rand_idx)


        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        bg = torch.rand((3), device="cuda") if opt.random_background else background

        # TODO env: cdgstest
        # render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        render_pkg = render_mean2D(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)

        image, viewspace_point_tensor, visibility_filter, radii, means2D = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"], render_pkg["means2D"] # What is visibility filter???

        # Loss computation and gradient computation
        # Apply change mask if available
        mask = None
        # if change_mask is not None and viewpoint_cam.image_name in change_masks:
        #     mask = change_masks[viewpoint_cam.image_name]
        #     if mask.dim() == 2:
        #         mask = mask.unsqueeze(0) # 从 [H, W] 变为 [1, H, W]
        #     if mask.shape != image.shape[1:]:
        #         if USE_WEIGHTED_CHANGE_MASK:
        #             mask = torch.nn.functional.interpolate(
        #             mask.unsqueeze(0),  # 从 [1, H, W] -> [1, 1, H, W]
        #             size=image.shape[1:], 
        #             mode='bilinear', 
        #             align_corners=False
        #         ).squeeze(0) # 从 [1, 1, H, W] -> [1, H, W]
        #         else:
        #             mask = torch.nn.functional.interpolate(mask.float().unsqueeze(0).unsqueeze(0), size=image.shape[1:], mode='nearest').squeeze().bool()


        if viewpoint_cam.alpha_mask is not None:
            alpha_mask = viewpoint_cam.alpha_mask.cuda()
            image *= alpha_mask

        gt_image = viewpoint_cam.original_image.cuda()

        # Apply change mask if available
        # if mask is not None:
        #     masked_image = image * mask
        #     masked_gt_image = gt_image * mask

        #     Ll1 = l1_loss(masked_image, masked_gt_image)
        #     if FUSED_SSIM_AVAILABLE:
        #         ssim_value = fused_ssim(masked_image.unsqueeze(0), masked_gt_image.unsqueeze(0))
        #     else:
        #         ssim_value = ssim(masked_image, masked_gt_image)

        Ll1 = l1_loss(image, gt_image)
        if FUSED_SSIM_AVAILABLE:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
        else:
            ssim_value = ssim(image, gt_image)

        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        # Depth regularization official implementation
        # Ll1depth_pure = 0.0
        # if depth_l1_weight(iteration) > 0 and viewpoint_cam.depth_reliable:
        #     invDepth = render_pkg["depth"]
        #     mono_invdepth = viewpoint_cam.invdepthmap.cuda()
        #     depth_mask = viewpoint_cam.depth_mask.cuda()

        #     Ll1depth_pure = torch.abs((invDepth  - mono_invdepth) * depth_mask).mean()
        #     Ll1depth = depth_l1_weight(iteration) * Ll1depth_pure 
        #     loss += Ll1depth
        #     Ll1depth = Ll1depth.item()
        # else:
        #     Ll1depth = 0

        loss.backward()
        iter_end.record()


        grad_vis_dir = os.path.join(scene.model_path, "grad_vis")
        loss_vis_dir = os.path.join(scene.model_path, "loss_vis")
        # if iteration % 100 == 0:
        #     save_colorbar = iteration == 100
        #     gaussians.visualize_gradients(
        #         viewpoint_cam,
        #         pipe,
        #         save_dir=grad_vis_dir,
        #         iteration=iteration,
        #         grad_type='xyz',
        #         colormap='hot',
        #         save_colorbar=save_colorbar
        #     )

        # Edit and log
        with torch.no_grad():
            # HERE Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            # ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}", "Depth Loss": f"{ema_Ll1depth_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render_mean2D, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                root_dir = dataset.model_path
                # state mask 0 is original 1 is new 2 is deleted
                # state_mask_path = os.path.join(root_dir, f"state_mask_{iteration}.pt")
                # gaussians.save_state_mask(state_mask_path)


            # Densification
            if iteration < opt.densify_until_iter:
                # visibility_filter = (visibility_filter).to(gaussians.max_radii2D.device)
                # gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])

                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)# 2d mask 和 1d gaussians 列表不对应，加不了change mask

           
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.exposure_optimizer.step()
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0]) 
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)


            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))


def training_report(iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                   
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
   
        torch.cuda.empty_cache()


if __name__ == "__main__":
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[], help="Iterations at which to save checkpoints")
    parser.add_argument("--start_checkpoint", type=str, default = None, help="Path to a checkpoint to start from")
    parser.add_argument("--change_mask", type=str, default=None, help="Directory to change masks")
    args = parser.parse_args(sys.argv[1:]) 
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    safe_state(args.quiet)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.change_mask) 

    print("\nTraining complete.")
