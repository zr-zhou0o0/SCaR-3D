# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr

'''
render 
with color override(black and white)
'''

import torch
from scene import Scene
import os
import re
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render_mean2D as render 
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, color_tensor, separate_sh):

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth")

    print("renpath:", render_path)
    print("gts_path", gts_path) 
    print("depth_path", depth_path) 

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)

    color = color_tensor

    print("gaus len", gaussians.get_xyz.shape[0])

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

        uid = int(view.image_name.split(".")[0].split("_")[-1])
        
        render_all = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, override_color=color)
        # render_all = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, override_color=None) # XXX temp test

        rendering = render_all["render"]
        depth = render_all["depth"]

        gt = view.original_image[0:3, :, :]

        if train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(uid) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(uid) + ".png"))
        torchvision.utils.save_image(depth, os.path.join(depth_path, '{0:05d}'.format(uid) + "_depth.png"))


def render_sets(render_name, dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, color : torch.tensor, separate_sh: bool):


    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        import time
        start_time = time.time()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            print('model path', dataset.model_path)
            print('render name', render_name)
            print('load iter', scene.loaded_iter)
            print('color', color)
            render_set(dataset.model_path, render_name, scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, dataset.train_test_exp, color, separate_sh)

        end_time = time.time()
        print("Rendering time:", end_time - start_time)


if __name__ == "__main__":
    
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true") 
    # parser.add_argument("--color", type=str, default=None, help="Color matrix path")
    parser.add_argument("--render_name", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    safe_state(args.quiet)


    render_sets(args.render_name, model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, None, SPARSE_ADAM_AVAILABLE)