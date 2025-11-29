#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

'''
scene/gaussian_model_incre.py
renamed as gaussian_model.py
use this file in the paper experiments
'''

import torch
import numpy as np
from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
import json
import matplotlib.pyplot as plt

from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation

# from gaussian_renderer import camera2rasterizer


class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm
        
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree, optimizer_type="default"):
        self.active_sh_degree = 0
        self.optimizer_type = optimizer_type
        self.max_sh_degree = sh_degree  
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        # self.existing_count = 0
        self.stored_gaussians = None
        # self.state_mask = torch.empty(0) # 0 is existing, 1 is add, 2 is deleted

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )
    
    def restore(self, model_args, training_args):
        (self.active_sh_degree, 
        self._xyz, 
        self._features_dc, 
        self._features_rest,
        self._scaling, 
        self._rotation, 
        self._opacity,
        self.max_radii2D, 
        xyz_gradient_accum, 
        denom,
        opt_dict, 
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        self.state_mask = torch.zeros(self._xyz.shape[0]) # , device="cuda") 


    # def zero_grad_for_existing(self, existing_count):
    #     """
    #     将前 existing_count 个高斯参数的梯度置零
    #     """
    #     # 所有需要处理的高斯参数列表
    #     gaussian_params = [
    #         self._xyz,
    #         self._features_dc,
    #         self._features_rest,
    #         self._scaling,
    #         self._rotation,
    #         self._opacity
    #     ]
        
    #     for param in gaussian_params:
    #         if param.grad is not None:
    #             # print("set zero grad")
    #             # print("param grad shape", param.grad.shape)
    #             # 确保不越界
    #             num_elements = param.shape[0]
    #             if existing_count > num_elements:
    #                 print(f"Warning: existing_count {existing_count} exceeds the number of elements {num_elements}.")
    #                 param.grad.zero_()  
    #                 continue
                
    #             # 对第一个维度（高斯数量维度）的前 existing_count 个元素梯度置零
    #             param.grad[:existing_count].zero_()
    #             # print(f"Zeroed gradients for the first {existing_count} elements of {param.shape}.")
    #             # print("paramgrad 0-20", param.grad[:10])
    #             # print("paramgrad -20", param.grad[-10:]) 
    #         else:
    #             print("param grad is None")   


    # def freeze_parameters(self):
    #     """
    #     Freeze all parameters of the GaussianModel to prevent gradient updates
        
    #     This is useful for hierarchical model training or transfer learning scenarios,
    #     such as when you want to add new points based on an existing model without 
    #     changing the original points.
        
    #     The freezing operation affects the following parameters:
    #     - Positions (_xyz)
    #     - Features (_features_dc, _features_rest)
    #     - Scaling (_scaling)
    #     - Rotation (_rotation)
    #     - Opacity (_opacity)
        
    #     Note: This operation modifies the original parameters to no longer require gradients
    #     """
    #     # Record the original parameter count
    #     original_point_count = self._xyz.shape[0]
    #     print(f"Freezing {original_point_count} points in GaussianModel")
        
    #     # Freeze each parameter
    #     self._xyz.requires_grad_(False)
    #     self._features_dc.requires_grad_(False)
    #     self._features_rest.requires_grad_(False)
    #     self._scaling.requires_grad_(False)
    #     self._rotation.requires_grad_(False)
    #     self._opacity.requires_grad_(False)
        
    #     # If optimizer is already set, update the parameters in the optimizer
    #     if self.optimizer is not None:
    #         # Replace parameters with frozen versions
    #         updated_param_groups = []
    #         for group in self.optimizer.param_groups:
    #             if group["name"] == "xyz":
    #                 group["params"] = [self._xyz]
    #             elif group["name"] == "f_dc":
    #                 group["params"] = [self._features_dc]
    #             elif group["name"] == "f_rest":
    #                 group["params"] = [self._features_rest]
    #             elif group["name"] == "opacity":
    #                 group["params"] = [self._opacity]
    #             elif group["name"] == "scaling":
    #                 group["params"] = [self._scaling]
    #             elif group["name"] == "rotation":
    #                 group["params"] = [self._rotation]
                
    #             # Set learning rate to 0 to ensure no updates
    #             group["lr"] = 0.0
    #             updated_param_groups.append(group)
            
    #         # Recreate the optimizer with updated parameter groups while retaining state
    #         self.optimizer = torch.optim.Adam(
    #             updated_param_groups,
    #             lr=0.0,
    #             eps=1e-15
    #         )
        

    # def zero_grad_for_state_zero(self):
    #     '''
    #     将所有 state_mask=0 的高斯参数的梯度置零，包括optimizer中的状态
    #     '''
    #     # 找出 state_mask==0 的行
    #     mask_zero = (self.state_mask == 0)
        
    #     if self.optimizer is not None:
    #         # 处理optimizer中每个参数组的梯度
    #         for group in self.optimizer.param_groups:
    #             param_name = group["name"]
    #             param = group["params"][0]  # 每个组只有一个参数
                
    #             if param.grad is None:
    #                 print(f"zero_grad_for_state_zero: {param_name} has no gradient.")
    #                 continue
                    
    #             # 确保 mask 与参数的第一维对齐
    #             if mask_zero.shape[0] != param.grad.shape[0]:
    #                 print(f"zero_grad_for_state_zero: mask length {mask_zero.shape[0]} != grad length {param.grad.shape[0]}")
    #                 continue
                
    #             # 将这些行的梯度置零
    #             param.grad[mask_zero] = 0
                
    #             # 同时处理optimizer的内部状态（exp_avg 和 exp_avg_sq）
    #             stored_state = self.optimizer.state.get(param, None)
    #             if stored_state is not None:
    #                 if 'exp_avg' in stored_state:
    #                     stored_state['exp_avg'][mask_zero] = 0
    #                 if 'exp_avg_sq' in stored_state:
    #                     stored_state['exp_avg_sq'][mask_zero] = 0
                
    #             # print(f"Zeroed gradients and optimizer states for {mask_zero.sum().item()} Gaussians (state=0) in {param_name}")
    #     else:
    #         # 如果没有optimizer，直接处理参数的梯度
    #         gaussian_params = {
    #             "xyz":            self._xyz,
    #             "features_dc":    self._features_dc,
    #             "features_rest":  self._features_rest,
    #             "scaling":        self._scaling,
    #             "rotation":       self._rotation,
    #             "opacity":        self._opacity
    #         }
            
    #         for name, param in gaussian_params.items():
    #             if param.grad is None:
    #                 print(f"zero_grad_for_state_zero: {name} has no gradient.")
    #                 continue
    #             # 确保 mask 与参数的第一维对齐
    #             if mask_zero.shape[0] != param.grad.shape[0]:
    #                 print(f"zero_grad_for_state_zero: mask length {mask_zero.shape[0]} != grad length {param.grad.shape[0]}")
    #                 continue
    #             # 将这些行的梯度置零
    #             param.grad[mask_zero] = 0



    # def save_state_mask(self, path):
    #     """
    #     Save the state mask as pt file.
    #     """
    #     mkdir_p(os.path.dirname(path))
    #     torch.save(self.state_mask, path)
    #     print(f"State mask saved to {path}")


    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_features_dc(self):
        return self._features_dc
    
    @property
    def get_features_rest(self):
        return self._features_rest
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    @property
    def get_exposure(self):
        return self._exposure

    def get_exposure_from_name(self, image_name):
        if self.pretrained_exposures is None:
            return self._exposure[self.exposure_mapping[image_name]]
        else:
            return self.pretrained_exposures[image_name]
    
    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd : BasicPointCloud, cam_infos : int, spatial_lr_scale : float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")
        self.exposure_mapping = {cam_info.image_name: idx for idx, cam_info in enumerate(cam_infos)}
        self.pretrained_exposures = None
        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1)
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]

        if self.optimizer_type == "default":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        elif self.optimizer_type == "sparse_adam":
            self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

        self.exposure_optimizer = torch.optim.Adam([self._exposure])

        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        
        self.exposure_scheduler_args = get_expon_lr_func(training_args.exposure_lr_init, training_args.exposure_lr_final,
                                                        lr_delay_steps=training_args.exposure_lr_delay_steps,
                                                        lr_delay_mult=training_args.exposure_lr_delay_mult,
                                                        max_steps=training_args.iterations)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        # hasattr
        if hasattr(self, 'pretrained_exposures'):
            if self.pretrained_exposures is None:
                for param_group in self.exposure_optimizer.param_groups:
                    param_group['lr'] = self.exposure_scheduler_args(iteration)
        else: 
            for param_group in self.exposure_optimizer.param_groups:
                param_group['lr'] = self.exposure_scheduler_args(iteration)

        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self._xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = self.inverse_opacity_activation(torch.min(self.get_opacity, torch.ones_like(self.get_opacity)*0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path, use_train_test_exp = False, cam_infos : int = None): # HERE MODIFIED. add a cam info for self._exposure
        plydata = PlyData.read(path)
        if use_train_test_exp:
            exposure_file = os.path.join(os.path.dirname(path), os.pardir, os.pardir, "exposure.json")
            if os.path.exists(exposure_file):
                with open(exposure_file, "r") as f:
                    exposures = json.load(f)
                self.pretrained_exposures = {image_name: torch.FloatTensor(exposures[image_name]).requires_grad_(False).cuda() for image_name in exposures}
                print(f"Pretrained exposures loaded.")
            else:
                print(f"No exposure to be loaded at {exposure_file}")
                self.pretrained_exposures = None

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

        exposure = torch.eye(3, 4, device="cuda")[None].repeat(len(cam_infos), 1, 1) # XXX
        self._exposure = nn.Parameter(exposure.requires_grad_(True))

        self.state_mask = torch.zeros(self._xyz.shape[0]) # , device="cuda") 

        print("init 0 state:", self._xyz.shape[0])


    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask): # optimizer's parameters have to be aligned with the model's parameters
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None: # mask points are points to be stored
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    # def prune_points(self, mask, select_phase=False): # here is delete mask
    #     '''
    #     the passed mask is a delete mask, bool
    #     '''
    #     # Prohibit the existing points to be deleted
    #     # if existing_count > 0:
    #     #     # 把mask的前existingpoint都置为false
    #     #     print("set existing count to false: ", existing_count, "/", mask.shape[0])
    #     #     mask[:existing_count] = False

    #     # Count how many existing Gaussians are being deleted
    #     # existing_count = self.existing_count
    #     # # if existing_count > 0:
    #     # existing_delete_mask = mask[:existing_count]
    #     # existing_delete_count = torch.sum(existing_delete_mask).item()
    #     # print(f"Deleting {existing_delete_count} Gaussians from existing set of {existing_count}")
    #     # if existing_delete_count > 0:
    #     #     self.store_deleted_gaussians(existing_delete_mask)
    #     #     self.existing_count -= existing_delete_count
    #     #     print(f"Updated existing count: {self.existing_count}")

    #     mask = mask.to(self.state_mask.device) # ensure mask is on the same device as state_mask
    #     # intersection_mask = mask & (self.state_mask == 0)
    #     intersection_mask = mask & (self.state_mask != 0)
    #     # if intersection_mask.sum() > 0:
    #         # print(f"Deleting {intersection_mask.sum().item()} Gaussians from existing set of {self.state_mask.shape[0]}")
    #         # self.store_deleted_gaussians(intersection_mask)

    #     # valid_points_mask = ~mask # here is valid mask, NOT logic
    #     valid_points_mask = ~intersection_mask # here is valid mask, NOT logic
    #     self.state_mask = self.state_mask[valid_points_mask] # update state mask

    #     # if not select_phase:
    #     optimizable_tensors = self._prune_optimizer(valid_points_mask)

    #     self._xyz = optimizable_tensors["xyz"]
    #     self._features_dc = optimizable_tensors["f_dc"]
    #     self._features_rest = optimizable_tensors["f_rest"]
    #     self._opacity = optimizable_tensors["opacity"]
    #     self._scaling = optimizable_tensors["scaling"]
    #     self._rotation = optimizable_tensors["rotation"]

    #     self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

    #     self.denom = self.denom[valid_points_mask]

    #     # valid_points_mask = (valid_points_mask).to(self.max_radii2D.device) # but why only max radii2d on cpu?
    #     if not select_phase:
    #         self.max_radii2D = self.max_radii2D[valid_points_mask]
        
    #     if hasattr(self, 'tmp_radii') and self.tmp_radii is not None:
    #         self.tmp_radii = self.tmp_radii[valid_points_mask]


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii):
        d = {"xyz": new_xyz,
        "f_dc": new_features_dc,
        "f_rest": new_features_rest,
        "opacity": new_opacities,
        "scaling" : new_scaling,
        "rotation" : new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.tmp_radii = torch.cat((self.tmp_radii, new_tmp_radii))
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

        new_state_mask = torch.ones((new_xyz.shape[0])) # , device="cuda")
        new_state_mask = new_state_mask.to(self.state_mask.device) 
        self.state_mask = torch.cat((self.state_mask, new_state_mask), dim=0)


    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0] # self.get_xyz.shape[0] = 36659, grads.shape[0] = 1295423
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")

        print("grads shape",grads.shape)  # 检查原始形状
        print("squeezed",grads.squeeze().shape)  # 检查squeeze后的形状, [1295423], Froze 1295423 existing Gaussians
        print("padded_grad shape", padded_grad.shape)  # 检查padded_grad的形状, [36659]

        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)
        state_mask = self.state_mask.to(selected_pts_mask.device)
        selected_pts_mask = torch.logical_and(selected_pts_mask, state_mask)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), 3),device="cuda")
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)
        new_tmp_radii = self.tmp_radii[selected_pts_mask].repeat(N)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation, new_tmp_radii)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)
        
        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        new_tmp_radii = self.tmp_radii[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation, new_tmp_radii)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size, radii):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.tmp_radii = radii
        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent) # self.get_xyz.shape[0] = 36659, grads.shape[0] = 1295423

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)
        tmp_radii = self.tmp_radii
        self.tmp_radii = None

        torch.cuda.empty_cache()

        # print("After densify and prune, total gaussian length: ", self.get_xyz.shape[0], "\nstate_mask length: ", self.state_mask.shape[0])
        # print("0 state: ", torch.sum(self.state_mask == 0).item())
        # print("1 state: ", torch.sum(self.state_mask == 1).item())
        # print("2 state: ", torch.sum(self.state_mask == 2).item())
        # assert self.get_xyz.shape[0] == self.state_mask.shape[0], "gaussian length and state mask length are not equal"


    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1

    # def state_mask_stats(self):
    #     print("total gaussians: ", self.get_xyz.shape[0])
    #     print("state mask length: ", self.state_mask.shape[0])
    #     print("0 state: ", torch.sum(self.state_mask == 0).item())
    #     print("1 state: ", torch.sum(self.state_mask == 1).item())
    #     print("2 state: ", torch.sum(self.state_mask == 2).item())
    
    # def prune_drifter(self, grad_threshold):
    #     # bound = torch.sum(self.state_mask == 0).item()

    #     grads = self.xyz_gradient_accum / self.denom
    #     grads[grads.isnan()] = 0.0

    #     selected_pts_mask = torch.where(torch.norm(grads, dim=-1) <= grad_threshold, True, False)

    #     # 确保所有张量都在同一个设备上
    #     selected_pts_mask = selected_pts_mask.to(self.state_mask.device)
    #     # self.state_mask = self.state_mask.to(selected_pts_mask.device)

    #     prune_pts_mask = torch.logical_and(self.state_mask, selected_pts_mask)
    #     # 获得它的形状/长度
    #     print("prune_pts_mask shape", prune_pts_mask.shape)
    #     prune_pts_mask = prune_pts_mask.squeeze()

    #     self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    #     self.prune_points(prune_pts_mask)

    #     print("After prune drifters, total gaussian length: ", self.get_xyz.shape[0], "\nstate_mask length: ", self.state_mask.shape[0])
    #     print("0 state: ", torch.sum(self.state_mask == 0).item())
    #     print("1 state: ", torch.sum(self.state_mask == 1).item())
    #     print("2 state: ", torch.sum(self.state_mask == 2).item())
    #     assert self.get_xyz.shape[0] == self.state_mask.shape[0], "gaussian length and state mask length are not equal"


    # def compute_average_gradients(self, viewspace_point_tensor, update_filter):
    #     xyz_grad = torch.norm(viewspace_point_tensor.grad[update_filter], dim=-1, keepdim=True)
    #     grad_sum = torch.sum(xyz_grad, dim=0)
    #     grad_max = torch.max(xyz_grad, dim=0).values
    #     grad_min = torch.min(xyz_grad, dim=0).values
    #     return grad_sum, grad_max, grad_min


    def apply_weights(self, camera, weights, weights_cnt, image_weights):
        from gaussian_renderer import camera2rasterizer
        rasterizer = camera2rasterizer(
            camera, torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")
        )
        rasterizer.apply_weights(
            self.get_xyz,
            None,
            self.get_opacity,
            None,
            weights,
            self.get_scaling,
            self.get_rotation,
            None,
            weights_cnt,
            image_weights,
        )


def combine_gslist(gslist):
        """
        Combine a list of GaussianModel objects into a single GaussianModel object.
        """
        # Initialize a new GaussianModel object with the same SH degree as the first model in the list
        combined_model = GaussianModel(gslist[0].max_sh_degree)
        
        # Prepare lists to hold parameters from all models
        xyz_list = []
        features_dc_list = []
        features_rest_list = []
        opacity_list = []
        scaling_list = []
        rotation_list = []
        mip_filter_list = []
        # Collect parameters from each model
        for model in gslist:
            xyz_list.append(model.get_xyz.detach())
            features_dc_list.append(model._features_dc.detach())
            features_rest_list.append(model._features_rest.detach())
            opacity_list.append(model._opacity.detach())
            scaling_list.append(model._scaling.detach())
            rotation_list.append(model._rotation.detach())

            if hasattr(model, "mip_filter"):
                mip_filter_list.append(model.mip_filter.detach())
        
        # Concatenate all parameters
        combined_model._xyz = nn.Parameter(torch.cat(xyz_list, dim=0))
        combined_model._features_dc = nn.Parameter(torch.cat(features_dc_list, dim=0))
        combined_model._features_rest = nn.Parameter(torch.cat(features_rest_list, dim=0))
        combined_model._opacity = nn.Parameter(torch.cat(opacity_list, dim=0))
        combined_model._scaling = nn.Parameter(torch.cat(scaling_list, dim=0))
        combined_model._rotation = nn.Parameter(torch.cat(rotation_list, dim=0))
        
        # Also initialize/combine other necessary properties
        combined_model.active_sh_degree = gslist[0].active_sh_degree
        
        # Initialize max_radii2D with the right size
        n_points = combined_model._xyz.shape[0]
        combined_model.max_radii2D = torch.zeros(n_points, device=combined_model._xyz.device)
        
        # Initialize other buffers with the appropriate sizes
        combined_model.xyz_gradient_accum = torch.zeros((n_points, 1), device=combined_model._xyz.device)
        combined_model.denom = torch.zeros((n_points, 1), device=combined_model._xyz.device)

        # NOTE: hard code pseudo optimizer
        l = [
            {'params': [combined_model._xyz], 'lr': 0.001, "name": "xyz"},
            {'params': [combined_model._features_dc], 'lr': 0.001, "name": "f_dc"},
            {'params': [combined_model._features_rest], 'lr': 0.001 / 20.0, "name": "f_rest"},
            {'params': [combined_model._opacity], 'lr': 0.001, "name": "opacity"},
            {'params': [combined_model._scaling], 'lr': 0.001, "name": "scaling"},
            {'params': [combined_model._rotation], 'lr': 0.001, "name": "rotation"}
        ]
        combined_model.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        
        # Copy spatial_lr_scale from the first model
        combined_model.spatial_lr_scale = gslist[0].spatial_lr_scale
        
        # Set percent_dense to the same as the first model
        combined_model.percent_dense = gslist[0].percent_dense
        
        # Print information about the combined model
        print(f"Combined {len(gslist)} models with a total of {n_points} points")
        for i, model in enumerate(gslist):
            print(f"  Model {i}: {model.get_xyz.shape[0]} points")
        
        return combined_model
