import os, sys
import math
import json
import importlib
import time
from .data.online_render_dataloader import load_obj
import glm
from pathlib import Path

import cv2
import torchvision
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
import open3d as o3d
import sys
# from .src.utils.mesh import Mesh
import nvdiffrast.torch as dr
from .src.utils import obj, mesh, render_utils, render
import torch
import torch.nn.functional as F
import random
from kiui.cam import orbit_camera
import itertools
# from .src.utils.material import Material
# from .utils.camera_util import (
#     FOV_to_intrinsics, 
#     center_looking_at_camera_pose, 
#     get_circular_camera_poses,
# )
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import re

def sample_spherical(phi, theta, cam_radius):
    theta = np.deg2rad(theta)
    phi = np.deg2rad(phi)   

    z = cam_radius * np.cos(phi) * np.sin(theta)
    x = cam_radius * np.sin(phi) * np.sin(theta)
    y = cam_radius * np.cos(theta)

    return x, y, z

def load_mipmap(env_path):
    diffuse_path = os.path.join(env_path, "diffuse.pth")
    diffuse = torch.load(diffuse_path, map_location=torch.device('cpu'))

    specular = []
    for i in range(6):
        specular_path = os.path.join(env_path, f"specular_{i}.pth")
        specular_tensor = torch.load(specular_path, map_location=torch.device('cpu'))
        specular.append(specular_tensor)
    return [specular, diffuse]

ENV = load_mipmap("models/lrm/env_mipmap/6")
materials = (0.0,0.9)
GLCTX = dr.RasterizeCudaContext()

def random_scene():
    train_res = [512, 512]
    cam_near_far = [0.1, 1000.0]
    fovy = np.deg2rad(50)
    spp = 1
    cam_radius = 3.5
    layers = 1
    iter_res = 512
    proj_mtx = render_utils.perspective(fovy, train_res[1] / train_res[0], cam_near_far[0], cam_near_far[1])

    all_azimuths = np.array([0, 90, 180, 270])
    all_elevations = np.array([60, 90, 90, 120])

    # all_azimuths = np.array([0])
    # all_elevations = np.array([60])
    all_mv = []
    all_campos = []
    all_mvp = []
    for index, (azimuths, elevations) in enumerate(zip(all_azimuths, all_elevations)):
        x, y, z = sample_spherical(azimuths, elevations, cam_radius)
        eye = glm.vec3(x, y, z)
        at = glm.vec3(0.0, 0.0, 0.0)
        up = glm.vec3(0.0, 1.0, 0.0)
        view_matrix = glm.lookAt(eye, at, up)
        mv = torch.from_numpy(np.array(view_matrix))
        mvp   = proj_mtx @ (mv)  #w2c
        campos = torch.linalg.inv(mv)[:3, 3]
        all_mv.append(mv[None, ...].cuda())
        all_campos.append(campos[None, ...].cuda())
        all_mvp.append(mvp[None, ...].cuda())
    return all_mv, all_mvp, all_campos

def rendering(ref_mesh):
    all_mv, all_mvp, all_campos = random_scene()
    iter_res = [512, 512]
    iter_spp = 1
    layers = 1
    all_albedo = []
    all_alpha = []
    all_image = []
    all_ccm = []
    all_depth = []
    all_normal = []
    for i in range(len(all_mv)):
        mvp = all_mvp[i]
        campos = all_campos[i]

        with torch.no_grad():
            buffer_dict = render.render_mesh(GLCTX, ref_mesh, mvp, campos, [ENV], None, None, 
                                            materials, iter_res, spp=iter_spp, num_layers=layers, msaa=True, 
                                            background=None, gt_render=True)
        image = buffer_dict['shaded'][0]
        albedo = (buffer_dict['albedo'][0]).clamp(0., 1.)
        alpha = buffer_dict['mask'][0][:, :, 3:]
        ccm = buffer_dict['ccm'][0][...,:3]
        alpha = buffer_dict['mask'][0][...,:3]
        albedo = buffer_dict['albedo'][0].clamp(0., 1.)
        # breakpoint()
        ccm = ccm * alpha
        depth = buffer_dict['depth'][0]
        normal = buffer_dict['gb_normal'][0]
        all_image.append(image)
        all_albedo.append(albedo)
        all_alpha.append(alpha)
        all_ccm.append(ccm)
        all_depth.append(depth)
        all_normal.append(normal)
    all_albedo = torch.stack(all_albedo)
    all_alpha = torch.stack(all_alpha)
    all_ccm = torch.stack(all_ccm)
    all_normal = torch.stack(all_normal)

    all_image = torch.stack(all_image)
    all_depth = torch.stack(all_depth)

    # breakpoint()
    return all_image.detach(), all_albedo.detach(), all_alpha.detach(), all_ccm.detach(), all_depth.detach(), all_normal.detach()

def render_mesh(mesh_path):
    ref_mesh = load_obj(mesh_path, return_attributes=False)
    ref_mesh = mesh.auto_normals(ref_mesh)
    ref_mesh = mesh.compute_tangents(ref_mesh)
    ref_mesh.rotate_x_90()
    # print(f"start ==> {mesh_path}")
    rgb, albedo, alpha, ccm, depth, normal = rendering(ref_mesh)
    depth = depth[...,:3] * alpha
    # breakpoint()
    torchvision.utils.save_image(rgb.permute(0, 3, 1, 2), f"debug_image/{mesh_path.split('/')[-1].split('.')[0]}_rgb.png")
    torchvision.utils.save_image(albedo.permute(0, 3, 1, 2), f"debug_image/{mesh_path.split('/')[-1].split('.')[0]}_albedo.png")
    torchvision.utils.save_image(alpha.permute(0, 3, 1, 2), f"debug_image/{mesh_path.split('/')[-1].split('.')[0]}_alpha.png")
    torchvision.utils.save_image(ccm.permute(0, 3, 1, 2), f"debug_image/{mesh_path.split('/')[-1].split('.')[0]}_ccm.png")
    torchvision.utils.save_image(depth.permute(0, 3, 1, 2), f"debug_image/{mesh_path.split('/')[-1].split('.')[0]}_depth.png", normalize=True)
    torchvision.utils.save_image(normal.permute(0, 3, 1, 2), f"debug_image/{mesh_path.split('/')[-1].split('.')[0]}_normal.png")
    print(f"end ==> {mesh_path}")

if __name__ == '__main__':
    render_mesh("./meshes_online/bubble_mart_blue/bubble_mart_blue.obj")
