import os, sys
import math
import json
import importlib
import time
import glm
from pathlib import Path

import cv2
import torchvision
import random
from tqdm import tqdm
import numpy as np
from PIL import Image
# import webdataset as wds
from torch.utils.data import DataLoader
import open3d as o3d
import sys
import nvdiffrast.torch as dr
from ..src.utils import obj, mesh, render_utils, render
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
import random
from kiui.cam import orbit_camera
import itertools
from ..src.utils.material import Material
from ..utils.camera_util import (
    FOV_to_intrinsics, 
    center_looking_at_camera_pose, 
    get_circular_camera_poses,
)
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import re
# import torch.multiprocessing as mp
# # 设置 torch.multiprocessing 的启动方法为 'spawn'
# mp.set_start_method('spawn', force=True)

GLCTX = [None] * torch.cuda.device_count()  # 存储每个 GPU 的上下文

def initialize_extension(gpu_id):
    global GLCTX
    if GLCTX[gpu_id] is None:
        print(f"Initializing extension module renderutils_plugin on GPU {gpu_id}...")
        torch.cuda.set_device(gpu_id)
        GLCTX[gpu_id] = dr.RasterizeCudaContext()
    return GLCTX[gpu_id]

def spherical_camera_pose(azimuths: np.ndarray, elevations: np.ndarray, radius=2.5):
    azimuths = np.deg2rad(azimuths)
    elevations = np.deg2rad(elevations)

    xs = radius * np.cos(elevations) * np.cos(azimuths)
    ys = radius * np.cos(elevations) * np.sin(azimuths)
    zs = radius * np.sin(elevations)

    cam_locations = np.stack([xs, ys, zs], axis=-1)
    cam_locations = torch.from_numpy(cam_locations).float()

    c2ws = center_looking_at_camera_pose(cam_locations)
    return c2ws


def get_camera(
    azimuths, elevations, blender_coord=True, extra_view=False,radius=1.0
):
    cameras = []
    for index, azimuth in enumerate(azimuths):
        elevation = elevations[index]
        elevation = 90 - elevation 
        pose = orbit_camera(-elevation, azimuth, radius=radius) # kiui's elevation is negated, [4, 4]

        # opengl to blender
        if blender_coord:
            pose[2] *= -1
            pose[[1, 2]] = pose[[2, 1]]

        cameras.append(pose.flatten())

    if extra_view:
        cameras.append(np.zeros_like(cameras[0]))

    return torch.from_numpy(np.stack(cameras, axis=0)).float() # [num_frames, 16]

def load_mipmap(env_path):
    diffuse_path = os.path.join(env_path, "diffuse.pth")
    diffuse = torch.load(diffuse_path, map_location=torch.device('cpu'))

    specular = []
    for i in range(6):
        specular_path = os.path.join(env_path, f"specular_{i}.pth")
        specular_tensor = torch.load(specular_path, map_location=torch.device('cpu'))
        specular.append(specular_tensor)
    return [specular, diffuse]

def convert_to_white_bg(image, write_bg=True):
    alpha = image[:, :, 3:]
    if write_bg:
        return image[:, :, :3] * alpha + 1. * (1 - alpha)
    else:
        return image[:, :, :3] * alpha
    
def load_obj(path, return_attributes=False):
    return obj.load_obj(path, clear_ks=True, mtl_override=None, return_attributes=return_attributes)

def custom_collate_fn(batch):
    return batch

def collate_fn_wrapper(batch):
    return custom_collate_fn(batch)

class ObjaverseData(Dataset):
    def __init__(self,
        root_dir='obj_demo',
        light_dir= 'data/env_mipmap/',
        target_view_num=4,
        fov=30,
        camera_distance=4.5,
        validation=False,
        random_camera=False,
        random_elevation=False,
        ):
        self.root_dir = Path(root_dir)
        self.light_dir = light_dir
        self.all_env_name = []
        self.if_validation = validation
        self.random_camera = random_camera
        for temp_dir in os.listdir(light_dir):
            if os.listdir(os.path.join(self.light_dir, temp_dir)):
                self.all_env_name.append(temp_dir)
        self.target_view_num = target_view_num
        self.fov = fov
        
        self.train_res = [512, 512]
        self.cam_near_far = [0.1, 1000.0]
        self.random_elevation = random_elevation
        self.spp = 1
        self.cam_radius = camera_distance
        self.layers = 1
        
        numbers = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        self.combinations = list(itertools.product(numbers, repeat=2))

        with open("pbr_objs_final_mesh_valid.json", 'r') as file:
            all_paths = json.load(file)

        if not self.if_validation:
            self.paths = all_paths[:-100]
        if self.if_validation:
            self.paths = all_paths[-100:]
        
        print('total object num:', len(self.paths))
        print('============= length of dataset %d =============' % len(self.paths))

    def __len__(self):
        return len(self.paths)
    
    def calculate_fov(self, initial_distance, initial_fov, new_distance):
        initial_fov_rad = math.radians(initial_fov)
        
        height = 2 * initial_distance * math.tan(initial_fov_rad / 2)
        
        new_fov_rad = 2 * math.atan(height / (2 * new_distance))
        
        new_fov = math.degrees(new_fov_rad)
        
        return new_fov

    def load_obj(self, path):
        return obj.load_obj(path, clear_ks=True, mtl_override=None)
    
    def sample_spherical(self, phi, theta, cam_radius):
        theta = np.deg2rad(theta)
        phi = np.deg2rad(phi)   

        z = cam_radius * np.cos(phi) * np.sin(theta)
        x = cam_radius * np.sin(phi) * np.sin(theta)
        y = cam_radius * np.cos(theta)
 
        return x, y, z
    
    def _random_scene(self, num_frame):
        if self.random_camera and not self.if_validation:
            random_perturbation = random.uniform(-1.5, 1.5)
            cam_radius = self.cam_radius + random_perturbation
            fov = self.calculate_fov(initial_distance=self.cam_radius, initial_fov=self.fov, new_distance=cam_radius)
            fov_rad = np.deg2rad(fov)
        else:
            cam_radius = self.cam_radius
            fov = self.fov
            fov_rad = np.deg2rad(self.fov)
        iter_res = self.train_res
        proj_mtx = render_utils.perspective(fov_rad, iter_res[1] / iter_res[0], self.cam_near_far[0], self.cam_near_far[1])

        start_angle = random.uniform(0, 360)
        azimuths = [(start_angle + i * 90) % 360 for i in range(num_frame)]
        if self.random_elevation:
            elevations = [random.uniform(30, 150)] * num_frame
        else:
            elevations = [90] * num_frame

        all_mv = []
        all_mvp = []
        all_campos = []

        input_extrinsics = get_camera(azimuths, elevations=elevations, extra_view=False, radius=cam_radius)
        input_extrinsics = input_extrinsics[:, :12]
        input_Ks = FOV_to_intrinsics(fov)
        input_intrinsics = input_Ks.flatten(0).unsqueeze(0).repeat(len(azimuths), 1)
        input_intrinsics = torch.stack([
            input_intrinsics[:, 0], input_intrinsics[:, 4], 
            input_intrinsics[:, 2], input_intrinsics[:, 5],
        ], dim=-1)
        camera_embedding = torch.cat([input_extrinsics, input_intrinsics], dim=-1)

        if not self.if_validation:
            camera_embedding = camera_embedding + torch.rand_like(camera_embedding) * 0.04

        for index, azimuth in enumerate(azimuths):
            x, y, z = self.sample_spherical(azimuth, elevations[index], cam_radius)
            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...])
            all_mvp.append(mvp[None, ...])
            all_campos.append(campos[None, ...])

        return all_mv, all_mvp, all_campos, None, camera_embedding
        
    def load_im(self, path, color):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        alpha = image[:, :, 3:]
        image = image[:, :, :3] * alpha + color * (1 - alpha)

        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()
        alpha = torch.from_numpy(alpha).permute(2, 0, 1).contiguous().float()
        return image, alpha
    
    def load_albedo(self, path, color, mask):
        '''
        replace background pixel with random color in rendering
        '''
        pil_img = Image.open(path)

        image = np.asarray(pil_img, dtype=np.float32) / 255.
        image = torch.from_numpy(image).permute(2, 0, 1).contiguous().float()

        color = torch.ones_like(image)
        image = image * mask + color * (1 - mask)
        return image
    
    def convert_to_white_bg(self, image):
        alpha = image[:, :, 3:]
        return image[:, :, :3] * alpha + 1. * (1 - alpha)

    def __getitem__(self, index):
        obj_path = os.path.join(self.root_dir, self.paths[index]+".pth")
        pose_list = []
        env_list = []
        material_list = []
        camera_pos = []
        c2w_list = []
        random_env = False
        random_mr = False
        selected_env = random.randint(0, len(self.all_env_name)-1)
        materials = random.choice(self.combinations)
        if random.random() < 0.5:
            materials = list(materials)
            materials[0] = 0.0
            materials = tuple(materials)

        all_mv, all_mvp, all_campos, can_c2w, camera_embedding = self._random_scene(self.target_view_num)
        
        for index in range(self.target_view_num):
            mv = all_mv[index]
            mvp = all_mvp[index]
            campos = all_campos[index]
            if random_env:
                selected_env = random.randint(0, len(self.all_env_name)-1)
            env_path = os.path.join(self.light_dir, self.all_env_name[selected_env])
            env = load_mipmap(env_path)

            if random_mr:
                materials = random.choice(self.combinations)
            pose_list.append(mvp)
            camera_pos.append(campos)
            c2w_list.append(mv)
            env_list.append(env)
            material_list.append(materials)
        data = {
            'target_view_num': self.target_view_num,
            'obj_path': obj_path,
            'pose_list': pose_list,
            'camera_pos': camera_pos,
            'c2w_list': c2w_list,
            'env_list': env_list,
            'material_list': material_list,
            'can_c2w': can_c2w,
            'camera_embedding': camera_embedding
        }
        
        return data

def rotate_x(a, device=None):
        s, c = np.sin(a), np.cos(a)
        return torch.tensor([[1, 0, 0, 0], 
                            [0, c,-s, 0], 
                            [0, s, c, 0], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)
def rotate_z(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, -s, 0, 0],
                        [ s,  c, 0, 0],
                        [ 0,  0, 1, 0],
                        [ 0,  0, 0, 1]], dtype=torch.float32, device=device)
def rotate_y(a, device=None):
    s, c = np.sin(a), np.cos(a)
    return torch.tensor([[ c, 0,  s, 0],
                        [ 0, 1,  0, 0],
                        [-s, 0,  c, 0],
                        [ 0, 0,  0, 1]], dtype=torch.float32, device=device)

def collate_fn(batch):
    gpu_id = torch.cuda.current_device()  # 获取当前线程的 GPU ID
    glctx = initialize_extension(gpu_id)
    batch_size = len(batch)
    iter_res = [512, 512]
    iter_spp = 1
    layers = 1

    target_images, target_alphas, target_depths, target_ccms, target_normals, target_albedos = [], [], [], [], [], []
    target_w2cs, target_Ks, target_camera_pos = [], [], []
    target_cam_emebdding = []


    for sample in batch:
        target_cam_emebdding.append(sample["camera_embedding"])
        obj_path = sample['obj_path']
        with torch.no_grad():
            mesh_attributes = torch.load(obj_path, map_location=torch.device('cpu'))
            v_pos = mesh_attributes["v_pos"].cuda()
            # random_rotate_degree = random.uniform(-30, 30)
            # v_pos = v_pos @ rotate_y(random_rotate_degree, device=v_pos.device)[:3, :3]
            v_nrm = mesh_attributes["v_nrm"].cuda()
            v_tex = mesh_attributes["v_tex"].cuda()
            v_tng = mesh_attributes["v_tng"].cuda()
            t_pos_idx = mesh_attributes["t_pos_idx"].cuda()
            t_nrm_idx = mesh_attributes["t_nrm_idx"].cuda()
            t_tex_idx = mesh_attributes["t_tex_idx"].cuda()
            t_tng_idx = mesh_attributes["t_tng_idx"].cuda()
            material = Material(mesh_attributes["mat_dict"])
            material = material.cuda()
            ref_mesh = mesh.Mesh(v_pos=v_pos, v_nrm=v_nrm, v_tex=v_tex, v_tng=v_tng, 
                            t_pos_idx=t_pos_idx, t_nrm_idx=t_nrm_idx, 
                            t_tex_idx=t_tex_idx, t_tng_idx=t_tng_idx, material=material)

        pose_list_sample = sample['pose_list']  # mvp
        camera_pos_sample = sample['camera_pos'] # campos, mv.inverse
        c2w_list_sample = sample['c2w_list']    # mv
        env_list_sample = sample['env_list']
        material_list_sample = sample['material_list']

        sample_target_images, sample_target_ccms, sample_target_alphas, sample_target_depths, sample_target_normals, sample_target_albedos = [], [], [], [], [], []
        sample_target_w2cs, sample_target_Ks, sample_target_camera_pos = [], [], []

        for i in range(len(pose_list_sample)):
            mvp = pose_list_sample[i]
            campos = camera_pos_sample[i]
            env = env_list_sample[i]
            materials = material_list_sample[i]

            with torch.no_grad():
                buffer_dict = render.render_mesh(glctx, ref_mesh, mvp.cuda(), campos.cuda(), [env], None, None, 
                                                    materials, iter_res, spp=iter_spp, num_layers=layers, msaa=True, 
                                                    background=None, gt_render=True)
            image = convert_to_white_bg(buffer_dict['shaded'][0], write_bg=False)
            # ccm = convert_to_white_bg(buffer_dict['ccm'][0], write_bg=False)

            # alpha = buffer_dict['mask'][0][...,:3]
            # albedo = convert_to_white_bg(buffer_dict['albedo'][0]).clamp(0., 1.)
            # ccm = ccm * alpha
            # depth = convert_to_white_bg(buffer_dict['depth'][0], write_bg=False)
            normal = convert_to_white_bg(buffer_dict['gb_normal'][0], write_bg=False)


            sample_target_images.append(image)
            # sample_target_ccms.append(ccm)
            # sample_target_albedos.append(albedo)
            # sample_target_alphas.append(alpha)
            # sample_target_depths.append(depth)
            sample_target_normals.append(normal)
            sample_target_w2cs.append(mvp)
            sample_target_camera_pos.append(campos)

        target_images.append(torch.stack(sample_target_images, dim=0).permute(0, 3, 1, 2))
        # target_albedos.append(torch.stack(sample_target_albedos, dim=0).permute(0, 3, 1, 2))
        # target_alphas.append(torch.stack(sample_target_alphas, dim=0).permute(0, 3, 1, 2))
        # target_depths.append(torch.stack(sample_target_depths, dim=0).permute(0, 3, 1, 2))
        # target_ccms.append(torch.stack(sample_target_ccms, dim=0).permute(0, 3, 1, 2))
        target_normals.append(torch.stack(sample_target_normals, dim=0).permute(0, 3, 1, 2))
        target_w2cs.append(torch.stack(sample_target_w2cs, dim=0))
        target_camera_pos.append(torch.stack(sample_target_camera_pos, dim=0))

        del ref_mesh
        del material
        del mesh_attributes
        torch.cuda.empty_cache()

    data = {
        'target_camera_embedding': torch.stack(target_cam_emebdding, dim=0),
        # 'target_albedos': torch.stack(target_albedos, dim=0).detach().cpu(), 
        'target_images': torch.stack(target_images, dim=0).detach().cpu(),         # (batch_size, target_view_num, 3, H, W)
        # 'target_alphas': torch.stack(target_alphas, dim=0).detach().cpu(),         # (batch_size, target_view_num, 1, H, W)
        # 'target_ccms': torch.stack(target_ccms, dim=0).detach().cpu(),  
        # 'target_depths': torch.stack(target_depths, dim=0).detach().cpu(),  
        'target_normals': torch.stack(target_normals, dim=0).detach().cpu(), 
    }

    return data
    

if __name__ == '__main__':
    dataset = ObjaverseData(root_dir="/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/Objaverse_highQuality_singleObj_texture_small_OBJ_Mesh_final",
                            light_dir="/hpc2hdd/JH_DATA/share/yingcongchen/PrivateShareGroup/yingcongchen_datasets/env_mipmap_large",
                            target_view_num=4,
                            fov=30,
                            camera_distance=5.0,
                            validation=True,
                            random_camera=False,
                            random_elevation=False,
                            )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    index = 0
    for batch in tqdm(dataloader):
        pass
        # batch, view, c, h, w
        # target_ccms = batch["target_ccms"][:,:,:3]
        # target_images = batch["target_images"]
        # target_depths = batch["target_depths"]
        # target_normals = batch["target_normals"]
        # target_albedos = batch["target_albedos"]
        # pass
        # torchvision.utils.save_image(target_ccms[2], f"debug_output/target_ccms_{index}.png", normalize=True)
        # torchvision.utils.save_image(target_images[2], f"debug_output/target_images_{index}.png", normalize=True)
        # torchvision.utils.save_image(target_depths[2], f"debug_output/target_depths_{index}.png", normalize=True)
        # torchvision.utils.save_image(target_normals[2], f"debug_output/target_normals_{index}.png", normalize=True)
        # torchvision.utils.save_image(target_albedos[2], f"debug_output/target_albedos_{index}.png", normalize=True)
        # breakpoint()
        # torchvision.utils.save_image(torch.cat([target_images[2], target_albedos[2]], dim=2), f"debug_output/target_images_albedos_{index}.png", normalize=True)
        # # exit()
        # index += 1
        # if index >= 10:
        #     exit()
