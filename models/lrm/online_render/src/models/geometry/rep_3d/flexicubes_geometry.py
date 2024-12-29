# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import numpy as np
import os
import nvdiffrast.torch as dr
from . import Geometry
from .flexicubes import FlexiCubes # replace later
from .dmtet import sdf_reg_loss_batch
from . import mesh
import torch.nn.functional as F
from src.utils import render

def get_center_boundary_index(grid_res, device):
    v = torch.zeros((grid_res + 1, grid_res + 1, grid_res + 1), dtype=torch.bool, device=device)
    v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = True
    center_indices = torch.nonzero(v.reshape(-1))

    v[grid_res // 2 + 1, grid_res // 2 + 1, grid_res // 2 + 1] = False
    v[:2, ...] = True
    v[-2:, ...] = True
    v[:, :2, ...] = True
    v[:, -2:, ...] = True
    v[:, :, :2] = True
    v[:, :, -2:] = True
    boundary_indices = torch.nonzero(v.reshape(-1))
    return center_indices, boundary_indices

###############################################################################
#  Geometry interface
###############################################################################
class FlexiCubesGeometry(Geometry):
    def __init__(
            self, grid_res=64, scale=2.0, device='cuda', renderer=None,
            render_type='neural_render', args=None):
        super(FlexiCubesGeometry, self).__init__()
        self.grid_res = grid_res
        self.device = device
        self.args = args
        self.fc = FlexiCubes(device, weight_scale=0.5)
        self.verts, self.indices = self.fc.construct_voxel_grid(grid_res)
        if isinstance(scale, list):
            self.verts[:, 0] = self.verts[:, 0] * scale[0]
            self.verts[:, 1] = self.verts[:, 1] * scale[1]
            self.verts[:, 2] = self.verts[:, 2] * scale[1]
        else:
            self.verts = self.verts * scale
            
        all_edges = self.indices[:, self.fc.cube_edges].reshape(-1, 2)
        self.all_edges = torch.unique(all_edges, dim=0)

        # Parameters used for fix boundary sdf
        self.center_indices, self.boundary_indices = get_center_boundary_index(self.grid_res, device)
        self.renderer = renderer
        self.render_type = render_type
        self.ctx = dr.RasterizeCudaContext(device=device)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    @torch.no_grad()
    def map_uv(self, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda")
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='floor')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, N, rounding_mode='floor'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx
    
    def rotate_x(self, a, device=None):
        s, c = np.sin(a), np.cos(a)
        return torch.tensor([[1, 0, 0, 0], 
                            [0, c,-s, 0], 
                            [0, s, c, 0], 
                         [0, 0, 0, 1]], dtype=torch.float32, device=device)
    def rotate_z(self, a, device=None):
        s, c = np.sin(a), np.cos(a)
        return torch.tensor([[ c, -s, 0, 0],
                            [ s,  c, 0, 0],
                            [ 0,  0, 1, 0],
                            [ 0,  0, 0, 1]], dtype=torch.float32, device=device)
    def rotate_y(self, a, device=None):
        s, c = np.sin(a), np.cos(a)
        return torch.tensor([[ c, 0,  s, 0],
                            [ 0, 1,  0, 0],
                            [-s, 0,  c, 0],
                            [ 0, 0,  0, 1]], dtype=torch.float32, device=device)


    def get_mesh(self, v_deformed_nx3, sdf_n, weight_n=None, with_uv=False, indices=None, is_training=False):
        if indices is None:
            indices = self.indices

        verts, faces, v_reg_loss = self.fc(v_deformed_nx3, sdf_n, indices, self.grid_res,
                                            beta_fx12=weight_n[:, :12], alpha_fx8=weight_n[:, 12:20],
                                            gamma_f=weight_n[:, 20], training=is_training
                                            )
        
        face_gidx = torch.arange(faces.shape[0], dtype=torch.long, device="cuda")
        uvs, uv_idx = self.map_uv(face_gidx, faces.shape[0])
        # breakpoint()
        
        verts = verts @ self.rotate_x(np.pi / 2, device=verts.device)[:3,:3]
        verts = verts @ self.rotate_y(np.pi / 2, device=verts.device)[:3,:3]



        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)
        
        return verts, faces, v_reg_loss, imesh


    # def render_mesh(self, mesh_v_nx3, mesh_f_fx3, mesh, camera_mv_bx4x4, camera_pos, resolution=256, hierarchical_mask=False):
    #     return_value = dict()
    #     if self.render_type == 'neural_render':
    #         tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal, gb_normal = self.renderer.render_mesh(
    #             mesh_v_nx3.unsqueeze(dim=0),
    #             mesh_f_fx3.int(),
    #             mesh,
    #             camera_mv_bx4x4,
    #             camera_pos,
    #             mesh_v_nx3.unsqueeze(dim=0),
    #             resolution=resolution,
    #             device=self.device,
    #             hierarchical_mask=hierarchical_mask
    #         )

    #         return_value['tex_pos'] = tex_pos
    #         return_value['mask'] = mask
    #         return_value['hard_mask'] = hard_mask
    #         return_value['rast'] = rast
    #         return_value['v_pos_clip'] = v_pos_clip
    #         return_value['mask_pyramid'] = mask_pyramid
    #         return_value['depth'] = depth
    #         return_value['normal'] = normal
    #         return_value['gb_normal'] = gb_normal
    #     else:
    #         raise NotImplementedError

    #     return return_value
    
    def render_mesh(self, mesh_v_nx3, mesh_f_fx3, mesh, camera_mv_bx4x4, camera_pos, env, planes, kd_fn, materials, resolution=256, hierarchical_mask=False):
        return_value = dict()
        # if self.render_type == 'neural_render':
        #     tex_pos, mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal, gb_normal = self.renderer.render_mesh(
        #         mesh_v_nx3.unsqueeze(dim=0),
        #         mesh_f_fx3.int(),
        #         mesh,
        #         camera_mv_bx4x4,
        #         camera_pos,
        #         mesh_v_nx3.unsqueeze(dim=0),
        #         resolution=resolution,
        #         device=self.device,
        #         hierarchical_mask=hierarchical_mask
        #     )

        #     return_value['tex_pos'] = tex_pos
        #     return_value['mask'] = mask
        #     return_value['hard_mask'] = hard_mask
        #     return_value['rast'] = rast
        #     return_value['v_pos_clip'] = v_pos_clip
        #     return_value['mask_pyramid'] = mask_pyramid
        #     return_value['depth'] = depth
        #     return_value['normal'] = normal
        #     return_value['gb_normal'] = gb_normal
        # else:
        #     raise NotImplementedError
        buffer_dict = render.render_mesh(self.ctx, mesh, camera_mv_bx4x4, camera_pos, env, planes, kd_fn, materials, [resolution, resolution], spp=1, num_layers=1, msaa=True, background=None)

        return buffer_dict
    

    def render(self, v_deformed_bxnx3=None, sdf_bxn=None, camera_mv_bxnviewx4x4=None, resolution=256):
        # Here I assume a batch of meshes (can be different mesh and geometry), for the other shapes, the batch is 1
        v_list = []
        f_list = []
        n_batch = v_deformed_bxnx3.shape[0]
        all_render_output = []
        for i_batch in range(n_batch):
            verts_nx3, faces_fx3 = self.get_mesh(v_deformed_bxnx3[i_batch], sdf_bxn[i_batch])
            v_list.append(verts_nx3)
            f_list.append(faces_fx3)
            render_output = self.render_mesh(verts_nx3, faces_fx3, camera_mv_bxnviewx4x4[i_batch], resolution)
            all_render_output.append(render_output)

        # Concatenate all render output
        return_keys = all_render_output[0].keys()
        return_value = dict()
        for k in return_keys:
            value = [v[k] for v in all_render_output]
            return_value[k] = value
            # We can do concatenation outside of the render
        return return_value
