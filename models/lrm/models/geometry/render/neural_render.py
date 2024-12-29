# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION & AFFILIATES and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION & AFFILIATES is strictly prohibited.

import torch
import torch.nn.functional as F
import nvdiffrast.torch as dr
from . import Renderer
from . import util
from . import renderutils as ru
_FG_LUT = None


def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(
        attr.contiguous(), rast, attr_idx, rast_db=rast_db,
        diff_attrs=None if rast_db is None else 'all')


def xfm_points(points, matrix, use_python=True):
    '''Transform points.
    Args:
        points: Tensor containing 3D points with shape [minibatch_size, num_vertices, 3] or [1, num_vertices, 3]
        matrix: A 4x4 transform matrix with shape [minibatch_size, 4, 4]
        use_python: Use PyTorch's torch.matmul (for validation)
    Returns:
        Transformed points in homogeneous 4D with shape [minibatch_size, num_vertices, 4].
    '''
    out = torch.matmul(torch.nn.functional.pad(points, pad=(0, 1), mode='constant', value=1.0), torch.transpose(matrix, 1, 2))
    if torch.is_anomaly_enabled():
        assert torch.all(torch.isfinite(out)), "Output of xfm_points contains inf or NaN"
    return out


def dot(x, y):
    return torch.sum(x * y, -1, keepdim=True)


def compute_vertex_normal(v_pos, t_pos_idx):
    i0 = t_pos_idx[:, 0]
    i1 = t_pos_idx[:, 1]
    i2 = t_pos_idx[:, 2]

    v0 = v_pos[i0, :]
    v1 = v_pos[i1, :]
    v2 = v_pos[i2, :]

    face_normals = torch.cross(v1 - v0, v2 - v0)

    # Splat face normals to vertices
    v_nrm = torch.zeros_like(v_pos)
    v_nrm.scatter_add_(0, i0[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i1[:, None].repeat(1, 3), face_normals)
    v_nrm.scatter_add_(0, i2[:, None].repeat(1, 3), face_normals)

    # Normalize, replace zero (degenerated) normals with some default value
    v_nrm = torch.where(
        dot(v_nrm, v_nrm) > 1e-20, v_nrm, torch.as_tensor([0.0, 0.0, 1.0]).to(v_nrm)
    )
    v_nrm = F.normalize(v_nrm, dim=1)
    assert torch.all(torch.isfinite(v_nrm))

    return v_nrm


class NeuralRender(Renderer):
    def __init__(self, device='cuda', camera_model=None):
        super(NeuralRender, self).__init__()
        self.device = device
        self.ctx = dr.RasterizeCudaContext(device=device)
        self.projection_mtx = None
        self.camera = camera_model
        
    # ==============================================================================================
    #  pixel shader
    # ==============================================================================================
    # def shade(
    #         self,
    #         gb_pos,
    #         gb_geometric_normal,
    #         gb_normal,
    #         gb_tangent,
    #         gb_texc,
    #         gb_texc_deriv,
    #         view_pos,
    #     ):
        
    #     ################################################################################
    #     # Texture lookups
    #     ################################################################################
    #     breakpoint()
    #     # Separate kd into alpha and color, default alpha = 1
    #     alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    #     kd = kd[..., 0:3]

    #     ################################################################################
    #     # Normal perturbation & normal bend
    #     ################################################################################
  
    #     perturbed_nrm = None

    #     gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    #     ################################################################################
    #     # Evaluate BSDF
    #     ################################################################################

    #     assert 'bsdf' in material or bsdf is not None, "Material must specify a BSDF type"
    #     bsdf = material['bsdf'] if bsdf is None else bsdf
    #     if bsdf == 'pbr':
    #         if isinstance(lgt, light.EnvironmentLight):
    #             shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
    #         else:
    #             assert False, "Invalid light type"
    #     elif bsdf == 'diffuse':
    #         if isinstance(lgt, light.EnvironmentLight):
    #             shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=False)
    #         else:
    #             assert False, "Invalid light type"
    #     elif bsdf == 'normal':
    #         shaded_col = (gb_normal + 1.0)*0.5
    #     elif bsdf == 'tangent':
    #         shaded_col = (gb_tangent + 1.0)*0.5
    #     elif bsdf == 'kd':
    #         shaded_col = kd
    #     elif bsdf == 'ks':
    #         shaded_col = ks
    #     else:
    #         assert False, "Invalid BSDF '%s'" % bsdf
        
    #     # Return multiple buffers
    #     buffers = {
    #         'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
    #         'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1),
    #         'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1)
    #     }
    #     return buffers
        
    # ==============================================================================================
    #  Render a depth slice of the mesh (scene), some limitations:
    #  - Single mesh
    #  - Single light
    #  - Single material
    # ==============================================================================================
    def render_layer(
            self,
            rast,
            rast_deriv,
            mesh,
            view_pos,
            resolution,
            spp,
            msaa
        ):
 
        # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
        rast_out_s = rast
        rast_out_deriv_s = rast_deriv

        ################################################################################
        # Interpolate attributes
        ################################################################################

        # Interpolate world space position
        gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

        # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
        v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
        v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
        v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
        face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
        face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
        gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

        # Compute tangent space
        assert mesh.v_nrm is not None and mesh.v_tng is not None
        gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
        gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

        # Texture coordinate
        # assert mesh.v_tex is not None
        # gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)
        perturbed_nrm = None
        gb_normal = ru.prepare_shading_normal(gb_pos, view_pos[:,None,None,:], perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

        return gb_pos, gb_normal

    def render_mesh(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            mesh,
            camera_mv_bx4x4,
            camera_pos,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False
    ):
        assert not hierarchical_mask
        
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera
  
        # view_pos = torch.linalg.inv(mtx_in)[:, :3, 3]
        view_pos = camera_pos
        v_nrm = mesh.v_nrm  #compute_vertex_normal(mesh_v_pos_bxnx3[0], mesh_t_pos_idx_fx3.long())  # vertex normals in world coordinates

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes

        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd.repeat(v_pos.shape[0], 1, 1), v_pos], dim=-1)  # Concatenate the pos [org_pos, clip space pose for rasterization]
        
        layers = []
        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh.t_pos_idx.int(), [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_pos, gb_normal = self.render_layer(rast, db, mesh, view_pos, resolution, spp, msaa=False)

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)
 
        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., -2:-1]
        ori_mesh_feature = gb_feat[..., :-4]

        normal, _ = interpolate(v_nrm[None, ...], rast, mesh_t_pos_idx_fx3)
        normal = dr.antialias(normal.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        # normal = F.normalize(normal, dim=-1)
        # normal = torch.lerp(torch.zeros_like(normal), (normal + 1.0) / 2.0, hard_mask.float())      # black background
        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal, gb_normal
    
    def render_mesh_light(
            self,
            mesh_v_pos_bxnx3,
            mesh_t_pos_idx_fx3,
            mesh,
            camera_mv_bx4x4,
            mesh_v_feat_bxnxd,
            resolution=256,
            spp=1,
            device='cuda',
            hierarchical_mask=False
    ):
        assert not hierarchical_mask
        
        mtx_in = torch.tensor(camera_mv_bx4x4, dtype=torch.float32, device=device) if not torch.is_tensor(camera_mv_bx4x4) else camera_mv_bx4x4
        v_pos = xfm_points(mesh_v_pos_bxnx3, mtx_in)  # Rotate it to camera coordinates
        v_pos_clip = self.camera.project(v_pos)  # Projection in the camera
       
        v_nrm = compute_vertex_normal(mesh_v_pos_bxnx3[0], mesh_t_pos_idx_fx3.long())  # vertex normals in world coordinates

        # Render the image,
        # Here we only return the feature (3D location) at each pixel, which will be used as the input for neural render
        num_layers = 1
        mask_pyramid = None
        assert mesh_t_pos_idx_fx3.shape[0] > 0  # Make sure we have shapes
        mesh_v_feat_bxnxd = torch.cat([mesh_v_feat_bxnxd.repeat(v_pos.shape[0], 1, 1), v_pos], dim=-1)  # Concatenate the pos

        with dr.DepthPeeler(self.ctx, v_pos_clip, mesh_t_pos_idx_fx3, [resolution * spp, resolution * spp]) as peeler:
            for _ in range(num_layers):
                rast, db = peeler.rasterize_next_layer()
                gb_feat, _ = interpolate(mesh_v_feat_bxnxd, rast, mesh_t_pos_idx_fx3)

        hard_mask = torch.clamp(rast[..., -1:], 0, 1)
        antialias_mask = dr.antialias(
            hard_mask.clone().contiguous(), rast, v_pos_clip,
            mesh_t_pos_idx_fx3)

        depth = gb_feat[..., -2:-1]
        ori_mesh_feature = gb_feat[..., :-4]

        normal, _ = interpolate(v_nrm[None, ...], rast, mesh_t_pos_idx_fx3)
        normal = dr.antialias(normal.clone().contiguous(), rast, v_pos_clip, mesh_t_pos_idx_fx3)
        normal = F.normalize(normal, dim=-1)
        normal = torch.lerp(torch.zeros_like(normal), (normal + 1.0) / 2.0, hard_mask.float())      # black background

        return ori_mesh_feature, antialias_mask, hard_mask, rast, v_pos_clip, mask_pyramid, depth, normal
