# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import render_utils
from models.lrm.models.geometry.render import renderutils as ru
import numpy as np
from PIL import Image

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')


def get_mip(roughness):
    return torch.where(roughness < 1.0
                    , (torch.clamp(roughness, 0.04, 1.0) - 0.04) / (1.0 - 0.04) * (6 - 2)
                    , (torch.clamp(roughness, 1.0, 1.0) - 1.0) / (1.0 - 1.0) + 6 - 2)

def shade_with_env(gb_pos, gb_normal, kd, metallic, roughness, view_pos, run_n_view, env, metallic_gt, roughness_gt, use_material_gt=True, gt_render=False):

    #mask = mask[..., 0]
    view_pos = view_pos.expand(-1,  gb_pos.shape[1],  gb_pos.shape[2], -1)  #.reshape(1, 512, 10240, 3)
 
    wo = render_utils.safe_normalize(view_pos - gb_pos)

    #roughness = roughness.reshape(roughness.shape[0], roughness.shape[1], roughness.shape[1], run_n_view, 1)
    #metallic  = metallic.reshape(metallic.shape[0], metallic.shape[1], metallic.shape[1], run_n_view, 1)
    #kd = kd.reshape(kd.shape[0], kd.shape[1], kd.shape[1], run_n_view, 3)
    # if len(diffuse_light) != 10:
    #     diffuse_light = [diffuse_light[0] for _ in range(10)]
    #     specular_light = [specular_light[0] for _ in range(10)]

    #     metallic_gt = torch.zeros((8, metallic_gt.shape[1], metallic_gt.shape[2], 1)).cuda()
    #     roughness_gt =  torch.zeros((8, roughness_gt.shape[1], roughness_gt.shape[2], 1)).cuda()

    #if use_material_gt:
    spec_col  = (1.0 - metallic_gt)*0.04 + kd * metallic_gt
    diff_col  = kd * (1.0 - metallic_gt)
    # else:

    #     spec_col  = (1.0 - metallic)*0.04 + kd * metallic
    #     diff_col  = kd * (1.0 - metallic)
    
    nrmvec = gb_normal
    reflvec = render_utils.safe_normalize(render_utils.reflect(wo, nrmvec))
    
    prb_rendered_list = []
    pbr_specular_light_list = []
    pbr_diffuse_light_list = []
    # pbr_specular_color_list = []
    # pbr_diffuse_color_list = []
    for i in range(run_n_view):
        specular_light, diffuse_light = env[i]
        diffuse_light = diffuse_light.cuda()
        specular_light_new = []
        for split_specular_light in specular_light:
            specular_light_new.append(split_specular_light.cuda())
        specular_light = specular_light_new

        shaded_col = torch.ones((gb_pos.shape[1], gb_pos.shape[2], 3)).cuda()

        diffuse = dr.texture(diffuse_light[None, ...], nrmvec[i,:,:,:][None, ...].contiguous(), filter_mode='linear', boundary_mode='cube')
        diffuse_comp = diffuse * diff_col[i,:,:,:][None, ...]
        
        # Lookup FG term from lookup texture
        NdotV = torch.clamp(render_utils.dot(wo[i,:,:,:], nrmvec[i,:,:,:]), min=1e-4)
        fg_uv = torch.cat((NdotV, roughness_gt[i,:,:,:]), dim=-1)
        #if not hasattr(self, '_FG_LUT'):
        _FG_LUT = torch.as_tensor(np.fromfile('./models/lrm/data/irrmaps/bsdf_256_256.bin', dtype=np.float32).reshape(1, 256, 256, 2), dtype=torch.float32, device='cuda')
        fg_lookup = dr.texture(_FG_LUT, fg_uv[None, ...], filter_mode='linear', boundary_mode='clamp')
        # Roughness adjusted specular env lookup
        
        miplevel = get_mip(roughness_gt[i,:,:,:])
        miplevel = miplevel[None, ...]
        spec = dr.texture(specular_light[0][None, ...], reflvec[i,:,:,:][None, ...].contiguous(), mip=list(m[None, ...] for m in specular_light[1:]), mip_level_bias=miplevel[..., 0], filter_mode='linear-mipmap-linear', boundary_mode='cube')

        # Compute aggregate lighting
        reflectance = spec_col[i,:,:,:][None, ...] * fg_lookup[...,0:1] + fg_lookup[...,1:2]
        specular_comp = spec * reflectance
        #shaded_col += spec * reflectance
        shaded_col = (specular_comp[0] + diffuse_comp[0])
        
        prb_rendered_list.append(shaded_col)
        pbr_specular_light_list.append(spec[0])
        pbr_diffuse_light_list.append(diffuse[0])
      
        # pbr_specular_color_list.append(metallic_gt[i].repeat(1,1,3))
        # pbr_diffuse_color_list.append(roughness_gt[i].repeat(1,1,3))


    shaded_col_all = torch.stack(prb_rendered_list, dim=0)
    pbr_specular_light =  torch.stack(pbr_specular_light_list, dim=0)
    pbr_diffuse_light =  torch.stack(pbr_diffuse_light_list, dim=0)
    # pbr_specular_color = torch.stack(pbr_specular_color_list, dim=0)
    # pbr_diffuse_color = torch.stack(pbr_diffuse_color_list, dim=0)
    
    #shaded_col_all = shaded_col_all.reshape(shaded_col_all.shape[0], shaded_col_all.shape[1], shaded_col_all.shape[1]*run_n_view, 3)
    shaded_col_all = render_utils.rgb_to_srgb(shaded_col_all).clamp(0.,1.)
    pbr_specular_light = render_utils.rgb_to_srgb(pbr_specular_light).clamp(0.,1.)
    pbr_diffuse_light = render_utils.rgb_to_srgb(pbr_diffuse_light).clamp(0.,1.)
    # pbr_specular_color = render_utils.rgb_to_srgb(pbr_specular_color).clamp(0.,1.)
    # pbr_diffuse_color = render_utils.rgb_to_srgb(pbr_diffuse_color).clamp(0.,1.)
    
    return shaded_col_all, pbr_specular_light, pbr_diffuse_light #, pbr_specular_color, pbr_diffuse_color

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        env,
        planes,
        kd_fn,
        materials,
        material,
        mask,
        gt_render
    ):

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    resolution = gb_pos.shape[1]
    N_views = view_pos.shape[0]

    if planes is None:
        kd = material['kd'].sample(gb_texc, gb_texc_deriv)
        
        matellic_gt, roughness_gt =  (materials[0] * torch.ones(*kd.shape[:-1])).unsqueeze(-1).cuda(), (materials[1] * torch.ones(*kd.shape[:-1])).unsqueeze(-1).cuda()
        matellic, roughness = None, None
    else:
        # predict kd with MLP and interpolated feature
        gb_pos_interp, mask = [gb_pos], [mask]
        gb_pos_interp = [torch.cat([pos[i_view:i_view + 1] for i_view in range(N_views)], dim=2) for pos in gb_pos_interp]
        mask = [torch.cat([ma[i_view:i_view + 1] for i_view in range(N_views)], dim=2) for ma in mask]
        kd, matellic, roughness = kd_fn( planes[None,...], gb_pos_interp, mask[0])
        kd = torch.cat( [torch.cat([kd[i:i + 1, :, resolution * i_view: resolution * (i_view + 1)]for i_view in range(N_views)], dim=0) for i in range(len(kd))], dim=0)
        
        matellic_val = [x[0] for x in materials]
        roughness_val = [y[1] for y in materials]
 
        matellic_gt = torch.full((N_views, resolution, resolution, 1), fill_value=0, dtype=torch.float32)
        roughness_gt = torch.full((N_views, resolution, resolution, 1), fill_value=0, dtype=torch.float32)
        
        for i in range(len(matellic_gt)):
            matellic_gt[i, :, :, 0].fill_(matellic_val[i])
            roughness_gt[i, :, :, 0].fill_(roughness_val[i])

        matellic_gt = matellic_gt.cuda()
        roughness_gt = roughness_gt.cuda()
  
    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1]) 
    kd = kd[..., 0:3].clamp(0., 1.)

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    #if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
    perturbed_nrm = None

    gb_normal_ = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True)

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    shaded_col, spec_light, diff_light = shade_with_env(gb_pos, gb_normal_, kd, matellic, roughness, view_pos, N_views, env, matellic_gt, roughness_gt, use_material_gt=True, gt_render=gt_render)
    
    buffers = {
        'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
        'spec_light': torch.cat((spec_light, alpha), dim=-1),
        'diff_light': torch.cat((diff_light, alpha), dim=-1),
        'gb_normal' : torch.cat((gb_normal_, alpha), dim=-1),
        'normal'    : torch.cat((gb_normal, alpha), dim=-1),
        'albedo'    :  torch.cat((kd, alpha), dim=-1),
        # 'spec_albedo': torch.cat((spec_albedo, alpha), dim=-1),
        # 'diff_albedo': torch.cat((diff_albedo, alpha), dim=-1),
    }
    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        env,
        planes,
        kd_fn,
        materials,
        v_pos_clip,
        resolution,
        spp,
        msaa,
        gt_render
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = render_utils.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
        rast_out_deriv_s = render_utils.scale_img_nhwc(rast_deriv, resolution, mag='nearest', min='nearest') * spp
    else:
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
    face_normals = render_utils.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())

    # Compute tangent space
    assert mesh.v_nrm is not None and mesh.v_tng is not None
    gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())
    gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents

    # Texture coordinate
    assert mesh.v_tex is not None
    gb_texc, gb_texc_deriv = interpolate(mesh.v_tex[None, ...], rast_out_s, mesh.t_tex_idx.int(), rast_db=rast_out_deriv_s)
    
    # render depth
    depth = torch.linalg.norm(view_pos.expand_as(gb_pos) - gb_pos, dim=-1)
    
    mask = torch.clamp(rast[..., -1:], 0, 1)
    antialias_mask = dr.antialias(mask.clone().contiguous(), rast, v_pos_clip,mesh.t_pos_idx.int())

    ################################################################################
    # Shade
    ################################################################################

    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, view_pos, env, planes, kd_fn, materials, mesh.material, mask, gt_render)
    buffers['depth'] = torch.cat((depth.unsqueeze(-1).repeat(1,1,1,3), torch.ones_like(gb_pos[..., 0:1])), dim=-1 )
    # print(gb_pos.shape)
    buffers['ccm'] = torch.cat((gb_pos, torch.ones_like(gb_pos[..., 0:1])), dim=-1 )
    buffers['mask'] = torch.cat((antialias_mask.repeat(1,1,1,3), torch.ones_like(gb_pos[..., 0:1])), dim=-1 )
    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            buffers[key] = render_utils.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')

    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        env,
        planes,
        kd_fn,
        materials,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None, 
        gt_render   = False
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in reversed(layers):
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
        return accum

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    layers = []
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        for _ in range(num_layers):
            rast, db = peeler.rasterize_next_layer()
            layers += [(render_layer(rast, db, mesh, view_pos, env, planes, kd_fn, materials, v_pos_clip, resolution, spp, msaa, gt_render), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = render_utils.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.ones(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')
        background_black = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')
        
    # Composite layers front-to-back
    out_buffers = {}

    for key in layers[0][0].keys():
        if key == 'mask':
            accum = composite_buffer(key, layers, background_black, True)
        else:
            accum = composite_buffer(key, layers, background, True)

        # Downscale to framebuffer resolution. Use avg pooling 
        out_buffers[key] = render_utils.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], render_utils.safe_normalize(perturbed_nrm)
