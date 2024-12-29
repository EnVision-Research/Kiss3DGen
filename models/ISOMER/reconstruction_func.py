import os
import numpy as np
import torch
from PIL import Image
import os
from .model.inference_pipeline import reconstruction_pipe

def reconstruction(
    normal_pils, 
    masks, 
    weights, 
    fov, 
    radius, 
    camera_angles_azi, 
    camera_angles_ele, 
    expansion_weight_stage1=0.1,
    init_type="ball",
    init_verts=None,
    init_faces=None,
    init_mesh_from_file="",
    stage1_steps=200,
    stage2_steps=200,
    projection_type="perspective",
    need_normal_rotation=False,
    rotation_angles_azi=None, # only used if need_normal_rotation
    rotation_angles_ele=None, # only used if need_normal_rotation
    normal_rotation_R=None, # only used if need_normal_rotation
    rm_bkg=False,
    rm_bkg_with_rembg=True, # only used if rm_bkg
    start_edge_len_stage1=0.1,
    end_edge_len_stage1=0.02,
    start_edge_len_stage2=0.02,
    end_edge_len_stage2=0.005,
    expansion_weight_stage2=0.0,
):
    
    if init_type == "file":
        assert ((init_verts is not None and init_faces is not None) or init_mesh_from_file), f'init_mesh_from_file or (init_verts and init_faces) must be provided if init_type=="file"'

    if not need_normal_rotation:
        rotation_angles_azi = None
        rotation_angles_ele = None
        normal_rotation_R = None

    bs = len(normal_pils)

    assert len(camera_angles_azi) == bs, f'len(camera_angles_azi) ({len(camera_angles_azi)} != batchsize ({bs}))'
    assert len(camera_angles_ele) == bs, f'len(camera_angles_ele) ({len(camera_angles_ele)} != batchsize ({bs}))'

    normal_pils_rgba = torch.cat([normal_pils[:,:,:,:3], masks.unsqueeze(-1)], dim=-1)

    assert normal_pils_rgba.shape[-1] == 4, f'normal_pils_rgba.shape is {normal_pils_rgba.shape}'


    normal_pils = [Image.fromarray((normal_pil.cpu()*255).numpy().astype(np.uint8)) for normal_pil in normal_pils_rgba]


    meshes = reconstruction_pipe(
        normal_pils=normal_pils, 
        rotation_angles_azi=rotation_angles_azi, 
        rotation_angles_ele=rotation_angles_ele, 
        weights=weights, 
        expansion_weight=expansion_weight_stage1, 
        init_type=init_type, 
        stage1_steps=stage1_steps, 
        stage2_steps=stage2_steps, 
        projection_type=projection_type, 
        fovy=fov,  
        radius=radius,
        camera_angles_azi=camera_angles_azi, 
        camera_angles_ele=camera_angles_ele, 
        rm_bkg=rm_bkg, rm_bkg_with_rembg=rm_bkg_with_rembg,
        normal_rotation_R=normal_rotation_R, 
        init_mesh_from_file=init_mesh_from_file, 
        start_edge_len_stage1=start_edge_len_stage1, 
        end_edge_len_stage1=end_edge_len_stage1, 
        start_edge_len_stage2=start_edge_len_stage2, 
        end_edge_len_stage2=end_edge_len_stage2,
        expansion_weight_stage2=expansion_weight_stage2,
        init_verts=init_verts,
        init_faces=init_faces,

    )


    return meshes


