import os
import numpy as np
import torch
from PIL import Image

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from ..scripts.fast_geo import fast_geo, create_sphere, create_box
from ..scripts.project_mesh import get_cameras_list_azi_ele
from ..mesh_reconstruction.recon import reconstruct_stage1
from ..mesh_reconstruction.refine import run_mesh_refine
from ..mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras_perspective

from ..data.utils import (
    simple_remove_bkg_normal, 
    load_glb,
    load_obj_with_verts_faces)
from ..scripts.utils import (
    to_pyml_mesh, 
    simple_clean_mesh, 
    normal_rotation_img2img_c2w, 
    rotate_normal_R, 
    get_rotation_matrix_azi_ele, 
    manage_elevation_azimuth)

@torch.enable_grad()
def reconstruction_pipe(normal_pils, 
                    rotation_angles_azi, 
                    rotation_angles_ele,
                    front_index=0,
                    back_index=2,
                    side_index=1,
                    weights=None,
                    expansion_weight=0.1, 
                    expansion_weight_stage2=0.0,
                    init_type="ball",
                    sphere_r=None, # only used if init_type=="ball"
                    box_width=1.0, # only used if init_type=="box"
                    box_length=1.0, # only used if init_type=="box"
                    box_height=1.0, # only used if init_type=="box"
                    init_verts=None,
                    init_faces=None,
                    init_mesh_from_file="",
                    stage1_steps=200,
                    stage2_steps=200,
                    projection_type="orthographic",
                    fovy=None,
                    radius=None,
                    ortho_dist=1.1,
                    camera_angles_azi=None,
                    camera_angles_ele=None,
                    rm_bkg=False,
                    rm_bkg_with_rembg=False, # only used if rm_bkg
                    normal_rotation_R=None,
                    train_stage1=True,
                    train_stage2=True,
                    use_remesh_stage1=True,
                    use_remesh_stage2=True,
                    start_edge_len_stage1=0.1,
                    end_edge_len_stage1=0.02,
                    start_edge_len_stage2=0.02,
                    end_edge_len_stage2=0.005,
                   ):

    assert projection_type in ['perspective', 'orthographic'], f"projection_type ({projection_type}) should be one of ['perspective', 'orthographic']"

    if stage1_steps == 0:
        train_stage1 = False
    if stage2_steps == 0:
        train_stage2 = False

    if normal_rotation_R is not None:
        assert normal_rotation_R.shape[-2] == 3 and normal_rotation_R.shape[-1] == 3
        assert len(normal_rotation_R.shape) == 2
        normal_rotation_R = normal_rotation_R.float()

    camera_angles_azi = camera_angles_azi.float()
    camera_angles_ele = camera_angles_ele.float()

    camera_angles_ele, camera_angles_azi =  manage_elevation_azimuth(camera_angles_ele, camera_angles_azi)

    if init_type in ["std", "thin"]:
        assert camera_angles_azi[front_index]%360==0, f"the camera_angles_azi associated with front image (index {front_index}) should be 0 not {camera_angles_azi[front_index]}"
        assert camera_angles_azi[back_index]%360==180, f"the camera_angles_azi associated with back image (index {back_index}) should be 180 not {camera_angles_azi[back_index]}"
        assert camera_angles_azi[side_index]%360==90, f"the camera_angles_azi associated with left side image (index {side_index}) should be 90, not {camera_angles_azi[back_index]}"

    if rm_bkg:
        if rm_bkg_with_rembg:
            os.environ["OMP_NUM_THREADS"] = '8'
        normal_pils = simple_remove_bkg_normal(normal_pils,rm_bkg_with_rembg)

    if rotation_angles_azi is not None:
        rotation_angles_azi = -rotation_angles_azi.float()
        rotation_angles_ele = rotation_angles_ele.float()

        rotation_angles_ele, rotation_angles_azi =  manage_elevation_azimuth(rotation_angles_ele, rotation_angles_azi)

        assert len(normal_pils) == len(rotation_angles_azi), f'len(normal_pils) ({len(normal_pils)}) != len(rotation_angles_azi) ({len(rotation_angles_azi)})'
        if rotation_angles_ele is None:
            rotation_angles_ele = [0] * len(normal_pils)
            
        normal_pils_rotated = []
        for i in range(len(normal_pils)):
            c2w_R = get_rotation_matrix_azi_ele(rotation_angles_azi[i], rotation_angles_ele[i])

            rotated_ = normal_rotation_img2img_c2w(normal_pils[i], c2w=c2w_R)
            normal_pils_rotated.append(rotated_)

        normal_pils = normal_pils_rotated
    
    if normal_rotation_R is not None:
        normal_pils_rotated = []
        for i in range(len(normal_pils)):
            rotated_ = rotate_normal_R(normal_pils[i], normal_rotation_R, save_addr="", device="cuda")
            normal_pils_rotated.append(rotated_)
        
        normal_pils = normal_pils_rotated

    normal_stg1 = [img for img in normal_pils]

    if init_type in ['thin', 'std']:
        front_ = normal_stg1[front_index]
        back_ = normal_stg1[back_index]
        side_ = normal_stg1[side_index]
        meshes, depth_front, depth_back, mesh_front, mesh_back = fast_geo(front_, back_, side_, init_type=init_type, return_depth_and_sep_mesh=True)

        
    elif init_type in ["ball", "box"]:

        if init_type == "ball":
            assert sphere_r is not None, f"sphere_r ({sphere_r}) should not be None when init_type is 'ball'"
            meshes = create_sphere(sphere_r)

        if init_type == "box":
            assert box_width is not None and box_length is not None and box_height is not None, f"box_width ({box_width}), box_length ({box_length}), and box_height ({box_height}) should not be None when init_type is 'box'"
            meshes = create_box(width=box_width, length=box_length, height=box_height)

        # add texture just in case
        num_meshes = len(meshes)
        num_verts_per_mesh = meshes.verts_packed().shape[0] // num_meshes
        black_texture = torch.zeros((num_meshes, num_verts_per_mesh, 3), device="cuda")
        textures = TexturesVertex(verts_features=black_texture)
        meshes.textures = textures

    elif init_type == "file":
        assert init_mesh_from_file or (init_verts is not None and init_faces is not None), f"init_mesh_from_file ({init_mesh_from_file}) should not be None when init_type is 'file', else init_verts and init_faces should not be None"

        if init_verts is not None and init_faces is not None:
            meshes = Meshes(verts=[init_verts], faces=[init_faces]).to('cuda')
        elif init_mesh_from_file.endswith('.glb'):
            meshes = load_glb(init_mesh_from_file).to('cuda')
        else:
            meshes = load_obj_with_verts_faces(init_mesh_from_file).to('cuda')

        # add texture just in case
        num_meshes = len(meshes)
        num_verts_per_mesh = meshes.verts_packed().shape[0] // num_meshes
        black_texture = torch.zeros((num_meshes, num_verts_per_mesh, 3), device="cuda")
        textures = TexturesVertex(verts_features=black_texture)
        meshes.textures = textures

    if projection_type == 'perspective':
        assert fovy is not None and radius is not None, f"fovy ({fovy}) and radius ({radius}) should not be None when projection_type is 'perspective'"
        cameras = get_cameras_list_azi_ele(camera_angles_azi, camera_angles_ele, fov_in_degrees=fovy,device="cuda", dist=radius, cam_type='fov')

    elif projection_type == 'orthographic':
        cameras = get_cameras_list_azi_ele(camera_angles_azi, camera_angles_ele, fov_in_degrees=fovy, device="cuda", focal=1., dist=ortho_dist, cam_type='orthographic')

    vertices, faces = meshes.verts_list()[0], meshes.faces_list()[0]
    
    render_camera_angles_azi = -camera_angles_azi
    render_camera_angles_ele = camera_angles_ele
    if projection_type == 'orthographic':
        mv, proj = make_star_cameras_orthographic(render_camera_angles_azi, render_camera_angles_ele)
    else:
        mv, proj = make_star_cameras_perspective(render_camera_angles_azi, render_camera_angles_ele, distance=radius, r=radius, fov=fovy, device='cuda')
    
    # stage 1
    if train_stage1:
        vertices, faces = reconstruct_stage1(normal_stg1, mv=mv, proj=proj, steps=stage1_steps, vertices=vertices, faces=faces, start_edge_len=start_edge_len_stage1, end_edge_len=end_edge_len_stage1, gain=0.05, return_mesh=False, loss_expansion_weight=expansion_weight, use_remesh=use_remesh_stage1)
    
    # stage 2
    if train_stage2:
        vertices, faces = run_mesh_refine(vertices, faces, normal_pils, mv=mv, proj=proj, weights=weights, steps=stage2_steps, start_edge_len=start_edge_len_stage2, end_edge_len=end_edge_len_stage2, decay=0.99, update_normal_interval=20, update_warmup=5, return_mesh=False, process_inputs=False, process_outputs=False, cameras=cameras, use_remesh=use_remesh_stage2, loss_expansion_weight=expansion_weight_stage2)

    meshes = simple_clean_mesh(to_pyml_mesh(vertices, faces), apply_smooth=True, stepsmoothnum=1, apply_sub_divide=True, sub_divide_threshold=0.25).to("cuda")

    return meshes
