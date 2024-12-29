import numpy as np
import torch
from PIL import Image
from pytorch3d.renderer import (
    TexturesVertex,
)
from .project_mesh import (
    get_cameras_list_azi_ele,
    multiview_color_projection

)
from .utils import save_py3dmesh_with_trimesh_fast

def projection(meshes, 
               img_list,
               weights,
               azimuths, 
               elevations, 
               projection_type='orthographic',
               auto_center=True, 
               resolution=1024,
               fovy=None,
               radius=None,
               ortho_dist=1.1,
               scale_factor=1.0,
               save_glb_addr=None,
               scale_verts=True,
               complete_unseen=True,
               below_confidence_strategy="smooth"
               ):

    assert len(img_list) == len(azimuths) == len(elevations) == len(weights), f"len(img_list) ({len(img_list)}) != len(azimuths) ({len(azimuths)}) != len(elevations) ({len(elevations)}) != len(weights) ({len(weights)})"
    
    projection_types = ['perspective', 'orthographic']
    assert projection_type in projection_types, f"projection_type ({projection_type}) should be one of {projection_types}"

    if auto_center:
        verts = meshes.verts_packed()
        max_bb = (verts - 0).max(0)[0]
        min_bb = (verts - 0).min(0)[0]
        scale = (max_bb - min_bb).max() / 2
        center = (max_bb + min_bb) / 2
        meshes.offset_verts_(-center)
        if scale_verts:
            meshes.scale_verts_((scale_factor / float(scale))) 
    elif scale_verts:
        meshes.scale_verts_((scale_factor))

    if projection_type == 'perspective':
        assert fovy is not None and radius is not None, f"fovy ({fovy}) and radius ({radius}) should not be None when projection_type is 'perspective'"
        cameras = get_cameras_list_azi_ele(azimuths, elevations, fov_in_degrees=fovy,device="cuda", dist=radius, cam_type='fov')
    elif projection_type == 'orthographic':
        cameras = get_cameras_list_azi_ele(azimuths, elevations, fov_in_degrees=fovy, device="cuda", focal=2/1.35, dist=ortho_dist, cam_type='orthographic')


    num_meshes = len(meshes)
    num_verts_per_mesh = meshes.verts_packed().shape[0] // num_meshes
    black_texture = torch.zeros((num_meshes, num_verts_per_mesh, 3), device="cuda")
    textures = TexturesVertex(verts_features=black_texture)
    meshes.textures = textures


    proj_mesh = multiview_color_projection(meshes, img_list, cameras, weights=weights, eps=0.05, resolution=resolution, device="cuda", reweight_with_cosangle="square", use_alpha=True, confidence_threshold=0.1, complete_unseen=complete_unseen, below_confidence_strategy=below_confidence_strategy)


    if save_glb_addr is not None:
        save_py3dmesh_with_trimesh_fast(proj_mesh, save_glb_addr)
        

