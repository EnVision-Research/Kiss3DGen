from tqdm import tqdm
from PIL import Image
import torch
import numpy as np
from typing import List
from ..mesh_reconstruction.remesh import calc_vertex_normals
from ..mesh_reconstruction.opt import MeshOptimizer
from ..mesh_reconstruction.func import make_star_cameras_orthographic, make_star_cameras_orthographic_py3d
from ..mesh_reconstruction.render import NormalsRenderer, Pytorch3DNormalsRenderer
from ..scripts.project_mesh import multiview_color_projection, get_cameras_list
from ..scripts.utils import to_py3d_mesh, from_py3d_mesh, init_target

def run_mesh_refine(vertices, faces, pils: List[Image.Image], mv, proj, weights, cameras, steps=100, start_edge_len=0.02, end_edge_len=0.005, decay=0.99, update_normal_interval=10, update_warmup=10, return_mesh=True, process_inputs=True, process_outputs=True, use_remesh=True, loss_expansion_weight=0):

    if process_inputs:
        vertices = vertices * 2 / 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]
    
    poission_steps = []

    renderer = NormalsRenderer(mv,proj,list(pils[0].size))
    

    target_images = init_target(pils, new_bkgd=(0., 0., 0.)) # 4s
    
    opt = MeshOptimizer(vertices,faces, ramp=5, edge_len_lims=(end_edge_len, start_edge_len), local_edgelen=False, laplacian_weight=0.02)

    vertices = opt.vertices
    alpha_init = None

    mask = target_images[..., -1] < 0.5

    for i in tqdm(range(steps)):
        opt.zero_grad()
        opt._lr *= decay
        normals = calc_vertex_normals(vertices,faces)
        images = renderer.render(vertices,normals,faces)

        if alpha_init is None:
            alpha_init = images.detach()
        
        # update explicit target and render images for L_ET calculation
        if i < update_warmup or i % update_normal_interval == 0:
            with torch.no_grad():

                py3d_mesh = to_py3d_mesh(vertices, faces, normals)
                
                _, _, target_normal = from_py3d_mesh(multiview_color_projection(py3d_mesh, pils, cameras_list=cameras, weights=weights, confidence_threshold=0.1, complete_unseen=False, below_confidence_strategy='original', reweight_with_cosangle='linear'))

                target_normal = target_normal * 2 - 1
                target_normal = torch.nn.functional.normalize(target_normal, dim=-1)
                debug_images = renderer.render(vertices,target_normal,faces)

        d_mask = images[..., -1] > 0.5
        loss_debug_l2 = (images[..., :3][d_mask] - debug_images[..., :3][d_mask]).pow(2).mean()
        
        loss_alpha_target_mask_l2 = (images[..., -1][mask] - target_images[..., -1][mask]).pow(2).mean()
        
        loss = loss_debug_l2 + loss_alpha_target_mask_l2
        
        loss_oob = (vertices.abs() > 0.99).float().mean() * 10

        loss = loss + loss_oob


        # this loss_expand does not exist in original ISOMER. we add it here (but default loss_expansion_weight is 0)
        loss_expand = 0.5 * ((vertices+normals).detach() - vertices).pow(2).mean()
        loss += loss_expand * loss_expansion_weight

        loss.backward()
        opt.step()
        
        
        if use_remesh:
            vertices,faces = opt.remesh(poisson=(i in poission_steps))
    
    vertices, faces = vertices.detach(), faces.detach()
    
    if process_outputs:
        vertices = vertices / 2 * 1.35
        vertices[..., [0, 2]] = - vertices[..., [0, 2]]

    if return_mesh:
        return to_py3d_mesh(vertices, faces)
    else:
        return vertices, faces
