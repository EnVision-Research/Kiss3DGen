import os
from PIL import Image
from .mesh_init import build_mesh, calc_w_over_h, fix_border_with_pymeshlab_fast
from pytorch3d.structures import Meshes, join_meshes_as_scene
import numpy as np

import torch
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

def create_sphere(radius, device='cuda'):

    sphere_mesh = ico_sphere(3, device=device)  # Increase the subdivision level (e.g., 2) for higher resolution sphere
    sphere_mesh = sphere_mesh.scale_verts(radius)

    meshes = Meshes(verts=[sphere_mesh.verts_list()[0]], faces=[sphere_mesh.faces_list()[0]])
    return meshes


def create_box(width, length, height, device='cuda'):
    """
    Create a box mesh given the width, length, and height.
    
    Args:
        width (float): Width of the box.
        length (float): Length of the box.
        height (float): Height of the box.
        device (str): Device for the tensor operations, default is 'cuda'.
    
    Returns:
        Meshes: A PyTorch3D Meshes object representing the box.
    """
    # Define the 8 vertices of the box
    verts = torch.tensor([
        [-width / 2, -length / 2, -height / 2],
        [ width / 2, -length / 2, -height / 2],
        [ width / 2,  length / 2, -height / 2],
        [-width / 2,  length / 2, -height / 2],
        [-width / 2, -length / 2,  height / 2],
        [ width / 2, -length / 2,  height / 2],
        [ width / 2,  length / 2,  height / 2],
        [-width / 2,  length / 2,  height / 2]
    ], device=device)

    # Define the 12 triangles (faces) of the box using vertex indices
    faces = torch.tensor([
        [0, 1, 2], [0, 2, 3],  # Bottom face
        [4, 5, 6], [4, 6, 7],  # Top face
        [0, 1, 5], [0, 5, 4],  # Front face
        [1, 2, 6], [1, 6, 5],  # Right face
        [2, 3, 7], [2, 7, 6],  # Back face
        [3, 0, 4], [3, 4, 7]   # Left face
    ], device=device)

    # Create the Meshes object
    meshes = Meshes(verts=[verts], faces=[faces])
    
    return meshes


# stage 0 inital mesh estimation
def fast_geo(front_normal: Image.Image, back_normal: Image.Image, side_normal: Image.Image, clamp=0., init_type="std", return_depth_and_sep_mesh=False):
    
    import time
    assert front_normal.mode != "RGB"
    assert back_normal.mode != "RGB"
    assert side_normal.mode != "RGB"

    front_normal = front_normal.resize((192, 192))
    back_normal = back_normal.resize((192, 192))
    side_normal = side_normal.resize((192, 192))
    
    # build mesh with front back projection # ~3s
    side_w_over_h = calc_w_over_h(side_normal)
    mesh_front, depth_front = build_mesh(front_normal, front_normal, clamp_min=clamp, scale=side_w_over_h, init_type=init_type, return_depth=True)
    mesh_back, depth_back = build_mesh(back_normal, back_normal, is_back=True, clamp_min=clamp, scale=side_w_over_h, init_type=init_type, return_depth=True)
    meshes = join_meshes_as_scene([mesh_front, mesh_back])
    
    # poisson reconstruction which guarantees a smooth connection between meshes
    # and simplify into 2000 fewer faces
    meshes = fix_border_with_pymeshlab_fast(meshes, poissson_depth=6, simplification=2000)


    if return_depth_and_sep_mesh:
        return meshes, depth_front, depth_back, mesh_front, mesh_back
    return meshes