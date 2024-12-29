import torch
import numpy as np
from PIL import Image
import os
from pytorch3d.io import load_obj
import trimesh
from pytorch3d.structures import Meshes
# from rembg import remove

def remove_color(arr):
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    
    # Convert to torch tensor
    if type(arr) is not torch.Tensor:
        arr = torch.tensor(arr, dtype=torch.int32)
    
    # Calculate diffs
    base = arr[0, 0]
    diffs = torch.abs(arr - base).sum(dim=-1)
    alpha = (diffs <= 80)
    
    arr[alpha] = 255
    alpha = ~alpha
    alpha = alpha.unsqueeze(-1).int() * 255
    arr = torch.cat([arr, alpha], dim=-1)
    
    return arr

def simple_remove_bkg_normal(imgs, rm_bkg_with_rembg, return_Image=False):
    """Only works for normal"""
    rets = []
    for img in imgs:
        if rm_bkg_with_rembg:
            from rembg import remove
            image = Image.fromarray(img.to(torch.uint8).detach().cpu().numpy())  if isinstance(img, torch.Tensor) else img
            removed_image = remove(image)
            arr = np.array(removed_image)
            arr = torch.tensor(arr, dtype=torch.uint8)
        else:
            arr = remove_color(img)

        if return_Image:
            rets.append(Image.fromarray(arr.to(torch.uint8).detach().cpu().numpy()))
        else:
            rets.append(arr.to(torch.uint8))
    
    return rets


def load_glb(file_path):
    # Load the .glb file as a scene and merge all meshes
    scene_or_mesh = trimesh.load(file_path)

    mesh = scene_or_mesh.dump(concatenate=True) if isinstance(scene_or_mesh, trimesh.Scene) else scene_or_mesh

    # Extract vertices and faces from the merged mesh
    verts = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(mesh.faces, dtype=torch.int64)
    
    
    textured_mesh = Meshes(verts=[verts], faces=[faces])


    return textured_mesh

def load_obj_with_verts_faces(file_path, return_mesh=True):
    verts, faces, _ = load_obj(file_path)
    
    verts = torch.tensor(verts, dtype=torch.float32)
    faces = faces.verts_idx 
    faces = torch.tensor(faces, dtype=torch.int64)

    if return_mesh:
        return Meshes(verts=[verts], faces=[faces])
    else:
        return verts, faces

def normalize_mesh(vertices):
    min_vals, _ = torch.min(vertices, axis=0)
    max_vals, _ = torch.max(vertices, axis=0)
    center = (max_vals + min_vals) / 2
    vertices = vertices - center
    max_extent = torch.max(max_vals - min_vals)
    scale = 2.0 / max_extent
    vertices = vertices * scale
    return vertices
