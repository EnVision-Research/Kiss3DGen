import pytorch3d
import torch
import imageio
import numpy as np
import os
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    AmbientLights,
    PerspectiveCameras, 
    RasterizationSettings, 
    look_at_view_transform,
    TexturesVertex,
    MeshRenderer, 
    Materials,
    MeshRasterizer, 
    SoftPhongShader, 
    PointLights
)
import trimesh
from tqdm import tqdm
from pytorch3d.transforms import RotateAxisAngle

from shader import MultiOutputShader

def _rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    return torch.where(f <= 0.0031308, f * 12.92, torch.pow(torch.clamp(f, 0.0031308), 1.0/2.4)*1.055 - 0.055)

def rgb_to_srgb(f: torch.Tensor) -> torch.Tensor:
    assert f.shape[-1] == 3 or f.shape[-1] == 4
    out = torch.cat((_rgb_to_srgb(f[..., 0:3]), f[..., 3:4]), dim=-1) if f.shape[-1] == 4 else _rgb_to_srgb(f)
    assert out.shape[0] == f.shape[0] and out.shape[1] == f.shape[1]
    return out

def render_video_from_obj(input_obj_path, output_video_path, num_frames=60, image_size=512, fps=15, device="cuda"):
    if not os.path.exists(input_obj_path):
        raise FileNotFoundError(f"Input OBJ file not found: {input_obj_path}")

    scene_data = trimesh.load(input_obj_path)

    if isinstance(scene_data, trimesh.Scene):
        mesh_data = trimesh.util.concatenate([geom for geom in scene_data.geometry.values()])
    else:
        mesh_data = scene_data

    if not hasattr(mesh_data, 'vertex_normals') or mesh_data.vertex_normals is None:
        mesh_data.compute_vertex_normals()

    vertices = torch.tensor(mesh_data.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh_data.faces, dtype=torch.int64, device=device)

    if mesh_data.visual.vertex_colors is None:
        vertex_colors = torch.ones_like(vertices)[None]
    else:
        vertex_colors = torch.tensor(mesh_data.visual.vertex_colors[:, :3], dtype=torch.float32)[None]
    textures = TexturesVertex(verts_features=vertex_colors)
    textures.to(device)
    mesh = pytorch3d.structures.Meshes(verts=[vertices], faces=[faces], textures=textures)

    lights = AmbientLights(ambient_color=((2.0,)*3,), device=device)
    # lights = PointLights(device=device, location=[[0.0, 0.0, 3.0]], ambient_color=[[0.5, 0.5, 0.5]], diffuse_color=[[1.0, 1.0, 1.0]])
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1,
    )

    frames = []
    camera_distance = 6.5
    elevs = 0.0
    center = (0.0, 0.0, 0.0)
    materials = Materials(
            device=device,
            diffuse_color=((1.0, 1.0, 1.0),),
            ambient_color=((1.0, 1.0, 1.0),),
            specular_color=((1.0, 1.0, 1.0),),
            shininess=0.0,
    )
        
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    for i in tqdm(range(num_frames)):
        azims = 360.0 * i / num_frames
        R, T = look_at_view_transform(
            dist=camera_distance,
            elev=elevs,
            azim=azims,
            at=(center,),
            degrees=True
        )

        # 手动设置相机的旋转矩阵
        cameras = PerspectiveCameras(device=device, R=R, T=T, focal_length=5.0)
        cameras.znear = 0.0001
        cameras.zfar = 10000000.0
        shader=MultiOutputShader(
                device=device,
                cameras=cameras,
                lights=lights,
                materials=materials,
                choices=["rgb", "mask", "normal", "albedo"]
            )

        renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
        render_result = renderer(mesh, cameras=cameras)

        render_result["albedo"] = rgb_to_srgb(render_result["albedo"]/255.0)*255.0
        rgb_image = render_result["albedo"] * render_result["mask"] + (1 - render_result["mask"]) * torch.ones_like(render_result["albedo"]) * 255.0
        normal_map = render_result["normal"]

        rgb = rgb_image[0, ..., :3].cpu().numpy()
        normal_map = torch.nn.functional.normalize(normal_map, dim=-1)  # Normal map
        normal_map = (normal_map + 1) / 2
        normal_map = normal_map * render_result["mask"] + (1 - render_result["mask"]) * torch.ones_like(render_result["normal"])
        normal = normal_map[0, ..., :3].cpu().numpy()  # Normal map
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        normal = np.clip(normal*255, 0, 255).astype(np.uint8)
        combined_image = np.concatenate((rgb, normal), axis=1)

        frames.append(combined_image)

    imageio.mimsave(output_video_path, frames, fps=fps)

    print(f"Video saved to {output_video_path}")

if __name__ == '__main__':
    input_obj_path = "354e2aee-091d-4dc6-bdb1-e09be5791218_isomer_recon_mesh.obj"
    output_video_path = "output.mp4"
    render_video_from_obj(input_obj_path, output_video_path)