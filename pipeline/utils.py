import os
import sys
import logging

__workdir__ = '/'.join(os.path.abspath(__file__).split('/')[:-2])
sys.path.insert(0, __workdir__)

print(__workdir__)

import numpy as np
import torch
from torchvision.transforms import v2
from PIL import Image
import rembg

from models.lrm.online_render.render_single import load_mipmap
from models.lrm.utils.camera_util import get_zero123plus_input_cameras, get_custom_zero123plus_input_cameras, get_flux_input_cameras
from models.lrm.utils.render_utils import rotate_x, rotate_y
from models.lrm.utils.mesh_util import save_obj, save_obj_with_mtl
from models.lrm.utils.infer_util import remove_background, resize_foreground

from models.ISOMER.reconstruction_func import reconstruction
from models.ISOMER.projection_func import projection

from utils.tool import NormalTransfer, get_render_cameras_frames, get_background, get_render_cameras_video, render_frames, mask_fix

logging.basicConfig(
    level = logging.INFO
)
logger = logging.getLogger('kiss3d_wrapper')

OUT_DIR = './outputs'
TMP_DIR = './outputs/tmp'

os.makedirs(TMP_DIR, exist_ok=True)

def lrm_reconstruct(model, infer_config, images, 
                    name='', export_texmap=False,
                    input_camera_type='zero123',
                    render_3d_bundle_image=True,
                    render_azimuths=[270, 0, 90, 180], 
                    render_elevations=[5, 5, 5, 5], 
                    render_radius=4.15):
    """
    image: Tensor, shape (1, c, h, w)
    """
    
    mesh_path_idx = os.path.join(TMP_DIR, f'{name}_recon_from_{input_camera_type}.obj')

    device = images.device
    if input_camera_type == 'zero123':
        input_cameras = get_custom_zero123plus_input_cameras(batch_size=1, radius=3.5, fov=30).to(device)
    elif input_camera_type == 'kiss3d':
        input_cameras = get_flux_input_cameras(batch_size=1, radius=3.5, fov=30).to(device)
    else:
        raise NotImplementedError(f'Unexpected input camera type: {input_camera_type}')
    
    images = v2.functional.resize(images, 512, interpolation=3, antialias=True).clamp(0, 1)

    logger.info(f"==> Runing LRM reconstruction ...")
    planes = model.forward_planes(images, input_cameras)
    mesh_out = model.extract_mesh(
            planes,
            use_texture_map=export_texmap,
            **infer_config,
        )
    if export_texmap:
        vertices, faces, uvs, mesh_tex_idx, tex_map = mesh_out
        save_obj_with_mtl(
            vertices.data.cpu().numpy(),
            uvs.data.cpu().numpy(),
            faces.data.cpu().numpy(),
            mesh_tex_idx.data.cpu().numpy(),
            tex_map.permute(1, 2, 0).data.cpu().numpy(),
            mesh_path_idx,
        )
    else:
        vertices, faces, vertex_colors = mesh_out
        save_obj(vertices, faces, vertex_colors, mesh_path_idx)
    logger.info(f"Mesh saved to {mesh_path_idx}")

    if render_3d_bundle_image:
        assert render_azimuths is not None and render_elevations is not None and render_radius is not None
        render_azimuths = torch.Tensor(render_azimuths).to(device)
        render_elevations = torch.Tensor(render_elevations).to(device)
        
        render_size = infer_config.render_resolution
        ENV = load_mipmap("models/lrm/env_mipmap/6")
        materials = (0.0,0.9)
        all_mv, all_mvp, all_campos, identity_mv = get_render_cameras_frames(
            batch_size=1, 
            radius=render_radius, 
            azimuths=render_azimuths, 
            elevations=render_elevations,
            fov=30
        )
        frames, albedos, pbr_spec_lights, pbr_diffuse_lights, normals, alphas = render_frames(
            model, 
            planes, 
            render_cameras=all_mvp,
            camera_pos=all_campos,
            env=ENV,
            materials=materials,
            render_size=render_size, 
            render_mv = all_mv,
            local_normal=True,
            identity_mv=identity_mv,
        )
    else:
        normals = None
        frames = None
        albedos = None


    vertices = torch.from_numpy(vertices).to(device)
    faces = torch.from_numpy(faces).to(device)
    vertices = vertices @ rotate_x(np.pi / 2, device=device)[:3, :3]
    vertices = vertices @ rotate_y(np.pi / 2, device=device)[:3, :3]

    return vertices.cpu(), faces.cpu(), normals, frames, albedos

normal_transfer = NormalTransfer()

def local_normal_global_transform(local_normal_images,azimuths_deg,elevations_deg):
    if local_normal_images.min() >= 0:
        local_normal = local_normal_images.float() * 2 - 1
    else:
        local_normal = local_normal_images.float()
    global_normal = normal_transfer.trans_local_2_global(local_normal, azimuths_deg, elevations_deg, radius=4.5, for_lotus=False)
    global_normal[...,0] *= -1
    global_normal = (global_normal + 1) / 2
    global_normal = global_normal.permute(0, 3, 1, 2)
    return global_normal


def isomer_reconstruct(
        rgb_multi_view,
        normal_multi_view,
        multi_view_mask,
        vertices,
        faces,
        save_path=None,
        azimuths=[0, 90, 180, 270],
        elevations=[5, 5, 5, 5],
        geo_weights=[1, 0.9, 1, 0.9],
        color_weights=[1, 0.5, 1, 0.5],
        reconstruction_stage1_steps=10,
        reconstruction_stage2_steps=50,
        radius=4.5):

    device = rgb_multi_view.device
    to_tensor_ = lambda x: torch.Tensor(x).float().to(device)

    # local normal to global normal
    global_normal = local_normal_global_transform(normal_multi_view.permute(0, 2, 3, 1).cpu(), to_tensor_(azimuths), to_tensor_(elevations)).to(device)
    global_normal = global_normal * multi_view_mask + (1-multi_view_mask)

    global_normal = global_normal.permute(0,2,3,1)
    multi_view_mask = multi_view_mask.squeeze(1)
    rgb_multi_view = rgb_multi_view.permute(0,2,3,1)

    logger.info(f"==> Runing ISOMER reconstruction ...")
    meshes = reconstruction(
        normal_pils=global_normal, 
        masks=multi_view_mask, 
        weights=to_tensor_(geo_weights), 
        fov=30, 
        radius=radius, 
        camera_angles_azi=to_tensor_(azimuths), 
        camera_angles_ele=to_tensor_(elevations), 
        expansion_weight_stage1=0.1,
        init_type="file",
        init_verts=vertices,
        init_faces=faces,
        stage1_steps=reconstruction_stage1_steps,
        stage2_steps=reconstruction_stage2_steps,
        start_edge_len_stage1=0.1,
        end_edge_len_stage1=0.02,
        start_edge_len_stage2=0.02,
        end_edge_len_stage2=0.005,
    )

    multi_view_mask_proj = mask_fix(multi_view_mask, erode_dilate=-10, blur=5)


    logger.info(f"==> Runing ISOMER projection ...")
    save_glb_addr = projection(
        meshes,
        masks=multi_view_mask_proj.to(device),
        images=rgb_multi_view.to(device),
        azimuths=to_tensor_(azimuths), 
        elevations=to_tensor_(elevations), 
        weights=to_tensor_(color_weights),
        fov=30,
        radius=radius,
        save_dir=TMP_DIR,
        save_glb_addr=save_path
    )

    logger.info(f"==> Save mesh to {save_glb_addr} ...")
    return save_glb_addr


def to_rgb_image(maybe_rgba):
    assert isinstance(maybe_rgba, Image.Image)
    if maybe_rgba.mode == 'RGB':
        return maybe_rgba, None
    elif maybe_rgba.mode == 'RGBA':
        rgba = maybe_rgba
        img = np.random.randint(127, 128, size=[rgba.size[1], rgba.size[0], 3], dtype=np.uint8)
        img = Image.fromarray(img, 'RGB')
        img.paste(rgba, mask=rgba.getchannel('A'))
        return img, rgba.getchannel('A')
    else:
        raise ValueError("Unsupported image type.", maybe_rgba.mode)
    
rembg_session = rembg.new_session("u2net")
def preprocess_input_image(input_image):
    """
    input_image: PIL.Image
    output_image: PIL.Image, (3, 512, 512), mode = RGB, background = white
    """
    image = remove_background(to_rgb_image(input_image)[0], rembg_session, bgcolor=(255, 255, 255, 255))
    image = resize_foreground(image, ratio=0.85, pad_value=255)
    return to_rgb_image(image)[0]
    