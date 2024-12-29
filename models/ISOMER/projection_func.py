import os
import numpy as np
import torch
from PIL import Image
import os
from .scripts.proj_commands import projection as isomer_projection
from .data.utils import simple_remove_bkg_normal

# mesh_address,
def projection(
    meshes,
    masks,
    images,
    azimuths,
    elevations,
    weights,
    fov,
    radius,
    save_dir,
    save_glb_addr=None,
    remove_background=False,
    auto_center=False,
    projection_type="perspective",
    below_confidence_strategy="smooth",
    complete_unseen=True,
    mesh_scale_factor=1.0,
    rm_bkg_with_rembg=True,
):
    
    if save_glb_addr is None:
        os.makedirs(save_dir, exist_ok=True)
        save_glb_addr=os.path.join(save_dir,  "rgb_projected.glb")

    bs = len(images)
    assert len(azimuths) == bs, f'len(azimuths) ({len(azimuths)} != batchsize ({bs}))'
    assert len(elevations) == bs, f'len(elevations) ({len(elevations)} != batchsize ({bs}))'
    assert len(weights) == bs, f'len(weights) ({len(weights)} != batchsize ({bs}))'
    
    image_rgba = torch.cat([images[:,:,:,:3], masks.unsqueeze(-1)], dim=-1)

    assert image_rgba.shape[-1] == 4, f'image_rgba.shape is {image_rgba.shape}'

    img_list = [Image.fromarray((image.cpu()*255).numpy().astype(np.uint8)) for image in image_rgba]


    if remove_background:
        if rm_bkg_with_rembg:
            os.environ["OMP_NUM_THREADS"] = '8'
        img_list = simple_remove_bkg_normal(img_list, rm_bkg_with_rembg, return_Image=True)

    resolution = img_list[0].size[0]
    new_img_list = []
    for i in range(len(img_list)): 
        new_img = img_list[i].resize((resolution,resolution))

        path_dir = os.path.join(save_dir, f'projection_images')
        os.makedirs(path_dir, exist_ok=True)
        
        path_ = os.path.join(path_dir, f'ProjectionImg{i}.png')

        new_img.save(path_)

        new_img_list.append(new_img)

    img_list = new_img_list
    
    isomer_projection(meshes, 
            img_list=img_list,
            weights=weights,
            azimuths=azimuths, 
            elevations=elevations, 
            projection_type=projection_type,
            auto_center=auto_center, 
            resolution=resolution,
            fovy=fov,
            radius=radius,
            scale_factor=mesh_scale_factor,
            save_glb_addr=save_glb_addr,
            scale_verts=True,
            complete_unseen=complete_unseen,
            below_confidence_strategy=below_confidence_strategy
            )

    return save_glb_addr


