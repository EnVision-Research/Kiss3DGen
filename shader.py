import torch
from pytorch3d.renderer.mesh.shader import ShaderBase
from pytorch3d.renderer import (
    SoftPhongShader,
)
from pytorch3d.renderer import BlendParams


class MultiOutputShader(ShaderBase):
    def __init__(self, device, cameras, lights, materials, ccm_scale=1.0, choices=None):
        super().__init__()
        self.device = device
        self.cameras = cameras
        self.lights = lights
        self.materials = materials
        self.ccm_scale = ccm_scale

        if choices is None:
            self.choices = ["rgb", "mask", "depth", "normal", "albedo", "ccm"]
        else:
            self.choices = choices
        blend_params = BlendParams(sigma=1e-4, gamma=1e-4)  
        self.phong_shader = SoftPhongShader(
            device=self.device,
            cameras=self.cameras,
            lights=self.lights,
            materials=self.materials,
            blend_params=blend_params
        )

    def forward(self, fragments, meshes, **kwargs):
        batch_size, H, W, _ = fragments.zbuf.shape
        output = {}

        if "rgb" in self.choices:
            rgb_images = self.phong_shader(fragments, meshes, **kwargs)
            rgb = rgb_images[..., :3]
            output["rgb"] = rgb
        
        if "mask" in self.choices:
            alpha = rgb_images[..., 3:4]
            mask = (alpha > 0).float()
            output["mask"] = mask
        
        if "albedo" in self.choices:
            albedo = meshes.sample_textures(fragments)
            output["albedo"] = albedo[..., 0, :]
        
        if "depth" in self.choices:
            depth = fragments.zbuf
            output["depth"] = depth

        if "normal" in self.choices:
            pix_to_face = fragments.pix_to_face[..., 0]
            bary_coords = fragments.bary_coords[..., 0, :]
            valid_mask = pix_to_face >= 0
            face_indices = pix_to_face[valid_mask]
            faces_packed = meshes.faces_packed()
            normals_packed = meshes.verts_normals_packed()
            face_vertex_normals = normals_packed[faces_packed[face_indices]] 
            bary = bary_coords.view(-1, 3)[valid_mask.view(-1)]
            interpolated_normals = (
                bary[..., 0:1] * face_vertex_normals[:, 0, :] +
                bary[..., 1:2] * face_vertex_normals[:, 1, :] +
                bary[..., 2:3] * face_vertex_normals[:, 2, :]
            )
            interpolated_normals = interpolated_normals / interpolated_normals.norm(dim=-1, keepdim=True)
            normal = torch.zeros(batch_size, H, W, 3, device=self.device)
            normal[valid_mask] = interpolated_normals
            output["normal"] = normal

        if "ccm" in self.choices:
            face_vertices = meshes.verts_packed()[meshes.faces_packed()]
            faces_at_pixels = face_vertices[fragments.pix_to_face]
            ccm = torch.sum(fragments.bary_coords.unsqueeze(-1) * faces_at_pixels, dim=-2)
            ccm = (ccm[..., 0, :] * self.ccm_scale + 1) / 2
            output["ccm"] = ccm

        return output