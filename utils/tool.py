import rembg
import cv2
import numpy as np
import glm
import torch
from tqdm import tqdm
import torchvision
import torchvision.transforms.v2 as T
from models.lrm.utils import render_utils
import os
# get the background of the image
import torch
import numpy as np
import scipy
import cv2
from rembg import remove


def load_mipmap(env_path):
    diffuse_path = os.path.join(env_path, "diffuse.pth")
    diffuse = torch.load(diffuse_path, map_location=torch.device('cpu'))

    specular = []
    for i in range(6):
        specular_path = os.path.join(env_path, f"specular_{i}.pth")
        specular_tensor = torch.load(specular_path, map_location=torch.device('cpu'))
        specular.append(specular_tensor)
    return [specular, diffuse]

def get_background(img_tensor):
    """
    Args:
        img_tensor: 输入图像张量，形状为 (B, 3, H, W)，数值范围为 [0, 1] 或 [0, 255]。
    Returns:
        mask_tensor: 输出掩码张量，形状为 (B, 1, H, W)，二值化。
    """
    B, C, H, W = img_tensor.shape
    assert C == 3, "Input tensor must have 3 channels (RGB)."
    
    # 将 tensor 转换为 numpy 格式 (B, H, W, C)，并归一化到 [0, 255]
    img_numpy = (img_tensor.permute(0, 2, 3, 1) * 255).byte().cpu().numpy()  # (B, H, W, C)
    
    masks = []
    for i in range(B):
        # 调用 rembg 生成掩码
        mask = remove(img_numpy[i], only_mask=True)
        
        # 转换为二值掩码
        mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
        
        # 添加到结果列表 (H, W, 1)
        masks.append(mask_binary[..., None])
    
    # 将所有掩码组合成 numpy 数组，形状为 (B, H, W, 1)
    masks = np.stack(masks, axis=0)
    
    # 转换为 PyTorch 张量，形状为 (B, 1, H, W)，值为 {0, 1}
    mask_tensor = torch.from_numpy(masks).permute(0, 3, 1, 2).float() / 255.0
    # breakpoint()
    return mask_tensor

def get_render_cameras_video(batch_size=1, M=120, radius=4.0, elevation=20.0, is_flexicubes=False, fov=50):
    """
    Get the rendering camera parameters.
    """
    train_res = [512, 512]
    cam_near_far = [0.1, 1000.0]
    fovy = np.deg2rad(fov)
    proj_mtx = render_utils.perspective(fovy, train_res[1] / train_res[0], cam_near_far[0], cam_near_far[1])
    all_mv = []
    all_mvp = []
    all_campos = []
    if isinstance(elevation, tuple):
        elevation_0 = np.deg2rad(elevation[0])
        elevation_1 = np.deg2rad(elevation[1])
        for i in range(M//2):
            azimuth = 2 * np.pi * i / (M // 2)
            z = radius * np.cos(azimuth) * np.sin(elevation_0)
            x = radius * np.sin(azimuth) * np.sin(elevation_0)
            y = radius * np.cos(elevation_0)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())
        for i in range(M//2):
            azimuth = 2 * np.pi * i / (M // 2)
            z = radius * np.cos(azimuth) * np.sin(elevation_1)
            x = radius * np.sin(azimuth) * np.sin(elevation_1)
            y = radius * np.cos(elevation_1)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())
    else:
        # elevation = 90 - elevation
        for i in range(M):
            azimuth = 2 * np.pi * i / M
            z = radius * np.cos(azimuth) * np.sin(elevation)
            x = radius * np.sin(azimuth) * np.sin(elevation)
            y = radius * np.cos(elevation)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())
    all_mv = torch.stack(all_mv, dim=0).unsqueeze(0).squeeze(2)
    all_mvp = torch.stack(all_mvp, dim=0).unsqueeze(0).squeeze(2)
    all_campos = torch.stack(all_campos, dim=0).unsqueeze(0).squeeze(2)
    return all_mv, all_mvp, all_campos

def get_render_cameras_frames(batch_size=1, radius=4.0, azimuths=0, elevations=20.0, fov=30):
    """
    Get the rendering camera parameters.
    """
    train_res = [512, 512]
    cam_near_far = [0.1, 1000.0]
    fovy = np.deg2rad(fov)
    proj_mtx = render_utils.perspective(fovy, train_res[1] / train_res[0], cam_near_far[0], cam_near_far[1])
    all_mv = []
    all_mvp = []
    all_campos = []
    elevations = 90 - elevations
    if isinstance(elevations, np.ndarray) or isinstance(elevations, torch.Tensor):
        if isinstance(elevations, torch.Tensor):
            elevations = elevations.cpu().numpy()
        if isinstance(azimuths, torch.Tensor):
            azimuths = azimuths.cpu().numpy()
        azimuths = np.deg2rad(azimuths)
        elevations = np.deg2rad(elevations)
        for azi, ele in zip(azimuths, elevations):
            z = radius * np.cos(azi) * np.sin(ele)
            x = radius * np.sin(azi) * np.sin(ele)
            y = radius * np.cos(ele)

            eye = glm.vec3(x, y, z)
            at = glm.vec3(0.0, 0.0, 0.0)
            up = glm.vec3(0.0, 1.0, 0.0)
            view_matrix = glm.lookAt(eye, at, up)
            mv = torch.from_numpy(np.array(view_matrix))
            mvp   = proj_mtx @ (mv)  #w2c
            campos = torch.linalg.inv(mv)[:3, 3]
            all_mv.append(mv[None, ...].cuda())
            all_mvp.append(mvp[None, ...].cuda())
            all_campos.append(campos[None, ...].cuda())

    else:
        z = radius * np.cos(azimuths) * np.sin(elevations)
        x = radius * np.sin(azimuths) * np.sin(elevations)
        y = radius * np.cos(elevations)

        eye = glm.vec3(x, y, z)
        at = glm.vec3(0.0, 0.0, 0.0)
        up = glm.vec3(0.0, 1.0, 0.0)
        view_matrix = glm.lookAt(eye, at, up)
        mv = torch.from_numpy(np.array(view_matrix))
        mvp   = proj_mtx @ (mv)  #w2c
        campos = torch.linalg.inv(mv)[:3, 3]
        all_mv.append(mv[None, ...].cuda())
        all_mvp.append(mvp[None, ...].cuda())
        all_campos.append(campos[None, ...].cuda())
    
    # TODO, identity pose
    identity_azimuths = np.array([0])
    identity_elevations = np.array([90])
    z = radius * np.cos(identity_azimuths) * np.sin(identity_elevations)
    x = radius * np.sin(identity_azimuths) * np.sin(identity_elevations)
    y = radius * np.cos(identity_elevations)
    eye = glm.vec3(x, y, z)
    at = glm.vec3(0.0, 0.0, 0.0)
    up = glm.vec3(0.0, 1.0, 0.0)
    view_matrix = glm.lookAt(eye, at, up)
    identity_mv = torch.from_numpy(np.array(view_matrix))

    all_mv = torch.stack(all_mv, dim=0).unsqueeze(0).squeeze(2)
    all_mvp = torch.stack(all_mvp, dim=0).unsqueeze(0).squeeze(2)
    all_campos = torch.stack(all_campos, dim=0).unsqueeze(0).squeeze(2)
    return all_mv, all_mvp, all_campos, identity_mv


def worldNormal2camNormal(rot_w2c, normal_map_world):
    H,W,_ = normal_map_world.shape
    # normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])
    normal_map_world = normal_map_world[...,:3]
    # faster version
    normal_map_flat = normal_map_world.view(-1, 3)
    normal_map_camera_flat = torch.matmul(normal_map_flat.float(), rot_w2c.T.float())

    # Reshape the transformed normal map back to its original shape
    normal_map_camera = normal_map_camera_flat.view(normal_map_world.shape)

    return normal_map_camera


def trans_normal(normal, RT_w2c, RT_w2c_target):

    # normal_world = camNormal2worldNormal(np.linalg.inv(RT_w2c[:3,:3]), normal)
    # normal_target_cam = worldNormal2camNormal(RT_w2c_target[:3,:3], normal_world)

    relative_RT = torch.matmul(RT_w2c_target[:3,:3], torch.linalg.inv(RT_w2c[:3,:3]))
    normal_target_cam = worldNormal2camNormal(relative_RT[:3,:3], normal)

    return normal_target_cam

def render_frames(model, planes, render_cameras, camera_pos, env, materials, render_size=512, chunk_size=1, 
                  is_flexicubes=False, render_mv=None, local_normal=False, identity_mv=None):
    """
    Render frames from triplanes.
    """
    frames = []
    albedos = []
    pbr_spec_lights = []
    pbr_diffuse_lights = []
    normals = []
    alphas = []
    for i in tqdm(range(0, render_cameras.shape[1])):
        out = model.forward_geometry(
            planes,
            render_cameras[:, i:i+chunk_size],
            camera_pos[:, i:i+chunk_size],
            [[env]*chunk_size],
            [[materials]*chunk_size],
            render_size=render_size,
        )
        frame = out['pbr_img']
        albedo = out['albedo']
        pbr_spec_light = out['pbr_spec_light']
        pbr_diffuse_light = out['pbr_diffuse_light']
        normal = out['normal']
        alpha = out['mask']
        # breakpoint()
        if local_normal:
            # TODO global normal to local
            target_w2c = render_mv[0,i,:3,:3]
            identity_w2c = identity_mv[:3,:3]
            # breakpoint()
            # torchvision.utils.save_image((normal.permute(0,3,1,2)+1)/2, f"debug_output/global_normal.png")
            normal = trans_normal(normal.squeeze(0), identity_w2c.cuda(), target_w2c.cuda())
            normal = normal / torch.norm(normal, dim=-1, keepdim=True)
            # torchvision.utils.save_image((normal.permute(2,0,1)+1)/2, f"debug_output/local_normal.png")
            background_normal = torch.tensor([1,1,1], dtype=torch.float32, device=normal.device)
            normal = normal.unsqueeze(0)
            normal[...,0] *= -1
            # breakpoint()
            normal = normal * alpha.squeeze(0).permute(0,2,3,1) + background_normal * (1-alpha.squeeze(0).permute(0,2,3,1))
        frames.append(frame)
        albedos.append(albedo)
        pbr_spec_lights.append(pbr_spec_light)
        pbr_diffuse_lights.append(pbr_diffuse_light)
        normals.append(normal)
        alphas.append(alpha)

    frames = torch.cat(frames, dim=1)[0]    # we suppose batch size is always 1
    alphas = torch.cat(alphas, dim=1)[0]    
    albedos = torch.cat(albedos, dim=1)[0]
    pbr_spec_lights = torch.cat(pbr_spec_lights, dim=1)[0]
    pbr_diffuse_lights = torch.cat(pbr_diffuse_lights, dim=1)[0]
    normals = torch.cat(normals, dim=0).permute(0,3,1,2)[:,:3]
    return frames, albedos, pbr_spec_lights, pbr_diffuse_lights, normals, alphas

# from https://github.com/cubiq/ComfyUI_essentials
def mask_fix(mask, erode_dilate=0, smooth=0, remove_isolated_pixels=0, blur=0, fill_holes=0):
    masks = []
    for m in mask:
        # erode and dilate
        if erode_dilate != 0:
            if erode_dilate < 0:
                m = torch.from_numpy(scipy.ndimage.grey_erosion(m.cpu().numpy(), size=(-erode_dilate, -erode_dilate)))
            else:
                m = torch.from_numpy(scipy.ndimage.grey_dilation(m.cpu().numpy(), size=(erode_dilate, erode_dilate)))

        # fill holes
        if fill_holes > 0:
            #m = torch.from_numpy(scipy.ndimage.binary_fill_holes(m.cpu().numpy(), structure=np.ones((fill_holes,fill_holes)))).float()
            m = torch.from_numpy(scipy.ndimage.grey_closing(m.cpu().numpy(), size=(fill_holes, fill_holes)))

        # remove isolated pixels
        if remove_isolated_pixels > 0:
            m = torch.from_numpy(scipy.ndimage.grey_opening(m.cpu().numpy(), size=(remove_isolated_pixels, remove_isolated_pixels)))

        # smooth the mask
        if smooth > 0:
            if smooth % 2 == 0:
                smooth += 1
            m = T.functional.gaussian_blur((m > 0.5).unsqueeze(0), smooth).squeeze(0)

        # blur the mask
        if blur > 0:
            if blur % 2 == 0:
                blur += 1
            m = T.functional.gaussian_blur(m.float().unsqueeze(0), blur).squeeze(0)

        masks.append(m.float())

    masks = torch.stack(masks, dim=0).float()

    return masks


class NormalTransfer:
    def __init__(self):
        self.identity_w2c = torch.tensor([
                                        [0.0,  0.0,  1.0,  0.0],
                                        [ 0.0,  1.0,  0.0,  0.0],
                                        [-1.0, 0.0,  0.0,  4.5]]).float()

    def look_at(self,camera_position, target_position, up_vector=np.array([0, 0, 1])):
        forward = camera_position - target_position
        forward = forward / np.linalg.norm(forward)

        right = np.cross(up_vector, forward)
        right = right / np.linalg.norm(right)

        up = np.cross(forward, right)

        rotation_matrix = np.array([right, up, forward]).T

        translation_matrix = np.eye(4)
        translation_matrix[:3, 3] = -camera_position

        rotation_homogeneous = np.eye(4)
        rotation_homogeneous[:3, :3] = rotation_matrix

        w2c = rotation_homogeneous @ translation_matrix
        return w2c
    
    def generate_target_pose(self, azimuths_deg, elevations_deg, radius=4.5):
        if isinstance(azimuths_deg, torch.Tensor):
            azimuths_deg = azimuths_deg.cpu().numpy()
        if isinstance(elevations_deg, torch.Tensor):
            elevations_deg = elevations_deg.cpu().numpy()
        azimuths = np.deg2rad(azimuths_deg)
        elevations = np.deg2rad(elevations_deg)

        x = radius * np.cos(azimuths) * np.cos(elevations)
        y = radius * np.sin(azimuths) * np.cos(elevations)
        z = radius * np.sin(elevations)
        camera_positions = np.stack([x, y, z], axis=-1)

        target_position = np.array([0, 0, 0])  # 目标点位置

        # 为每个相机位置生成 w2c 矩阵
        w2c_matrices = [self.look_at(cam_pos, target_position) for cam_pos in camera_positions]
        w2c_matrices = np.stack(w2c_matrices, axis=0)
        return w2c_matrices
    
    def convert_to_blender(self, pose):
        # Swap the y and z axes
        w2c_opengl = pose
        w2c_opengl[[1, 2], :] = w2c_opengl[[2, 1], :]
        
        # Invert the y axis
        w2c_opengl[1] *= -1
        R = w2c_opengl[:3, :3]
        t = w2c_opengl[:3, 3]

        cam_rec = np.asarray([[1, 0, 0], [0, -1, 0], [0, 0, -1]], np.float32)
        R = R.T
        t = -R @ t
        R_world2cv = cam_rec @ R
        t_world2cv = cam_rec @ t

        RT = np.concatenate([R_world2cv,t_world2cv[:,None]],1)
        return RT

    def worldNormal2camNormal(self, rot_w2c, normal_map_world):
        H,W,_ = normal_map_world.shape
        # normal_img = np.matmul(rot_w2c[None, :, :], worldNormal.reshape(-1,3)[:, :, None]).reshape([H, W, 3])
        normal_map_world = normal_map_world[...,:3]
        # faster version
        normal_map_flat = normal_map_world.contiguous().view(-1, 3)

        normal_map_camera_flat = torch.matmul(normal_map_flat.float(), rot_w2c.T.float())

        # Reshape the transformed normal map back to its original shape
        normal_map_camera = normal_map_camera_flat.view(normal_map_world.shape)

        return normal_map_camera

    def trans_normal(self, normal, RT_w2c, RT_w2c_target):
        """
        :param normal: (H,W,3), torch tensor, range [-1,1]
        :param RT_w2c: (4,4), torch tensor, world to camera
        :param RT_w2c_target: (4,4), torch tensor, world to camera
        :return: normal_target_cam: (H,W,3), torch tensor, range [-1,1]
        """
        relative_RT = torch.matmul(RT_w2c_target[:3,:3], torch.linalg.inv(RT_w2c[:3,:3]))
        normal_target_cam = self.worldNormal2camNormal(relative_RT[:3,:3], normal)

        return normal_target_cam

    def trans_local_2_global(self, normal_local, azimuths_deg, elevations_deg, radius=4.5, for_lotus=True):
        """
        :param normal_local: (B,H,W,3), torch tensor, range [-1,1]
        :param azimuths_deg: (B,), numpy array, range [0,360]
        :param elevations_deg: (B,), numpy array, range [-90,90]
        :param radius: float, default 4.5
        :return: global_normal: (B,H,W,3), torch tensor, range [-1,1]

        """
        # print(f"normal_local.shape:{normal_local.shape}")
        # print(f"azimuths_deg.shape:{azimuths_deg.shape}")
        # print(f"elevations_deg.shape:{elevations_deg.shape}")
        assert normal_local.shape[0] == azimuths_deg.shape[0] == elevations_deg.shape[0]
        identity_w2c = self.identity_w2c

        # generate target pose
        target_w2c = self.generate_target_pose(azimuths_deg, elevations_deg, radius)
        target_w2c = torch.from_numpy(np.stack([self.convert_to_blender(w2c) for w2c in target_w2c])).float()
        global_normal = []

        # transform normal
        for i in range(normal_local.shape[0]):
            normal_local_i = normal_local[i]
            normal_zero123 = self.trans_normal(normal_local_i, target_w2c[i], identity_w2c)
            global_normal.append(normal_zero123)

        global_normal = torch.stack(global_normal, dim=0)
        if for_lotus:
            global_normal[...,0] *= -1
        global_normal = global_normal / torch.norm(global_normal, dim=-1, keepdim=True)
        return global_normal

    def trans_global_2_local(self, normal_local, azimuths_deg, elevations_deg, radius=4.5):
        """
        :param normal_global: (B,H,W,3), torch tensor, range [-1,1]
        :param azimuths_deg: (B,), numpy array, range [0,360]
        :param elevations_deg: (B,), numpy array, range [-90,90]
        :param radius: float, default 4.5
        :return: local_normal: (B,H,W,3), torch tensor, range [-1,1]

        """
        print(f"normal_local.shape:{normal_local.shape}")
        print(f"azimuths_deg.shape:{azimuths_deg.shape}")
        print(f"elevations_deg.shape:{elevations_deg.shape}")
        assert normal_local.shape[0] == azimuths_deg.shape[0] == elevations_deg.shape[0]
        identity_w2c = self.identity_w2c

        # generate target pose
        target_w2c = self.generate_target_pose(azimuths_deg, elevations_deg, radius)
        target_w2c = torch.from_numpy(np.stack([w2c for w2c in target_w2c])).float()
        local_normal = []

        # transform normal
        for i in range(normal_local.shape[0]):
            normal_local_i = normal_local[i]
            normal = self.trans_normal(normal_local_i, identity_w2c, target_w2c[i])
            local_normal.append(normal)

        local_normal = torch.stack(local_normal, dim=0)
        # global_normal[...,0] *= -1
        local_normal = local_normal / torch.norm(local_normal, dim=-1, keepdim=True)
        return local_normal