# modified from https://github.com/Profactor/continuous-remeshing
import torch
import numpy as np
import trimesh
from typing import Tuple
from pytorch3d.renderer.cameras import camera_position_from_spherical_angles, look_at_rotation
from pytorch3d.renderer import (
    FoVOrthographicCameras,
    look_at_view_transform,
)

def to_numpy(*args):
    def convert(a):
        if isinstance(a,torch.Tensor):
            return a.detach().cpu().numpy()
        assert a is None or isinstance(a,np.ndarray)
        return a
    
    return convert(args[0]) if len(args)==1 else tuple(convert(a) for a in args)

def laplacian(
        num_verts:int,
        edges: torch.Tensor #E,2
        ) -> torch.Tensor: #sparse V,V
    """create sparse Laplacian matrix"""
    V = num_verts
    E = edges.shape[0]

    #adjacency matrix,
    idx = torch.cat([edges, edges.fliplr()], dim=0).type(torch.long).T  # (2, 2*E)
    ones = torch.ones(2*E, dtype=torch.float32, device=edges.device)
    A = torch.sparse.FloatTensor(idx, ones, (V, V))

    #degree matrix
    deg = torch.sparse.sum(A, dim=1).to_dense()
    idx = torch.arange(V, device=edges.device)
    idx = torch.stack([idx, idx], dim=0)
    D = torch.sparse.FloatTensor(idx, deg, (V, V))

    return D - A

def _translation(x, y, z, device):
    return torch.tensor([[1., 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]],device=device) #4,4


def _perspective(fovy, aspect=1.0, n=0.1, f=1000.0, device=None):
    fovy = fovy * torch.pi / 180 
    y = np.tan(fovy / 2)
    return torch.tensor([[1/(y*aspect),    0,            0,              0], 
                         [           0, 1/-y,            0,              0], 
                         [           0,    0, -(f+n)/(f-n), -(2*f*n)/(f-n)], 
                         [           0,    0,           -1,              0]], dtype=torch.float32, device=device)

def _projection(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    """
        see https://blog.csdn.net/wodownload2/article/details/85069240/
    """
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    p = torch.zeros([4,4],device=device)
    p[0,0] = 2*n/(r-l)
    p[0,2] = (r+l)/(r-l)
    p[1,1] = 2*n/(t-b) * (-1 if flip_y else 1)
    p[1,2] = (t+b)/(t-b)
    p[2,2] = -(f+n)/(f-n)
    p[2,3] = -(2*f*n)/(f-n)
    p[3,2] = -1
    return p #4,4

def _orthographic(r, device, l=None, t=None, b=None, n=1.0, f=50.0, flip_y=True):
    if l is None:
        l = -r
    if t is None:
        t = r
    if b is None:
        b = -t
    o = torch.zeros([4,4],device=device)
    o[0,0] = 2/(r-l)
    o[0,3] = -(r+l)/(r-l)
    o[1,1] = 2/(t-b) * (-1 if flip_y else 1)
    o[1,3] = -(t+b)/(t-b)
    o[2,2] = -2/(f-n)
    o[2,3] = -(f+n)/(f-n)
    o[3,3] = 1
    return o #4,4

def make_star_cameras_orig(phis,pol_count,distance:float=10.,r=None,image_size=[512,512],device='cuda'):
    if r is None:
        r = 1/distance
    A = len(phis)
    P = pol_count
    C = A * P # total number of cameras

    phi = phis * torch.pi / 180
    phi_rot = torch.eye(3,device=device)[None,None].expand(A,1,3,3).clone()
    phi_rot[:,0,2,2] = phi.cos()
    phi_rot[:,0,2,0] = -phi.sin()
    phi_rot[:,0,0,2] = phi.sin()
    phi_rot[:,0,0,0] = phi.cos()
    
    theta = torch.arange(1,P+1) * (torch.pi/(P+1)) - torch.pi/2
    theta_rot = torch.eye(3,device=device)[None,None].expand(1,P,3,3).clone()
    theta_rot[0,:,1,1] = theta.cos()
    theta_rot[0,:,1,2] = -theta.sin()
    theta_rot[0,:,2,1] = theta.sin()
    theta_rot[0,:,2,2] = theta.cos()

    mv = torch.empty((C,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(C,3,3)
    mv_ = _translation(0, 0, -distance, device) @ mv

    return mv_, _projection(r,device)

def make_star_cameras_mv_new(phis,eles,distance:float=10.,r=None,fov=None,image_size=[512,512],device='cuda',translation=True):
    import glm
    def sample_spherical(phi, theta, cam_radius):
        theta = torch.deg2rad(theta)
        phi = torch.deg2rad(phi)   

        z = cam_radius * torch.cos(phi) * torch.sin(theta)
        x = cam_radius * torch.sin(phi) * torch.sin(theta)
        y = cam_radius * torch.cos(theta)
 
        return x, y, z
    
    all_mvs = []
    for i in range(len(phis)):
        azimuth = - phis[i] + 1e-10
        ele = - eles[i] + 1e-10 + 90
        x, y, z = sample_spherical(azimuth, ele, distance)
        eye = glm.vec3(x, y, z)
        at = glm.vec3(0.0, 0.0, 0.0)
        up = glm.vec3(0.0, 1.0, 0.0)
        view_matrix = glm.lookAt(eye, at, up)
        all_mvs.append(torch.from_numpy(np.array(view_matrix)).cuda())
    mv = torch.stack(all_mvs)
    
    return mv
    
def make_star_cameras_mv(phis,eles,distance:float=10.,r=None,fov=None,image_size=[512,512],device='cuda',translation=True):
    if r is None:
        r = 0.15
    A = len(phis)
    assert len(eles) == A, f'len(phis): {len(phis)}, len(eles): {len(eles)}'

    phi = phis * torch.pi / 180
    phi_rot = torch.eye(3,device=device)[None].expand(A,3,3).clone()
    phi_rot[:,2,2] = phi.cos()
    phi_rot[:,2,0] = -phi.sin()
    phi_rot[:,0,2] = phi.sin()
    phi_rot[:,0,0] = phi.cos()
    
    
    theta = eles * torch.pi / 180
    theta_rot = torch.eye(3,device=device)[None].expand(A,3,3).clone()
    theta_rot[:,1,1] = theta.cos()
    theta_rot[:,1,2] = -theta.sin()
    theta_rot[:,2,1] = theta.sin()
    theta_rot[:,2,2] = theta.cos()

    mv = torch.empty((A,4,4), device=device)
    mv[:] = torch.eye(4, device=device)
    mv[:,:3,:3] = (theta_rot @ phi_rot).reshape(A,3,3)

    if translation:
        mv_ = _translation(0, 0, -distance, device) @ mv 
    else: 
        mv_ = mv
    return mv_

def make_star_cameras(phis,eles,distance:float=10.,r=None,fov=None,image_size=[512,512],device='cuda',translation=True):
    mv_ = make_star_cameras_mv_new(phis, eles, distance, r, device=device, translation=translation)
    return mv_, _perspective(fov,device=device)

def make_star_cameras_perspective(phis, eles, distance:float=10., r=None, fov=None, device='cuda'):

    return make_star_cameras(phis, eles, distance, r, fov=fov, device=device, translation=True)

def make_star_cameras_orthographic(phis, eles, distance:float=10., r=None, device='cuda'):

    mv = make_star_cameras_mv_new(phis, eles, distance, r, device=device)
    if r is None:
        r = 1
    return mv, _orthographic(r,device)

def make_sphere(level:int=2,radius=1.,device='cuda') -> Tuple[torch.Tensor,torch.Tensor]:
    sphere = trimesh.creation.icosphere(subdivisions=level, radius=1.0, color=None)
    vertices = torch.tensor(sphere.vertices, device=device, dtype=torch.float32) * radius
    faces = torch.tensor(sphere.faces, device=device, dtype=torch.long)
    return vertices,faces


def get_camera(R, T, focal_length=1 / (2**0.5)):
    focal_length = 1 / focal_length
    camera = FoVOrthographicCameras(device=R.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length)
    return camera

def make_star_cameras_orthographic_py3d(azim_list, device, focal=2/1.35, dist=1.1):
    R, T = look_at_view_transform(dist, 0, azim_list)
    focal_length = 1 / focal
    return FoVOrthographicCameras(device=R.device, R=R, T=T, min_x=-focal_length, max_x=focal_length, min_y=-focal_length, max_y=focal_length).to(device)


def rotation_matrix_to_euler_angles(R, return_degrees=True):
    sy = torch.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = torch.atan2(R[2, 1], R[2, 2])
        y = torch.atan2(-R[2, 0], sy)
        z = torch.atan2(R[1, 0], R[0, 0])
    else:
        x = torch.atan2(-R[1, 2], R[1, 1])
        y = torch.atan2(-R[2, 0], sy)
        z = 0

    if return_degrees:
        return torch.tensor([x, y, z]) * 180 / np.pi
    else:
        return torch.tensor([x, y, z])
