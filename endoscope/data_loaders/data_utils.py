# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
from scipy.spatial.transform import Rotation as R
import cv2
import random

rng = np.random.RandomState(234)
_EPS = np.finfo(float).eps * 4.0
TINY_NUMBER = 1e-6      # float32 only has 7 decimal digits precision


def vector_norm(data, axis=None, out=None):
    """Return length, i.e. eucledian norm, of ndarray along axis.
    """
    data = np.array(data, dtype=np.float64, copy=True)
    if out is None:
        if data.ndim == 1:
            return math.sqrt(np.dot(data, data))
        data *= data
        out = np.atleast_1d(np.sum(data, axis=axis))
        np.sqrt(out, out)
        return out
    else:
        data *= data
        np.sum(data, axis=axis, out=out)
        np.sqrt(out, out)


def quaternion_about_axis(angle, axis):
    """Return quaternion for rotation about axis.
    """
    quaternion = np.zeros((4, ), dtype=np.float64)
    quaternion[:3] = axis[:3]
    qlen = vector_norm(quaternion)
    if qlen > _EPS:
        quaternion *= math.sin(angle/2.0) / qlen
    quaternion[3] = math.cos(angle/2.0)
    return quaternion


def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    """
    q = np.array(quaternion[:4], dtype=np.float64, copy=True)
    nq = np.dot(q, q)
    if nq < _EPS:
        return np.identity(4)
    q *= math.sqrt(2.0 / nq)
    q = np.outer(q, q)
    return np.array((
        (1.0-q[1, 1]-q[2, 2],     q[0, 1]-q[2, 3],     q[0, 2]+q[1, 3], 0.0),
        (    q[0, 1]+q[2, 3], 1.0-q[0, 0]-q[2, 2],     q[1, 2]-q[0, 3], 0.0),
        (    q[0, 2]-q[1, 3],     q[1, 2]+q[0, 3], 1.0-q[0, 0]-q[1, 1], 0.0),
        (                0.0,                 0.0,                 0.0, 1.0)
        ), dtype=np.float64)


def rectify_inplane_rotation(src_pose, tar_pose, src_img, th=40):
    relative = np.linalg.inv(tar_pose).dot(src_pose)
    relative_rot = relative[:3, :3]
    r = R.from_matrix(relative_rot)
    euler = r.as_euler('zxy', degrees=True)
    euler_z = euler[0]
    if np.abs(euler_z) < th:
        return src_pose, src_img

    R_rectify = R.from_euler('z', -euler_z, degrees=True).as_matrix()
    src_R_rectified = src_pose[:3, :3].dot(R_rectify)
    out_pose = np.eye(4)
    out_pose[:3, :3] = src_R_rectified
    out_pose[:3, 3:4] = src_pose[:3, 3:4]
    h, w = src_img.shape[:2]
    center = ((w - 1.) / 2., (h - 1.) / 2.)
    M = cv2.getRotationMatrix2D(center, -euler_z, 1)
    src_img = np.clip((255*src_img).astype(np.uint8), a_max=255, a_min=0)
    rotated = cv2.warpAffine(src_img, M, (w, h), borderValue=(255, 255, 255), flags=cv2.INTER_LANCZOS4)
    rotated = rotated.astype(np.float32) / 255.
    return out_pose, rotated


def random_crop(rgb, camera, src_rgbs, src_cameras, size=(200, 300), center=None):   #size=(400, 600)
    h, w = rgb.shape[:2]            #  H:756,   W:1008
    out_h, out_w = size[0], size[1] #out_h:704, out_w:340
    if out_w >= w or out_h >= h:
        return rgb, camera, src_rgbs, src_cameras

    if center is not None:  #不执行
        center_h, center_w = center
    else:
        #官方
        # center_h = np.random.randint(low=out_h // 2 + 1, high=h - out_h // 2 - 1)  #395 获得中心化后的高度h中心坐标
        # center_w = np.random.randint(low=out_w // 2 + 1, high=w - out_w // 2 - 1)  #723

        #qin jia
        # print("h:{}; out_h:{}; out_w:{}; low:{}; high:{}".format(h, out_h, out_w, out_h // 2 + 1, h - out_h // 2 - 1))  #qin jia
        low_h = out_h // 2 + 1
        high_h = h - out_h // 2 - 1
        low_w = out_w // 2 + 1
        high_w = w - out_w // 2 - 1
        center_h = random.randint(low_h, high_h)  #395 获得中心化后的高度h中心坐标
        center_w = random.randint(low_w, high_w)  #723

    rgb_out = rgb[center_h - out_h // 2:center_h + out_h // 2, center_w - out_w // 2:center_w + out_w // 2, :]  #(704，340，3)  rgb[43:747]  获得源图像裁剪后图像的坐标范围
    src_rgbs = np.array(src_rgbs)  #(11，756，1008，3)
    src_rgbs = src_rgbs[:, center_h - out_h // 2:center_h + out_h // 2,
               center_w - out_w // 2:center_w + out_w // 2, :]  #(11,704,340,3)  裁剪后的源图像rgb
    camera[0] = out_h
    camera[1] = out_w
    camera[4] -= center_w - out_w // 2  #-49  原始的camera[4]为w/2=1008/2  表示图像中心的坐标，用于图像坐标系转为像素坐标系；
    camera[8] -= center_h - out_h // 2  #335  原始的camera[8]为h/2=756/2
    src_cameras[:, 4] -= center_w - out_w // 2  #-49     通过原始图像的中心坐标，求裁剪后图像的中心坐标
    src_cameras[:, 8] -= center_h - out_h // 2  #335
    src_cameras[:, 0] = out_h             #704
    src_cameras[:, 1] = out_w             #340
    return rgb_out, camera, src_rgbs, src_cameras  #(704,340,3) (34,)  (11,704,340,3) (11,34)


def random_flip(rgb, camera, src_rgbs, src_cameras):
    h, w = rgb.shape[:2]           #h:704, w:340
    h_r, w_r = src_rgbs.shape[1:3] #h_r:704, w_r:340
    rgb_out = np.flip(rgb, axis=1).copy()  #矩阵在轴1上翻转 意思是将高翻转，是否指将图像沿高度翻转，坐标原点从左下角变到右上角
    src_rgbs = np.flip(src_rgbs, axis=-2).copy()  #源图像同样翻转
    camera[2] *= -1  #焦距取负
    camera[4] = w - 1. - camera[4]  #388
    src_cameras[:, 2] *= -1
    src_cameras[:, 4] = w_r - 1. - src_cameras[:, 4]
    return rgb_out, camera, src_rgbs, src_cameras


def get_color_jitter_params(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    color_jitter = transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)
    transform = transforms.ColorJitter.get_params(color_jitter.brightness,
                                                  color_jitter.contrast,
                                                  color_jitter.saturation,
                                                  color_jitter.hue)
    return transform


def color_jitter(img, transform):
    '''
    Args:
        img: np.float32 [h, w, 3]
        transform:
    Returns: transformed np.float32
    '''
    img = Image.fromarray((255.*img).astype(np.uint8))
    img_trans = transform(img)
    img_trans = np.array(img_trans).astype(np.float32) / 255.
    return img_trans


def color_jitter_all_rgbs(rgb, ref_rgbs, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    transform = get_color_jitter_params(brightness, contrast, saturation, hue)
    rgb_trans = color_jitter(rgb, transform)
    ref_rgbs_trans = []
    for ref_rgb in ref_rgbs:
        ref_rgbs_trans.append(color_jitter(ref_rgb, transform))

    ref_rgbs_trans = np.array(ref_rgbs_trans)
    return rgb_trans, ref_rgbs_trans


def deepvoxels_parse_intrinsics(filepath, trgt_sidelength, invert_y=False):
    # Get camera intrinsics
    with open(filepath, 'r') as file:
        f, cx, cy = list(map(float, file.readline().split()))[:3]
        grid_barycenter = torch.Tensor(list(map(float, file.readline().split())))
        near_plane = float(file.readline())
        scale = float(file.readline())
        height, width = map(float, file.readline().split())

        try:
            world2cam_poses = int(file.readline())
        except ValueError:
            world2cam_poses = None

    if world2cam_poses is None:
        world2cam_poses = False

    world2cam_poses = bool(world2cam_poses)

    cx = cx / width * trgt_sidelength
    cy = cy / height * trgt_sidelength
    f = trgt_sidelength / height * f

    fx = f
    if invert_y:
        fy = -f
    else:
        fy = f

    # Build the intrinsic matrices
    full_intrinsic = np.array([[fx, 0., cx, 0.],
                               [0., fy, cy, 0],
                               [0., 0, 1, 0],
                               [0, 0, 0, 1]])

    return full_intrinsic, grid_barycenter, scale, near_plane, world2cam_poses


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='vector',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)  #20
    num_select = min(num_select, num_cams-1)  #19
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)  #(20,4,4)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':  #两个向量归一化后相乘，裁剪到[-1,1],就是两个单位向量点积的结果(即cos角度),再求反余弦，得到角度
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':  #执行    两个向量差的模长
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)  #(20,)  默认按行求二范数  作为渲染图像与源图像的偏移距离
        dists_tar = np.linalg.norm(tar_cam_locs, axis=1)  #(19,)  qin jia
        dists_ref = np.linalg.norm(ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    #这一段是官方的，结合最后一行使用，因为改为了我的代码，现在这一段是无效的
    if tar_id >= 0:  #19
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself  确保没有选择目标id本身，所以将目标本身和目标渲染的偏移值赋值一个较大的值
    sorted_ids = np.argsort(dists)          #(20,)  对dists按照大小进行排序，从小到大
    selected_ids = sorted_ids[:num_select]  #(19,) 排除最后一个，即本身




    #qin jia  为了要均衡考虑相机到物体的距离，在选取参考相机时，要选取比渲染相机和物体之间距离近的和远的，通过源相机的占比
    # sorted_low_ids=[]
    # sorted_hig_ids=[]
    # dist_cha = dists_ref - dists_tar
    # dist_cha_abs =np.abs(dist_cha)
    #
    # for i in range(len(dist_cha)):
    #     if dist_cha[i]<=0:
    #         sorted_low_ids.append(dist_cha[i])
    #     else:
    #         sorted_hig_ids.append(dist_cha[i])
    # lowB = np.sort(sorted_low_ids)
    # higB = np.sort(sorted_hig_ids)
    # n_low = round(len(lowB) / (len(lowB)+len(higB))*num_select)  #int/int=float， round(float):四舍五入取整数， int(float):只取整数部分
    # n_hig = num_select-n_low   #通过占比来确定参考视图内外的数量分别为多少个
    # selected_low_ids1 = lowB[-n_low:]
    # selected_hig_ids1 = higB[:n_hig]
    # # print(("numble of lower tar_camera to select:{}/{} , numble of higher tar_camera to select :{}/{}").format(n_low, num_select, n_hig, num_select))
    #
    # lowhig = np.append(selected_low_ids1, selected_hig_ids1)
    # lowhigAbs=np.abs(np.array(lowhig))
    # lowhigAbs1=np.sort(lowhigAbs)
    #
    # dist_cha_abs_list=dist_cha_abs.tolist()   #将数组转换为列表
    # ids=[]
    # for i in range(len(lowhig)):
    #     v1=dist_cha_abs_list.index(lowhigAbs1[i])
    #     # if lowhigAbs1[i]==dist_cha_abs[v1]:
    #     ids.append(v1)
    # if tar_id in ids:
    #     # ids.pop(tar_id)
    #     ids.remove(tar_id)
    # ids=np.array(ids)
    # selected_ids=ids
    # # print(selected_ids)



    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids  #(19,) 排除最后一个，即本身
