# Copyright 2020 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import os
import imageio
from .colmap_read_model import read_images_binary

########## Slightly modified version of LLFF data loading code
##########  see https://github.com/Fyusion/LLFF for original

def parse_llff_pose(pose):
    '''convert llff format pose to 4x4 matrix of intrinsics and extrinsics (opencv convention)  #opencv坐标系
    Args:pose: matrix [3, 4]
    Returns: intrinsics [4, 4] and c2w [4, 4] '''
    h, w, f = pose[:3, -1]
    c2w = pose[:3, :4]
    c2w_4x4 = np.eye(4)
    c2w_4x4[:3] = c2w
    c2w_4x4[:, 1:3] *= -1  #为何取负数？？？？
    intrinsics = np.array([[f, 0, w / 2., 0], [0, f, h / 2., 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    return intrinsics, c2w_4x4

def batch_parse_llff_poses(poses):
    all_intrinsics = []
    all_c2w_mats = []
    for pose in poses:
        intrinsics, c2w_mat = parse_llff_pose(pose)
        all_intrinsics.append(intrinsics)
        all_c2w_mats.append(c2w_mat)
    all_intrinsics = np.stack(all_intrinsics)
    all_c2w_mats = np.stack(all_c2w_mats)
    return all_intrinsics, all_c2w_mats

def _minify(basedir, factors=[], resolutions=[]):
    needtoload = False
    for r in factors:
        imgdir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(imgdir):
            needtoload = True
    for r in resolutions:
        imgdir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(imgdir):
            needtoload = True
    if not needtoload:
        return

    from subprocess import check_output

    imgdir = os.path.join(basedir, 'images')
    imgs = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    imgdir_orig = imgdir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resizearg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resizearg = '{}x{}'.format(r[1], r[0])
        imgdir = os.path.join(basedir, name)
        if os.path.exists(imgdir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(imgdir)
        check_output('cp {}/* {}'.format(imgdir_orig, imgdir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resizearg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(imgdir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(imgdir, ext), shell=True)
            print('Removed duplicates')
        print('Done')

def _load_data(basedir, factor=None, width=None, height=None, load_imgs=True):
    poses_arr = np.load(os.path.join(basedir, 'poses_bounds.npy'))  #(20,17)
    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  #(3,5,20)
    bds = poses_arr[:, -2:].transpose([1, 0])  #(2,20)

    img0 = [os.path.join(basedir, 'images', f) for f in sorted(os.listdir(os.path.join(basedir, 'images'))) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')][0]
    sh = imageio.imread(img0).shape  #(3024,4032,3)
    sfx = ''
    if factor is not None and factor != 1:  #执行  factor=4
        sfx = '_{}'.format(factor)
        _minify(basedir, factors=[factor])
        factor = factor
    elif height is not None:
        factor = sh[0] / float(height)
        width = int(sh[1] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    elif width is not None:
        factor = sh[1] / float(width)
        height = int(sh[0] / factor)
        _minify(basedir, resolutions=[[height, width]])
        sfx = '_{}x{}'.format(width, height)
    else:
        factor = 1

    imgdir = os.path.join(basedir, 'images' + sfx)   #nerf_llff_data/fern/images_4
    if not os.path.exists(imgdir):
        print(imgdir, 'does not exist, returning')
        return

    imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f.endswith('JPG') or f.endswith('jpg') or f.endswith('png')] #{list:20}

    if poses.shape[-1] != len(imgfiles):  #不执行
        imagesfile = os.path.join(basedir, 'sparse/0/images.bin')
        imdata = read_images_binary(imagesfile)
        imnames = [imdata[k].name[0:-4] for k in imdata]
        imgfiles = [os.path.join(imgdir, f) for f in sorted(os.listdir(imgdir)) if f[0:-4] in imnames]
        print('{}: Mismatch between imgs {} and poses {} !!!!'.format(basedir, len(imgfiles), poses.shape[-1]))
        return

    sh = imageio.imread(imgfiles[0]).shape  #{tuple:3} 值为(756,1008,3)
    poses[:2, 4, :] = np.array(sh[:2]).reshape([2, 1])  #将原poses中的长宽3024*4032缩放4倍为756*1008
    poses[2, 4, :] = poses[2, 4, :] * 1. / factor  #焦距也缩放4倍，为815.13

    def imread(f):
        if f.endswith('png'):
            return imageio.imread(f, ignoregamma=True)
        else:
            return imageio.imread(f)

    if not load_imgs:  #执行
        imgs = None
    else:
        imgs = [imread(f)[..., :3] / 255. for f in imgfiles]
        imgs = np.stack(imgs, -1)
        print('Loaded image data', imgs.shape, poses[:, -1, 0])
    return poses, bds, imgs, imgfiles  #(3,5,20) (2,20) None {list:20}

def normalize(x):
    return x / np.linalg.norm(x)

def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.stack([vec0, vec1, vec2, pos], 1)
    return m

def ptstocam(pts, c2w):
    tt = np.matmul(c2w[:3, :3].T, (pts - c2w[:3, 3])[..., np.newaxis])[..., 0]
    return tt

def poses_avg(poses):
    hwf = poses[0, :3, -1:]  ##(3,1)
    center = poses[:, :3, 3].mean(0)  #(3,)
    vec2 = normalize(poses[:, :3, 2].sum(0))  #(3,)
    up = poses[:, :3, 1].sum(0)   #(3,)
    c2w = np.concatenate([viewmatrix(vec2, up, center), hwf], 1)
    return c2w

def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, rots, N):
    render_poses = []
    rads = np.array(list(rads) + [1.])  #（4，）
    hwf = c2w[:, 4:5]  #(3,1) [756,1008,815.13] 获得图像的长宽焦距
    for theta in np.linspace(0., 2. * np.pi * rots, N + 1)[:-1]:
        c = np.dot(c2w[:3, :4], np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.]) * rads)
        z = normalize(c - np.dot(c2w[:3, :4], np.array([0, 0, -focal, 1.])))
        render_poses.append(np.concatenate([viewmatrix(z, up, c), hwf], 1))
    return render_poses

def recenter_poses(poses):
    poses_ = poses + 0  #（20,3,5）
    bottom = np.reshape([0, 0, 0, 1.], [1, 4])  #(1,4)
    c2w = poses_avg(poses)  #（3，5）  平均位姿/相机到世界坐标系的投影矩阵
    c2w = np.concatenate([c2w[:3, :4], bottom], -2)  #（3，4）拼接一行为(4,4)，即标准的图像位姿（包括旋转和平移）
    bottom = np.tile(np.reshape(bottom, [1, 1, 4]), [poses.shape[0], 1, 1])  #（20,1,4）
    poses = np.concatenate([poses[:, :3, :4], bottom], -2)   #(20,4,4)   20个图像的位姿

    poses = np.linalg.inv(c2w) @ poses   #？？？？？？？？？？？？？*********  是将位姿从相机坐标系转换到世界坐标系吗？
    poses_[:, :3, :4] = poses[:, :3, :4]  #只需要poses的前3行前4列
    poses = poses_ #(20,3,5)  再加上第五列(高、宽和焦距)
    return poses  #(20,3,5)

#####################
def spherify_poses(poses, bds):
    p34_to_44 = lambda p: np.concatenate([p, np.tile(np.reshape(np.eye(4)[-1, :], [1, 1, 4]), [p.shape[0], 1, 1])], 1)
    rays_d = poses[:, :3, 2:3]
    rays_o = poses[:, :3, 3:4]
    def min_line_dist(rays_o, rays_d):
        A_i = np.eye(3) - rays_d * np.transpose(rays_d, [0, 2, 1])
        b_i = -A_i @ rays_o
        pt_mindist = np.squeeze(-np.linalg.inv((np.transpose(A_i, [0, 2, 1]) @ A_i).mean(0)) @ (b_i).mean(0))
        return pt_mindist

    pt_mindist = min_line_dist(rays_o, rays_d)

    center = pt_mindist
    up = (poses[:, :3, 3] - center).mean(0)

    vec0 = normalize(up)
    vec1 = normalize(np.cross([.1, .2, .3], vec0))
    vec2 = normalize(np.cross(vec0, vec1))
    pos = center
    c2w = np.stack([vec1, vec2, vec0, pos], 1)

    poses_reset = np.linalg.inv(p34_to_44(c2w[None])) @ p34_to_44(poses[:, :3, :4])

    rad = np.sqrt(np.mean(np.sum(np.square(poses_reset[:, :3, 3]), -1)))

    sc = 1. / rad
    poses_reset[:, :3, 3] *= sc
    bds *= sc
    rad *= sc

    centroid = np.mean(poses_reset[:, :3, 3], 0)
    zh = centroid[2]
    radcircle = np.sqrt(rad ** 2 - zh ** 2)
    new_poses = []

    for th in np.linspace(0., 2. * np.pi, 120):  #120
        camorigin = np.array([radcircle * np.cos(th), radcircle * np.sin(th), zh])
        up = np.array([0, 0, -1.])

        vec2 = normalize(camorigin)
        vec0 = normalize(np.cross(vec2, up))
        vec1 = normalize(np.cross(vec2, vec0))
        pos = camorigin
        p = np.stack([vec0, vec1, vec2, pos], 1)

        new_poses.append(p)

    new_poses = np.stack(new_poses, 0)

    new_poses = np.concatenate([new_poses, np.broadcast_to(poses[0, :3, -1:], new_poses[:, :3, -1:].shape)], -1)
    poses_reset = np.concatenate(
        [poses_reset[:, :3, :4], np.broadcast_to(poses[0, :3, -1:], poses_reset[:, :3, -1:].shape)], -1)

    return poses_reset, new_poses, bds

def load_llff_data(basedir, factor=8, recenter=True, bd_factor=.75, spherify=False, path_zflat=False, load_imgs=True):   #bd_factor=0.75, factor=4, basedir:data/nerf_llff_data/fern
    out = _load_data(basedir, factor=factor, load_imgs=load_imgs)  # poses,bds,imgs,imgfiles (3,5,20) (2,20) None {list:20}   factor=4,load_imgs=False   downsamples original imgs by 4x
    if out is None:  #不执行
        return
    else:
        poses, bds, imgs, imgfiles = out   #poses, bds, imgs, imgfiles  #(3,5,20) (2,20) None {list:20}

    # print('Loaded', basedir, bds.min(), bds.max())

    # Correct rotation matrix ordering and move variable dim to axis 0  正确的旋转矩阵顺序和移动可变的维度dim(轴1)到轴0
    poses = np.concatenate([poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1)
    poses = np.moveaxis(poses, -1, 0).astype(np.float32)  #（20，3，5）  移动最后一个轴到第0个轴
    if imgs is not None:  #不执行
        imgs = np.moveaxis(imgs, -1, 0).astype(np.float32)
        images = imgs
        images = images.astype(np.float32)
    else:
        images = None
    bds = np.moveaxis(bds, -1, 0).astype(np.float32)   #(20,2)

    # Rescale if bd_factor is provided  如果提供bd因子，则缩放
    sc = 1. if bd_factor is None else 1. / (bds.min() * bd_factor)  #sc=0.0785 用于位姿和bds的缩放   ？？？？？？  bd_factor=0.75
    poses[:, :3, 3] *= sc   #位姿缩放sc倍(0.0785)
    bds *= sc  #bds缩放sc倍(0.0785)

    if recenter:  #执行
        poses = recenter_poses(poses)   #(20,3,5) 20个图像的位姿被中心化   ****  从相机坐标系转换到世界坐标系

    if spherify:  #不执行
        poses, render_poses, bds = spherify_poses(poses, bds)

    else:

        c2w = poses_avg(poses)  #(3,5) 所有图像的中心位姿的平均位姿 作为c2w
        # print('recentered', c2w.shape)
        # print(c2w[:3, :4])

        ## Get spiral  获得螺旋轨迹
        # Get average pose 获得平均位姿  和前面的c2w有什么区别？？
        up = normalize(poses[:, :3, 1].sum(0))   #（3,）指什么？？？？ 先对20个位姿在第2列进行求和，再进行归一化，得到最up的位姿

        # Find a reasonable "focus depth" for this dataset   为这个数据集找到一个合理的“焦点深度”
        close_depth, inf_depth = bds.min() * .9, bds.max() * 5.   #1.20 31.40   为何要分别乘0.9和5
        dt = .75
        mean_dz = 1. / (((1. - dt) / close_depth + dt / inf_depth))  #4.306  平均深度为什么要这么求？？？？？
        focal = mean_dz  #4.306   平均深度作为焦距

        # Get radii for spiral path  获得螺旋路径的半径
        shrink_factor = .8   #收缩因子
        zdelta = close_depth * .2   #0.24
        tt = poses[:, :3, 3]  #(20,3)  tt表示平移  ptstocam(poses[:3,3,:].T, c2w).T
        rads = np.percentile(np.abs(tt), 40, 0)  #(3,)  90  取tt的第0维度上的第90%分位的数值，是第17.1个图像(共20个含有19个间隔 )的位姿
        c2w_path = c2w #(3,5)
        N_views = 120  #渲染120个新视角
        N_rots = 3  #旋转2圈，即一圈60个视角
        if path_zflat:  #不执行
            #             zloc = np.percentile(tt, 10, 0)[2]
            zloc = -close_depth * .1
            c2w_path[:3, 3] = c2w_path[:3, 3] + zloc * c2w_path[:3, 2]
            rads[2] = 0.
            N_rots = 1
            N_views /= 2

        # Generate poses for spiral path  生成螺旋路径的位姿  即渲染位姿************************
        render_poses = render_path_spiral(c2w_path, up, rads, focal, zdelta, zrate=.5, rots=N_rots, N=N_views)  #{120},每个都为(3,5)

    render_poses = np.array(render_poses).astype(np.float32)  #（120,3,5）

    c2w = poses_avg(poses)  #(3,5) 所有图像的中心位姿的平均位姿 作为c2w
    # print('Data:')
    # print(poses.shape, images.shape, bds.shape)

    dists = np.sum(np.square(c2w[:3, 3] - poses[:, :3, 3]), -1)  #(20,)  平均位姿分别和20个源视图位姿作差
    i_test = np.argmin(dists)  #12  求值最小的下标  从20张源视图中找到最接近平均位姿的源图像位姿的下标
    # print('HOLDOUT view is', i_test)  ？？？？？？？？？？？？？？
    poses = poses.astype(np.float32)

    return images, poses, bds, render_poses, i_test, imgfiles  #None 源视图位姿(20,3,5) 源视图bds(20,2) 渲染视图位姿(120,3,5) 源视图中离平均位姿最近的下标;12  源视图列表{list:20}

if __name__ == '__main__':
    scene_path = '/home/qianqianwang/datasets/nerf_llff_data/trex/'
    images, poses, bds, render_poses, i_test, img_files = load_llff_data(scene_path)
    print(bds)

