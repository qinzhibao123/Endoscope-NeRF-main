import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
sys.path.append('../')
from .data_utils import random_crop, random_flip, get_nearest_pose_ids
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
import random

class LLFFDataset(Dataset):
    def __init__(self, args, mode, **kwargs):
        # base_dir = os.path.join(args.rootdir, 'data/real_iconic_noface/')  #  'data/real_iconic_noface/'  'data/nerf_llff_data/'
        # base_dir = os.path.join(args.rootdir, '/media/qin/E/AI/Github/IBRNet/data/real_iconic_noface/')
        base_dir = os.path.join(args.rootdir, '/home/qin/Github/IBRNet-me/data/train_data_denoising/')  # iron  Lungs_2_train  train_data_denoising   train_data_denoising_hsv
        self.args = args
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        scenes = os.listdir(base_dir)
        for i, scene in enumerate(scenes):  #i:0  scene:'data3_ninjabike'
            scene_path = os.path.join(base_dir, scene)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(scene_path, load_imgs=False, factor=4)
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            if mode == 'train':  #当训练时，渲染视图和训练视图一样，20张，为了计算损失
                i_train = np.array(np.arange(int(poses.shape[0])))
                i_render = i_train
            else:
                i_test = np.arange(poses.shape[0])[::self.args.llffhold]
                i_train = np.array([j for j in np.arange(int(poses.shape[0])) if (j not in i_test and j not in i_test)])
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]]*num_render)
            self.render_train_set_ids.extend([i]*num_render)

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):  #503??
        rgb_file = self.render_rgb_files[idx] #随机抽取一个图像    ./image019.png
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.  #(756,1008,3)
        render_pose = self.render_poses[idx]       #(4,4) 当前图像的渲染位姿，也即当前图像的源位姿，用于将渲染出的结果和真实对比
        intrinsics = self.render_intrinsics[idx]    #(4,4) 当前图像的内参
        depth_range = self.render_depth_range[idx]  #{list:2}  [1.33,8.60] 当前图像的深度范围

        train_set_id = self.render_train_set_ids[idx] #20   随机图像所在的场景id(real_iconic_noface共42个场景)
        train_rgb_files = self.train_rgb_files[train_set_id]  #{list:20} 当前id场景下的所有图像路径
        train_poses = self.train_poses[train_set_id]   #(20,4,4)  当前id场景下的所有图像位姿
        train_intrinsics = self.train_intrinsics[train_set_id] #(20,4,4)  当前id场景下的所有图像内参  都一样

        img_size = rgb.shape[:2]  #(756,1008)
        camera = np.concatenate((list(img_size), intrinsics.flatten(), render_pose.flatten())).astype(np.float32)#(34,)  图像尺寸，内参和位姿拼接而成

        if self.mode == 'train':  #执行
            id_render = train_rgb_files.index(rgb_file)  #19  当前图像在当前场景中的id 作为渲染图像的id
            # subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])  #此次为2，表示什么？？？？   按照数组p中的概率随机从数组a中抽取数字
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])  # 此次为2，表示什么？？？？   按照数组p中的概率随机从数组a中抽取数字
            num_select = self.num_source_views + np.random.randint(low=-2, high=3)  #此次11
        else:
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views
        #获得与渲染图像在偏移上临近的源图像，   (19,) 排除最后一个，即本身
        nearest_pose_ids = get_nearest_pose_ids(render_pose, train_poses, min(self.num_source_views*subsample_factor, 20), tar_id=id_render, angular_dist_method='dist')
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)  #此次(11,) 为什么要随机选取几个，而不选择前几个最临近的

        assert id_render not in nearest_pose_ids  #这一句确保要渲染的id不在临近点id中
        # occasionally include input image  为什么要偶尔包含输入图像
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == 'train':  #99.5%的概率不执行
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render #将渲染id偶尔替换nearest_pose_ids最后一个id

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:  #11
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.  #第12(id=11)个图片
            train_pose = train_poses[id] #(4,4)
            train_intrinsics_ = train_intrinsics[id] #(4,4)
            src_rgbs.append(src_rgb)   #添加到列表
            img_size = src_rgb.shape[:2]  #[756,1008]
            src_camera = np.concatenate((list(img_size), train_intrinsics_.flatten(), train_pose.flatten())).astype(np.float32)  #(34,)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)  #(11,756,1008,3)
        src_cameras = np.stack(src_cameras, axis=0)  #(11,34)
        if self.mode == 'train':  #执行
            #官方的
            # crop_h = np.random.randint(low=250, high=750)  #704
            # crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h #704
            # crop_w = int(400 * 600 / crop_h)  #340
            # crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w  #340
            # rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w))  #(704,340,3) (34,) (11,704,340,3) (11,34)


            #qin 修改
            low1 = int(0.5 * src_rgb.shape[0])
            high1 = int(1.0 * src_rgb.shape[0])
            crop_h = random.randint(low1, high1)  # 704
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h  # 704
            if src_rgb.shape[0] % 192 == 0:
                crop_w = int(low1 * high1*1.333/crop_h)   #4/3
            else:
                crop_w = int(low1 * high1*1.778/crop_h)   #16/9
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w  # 340
            rgb, camera, src_rgbs, src_cameras = random_crop(rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w))


        if self.mode == 'train' and random.choice([0, 1]):  #执行概率占一半   图像翻转(深度学习中的图像扩展，是否是增加数据集的量还是增加图像的泛化性)   np.random.choice([0, 1])随机选取0和1
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)  # #(704,340,3) (34,) (11,704,340,3) (11,34) 图像在高维度上翻转

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])  #[1.2,13.76]

        return {'rgb': torch.from_numpy(rgb[..., :3]), 'camera': torch.from_numpy(camera), 'rgb_path': rgb_file, 'src_rgbs': torch.from_numpy(src_rgbs[..., :3]),
                'src_cameras': torch.from_numpy(src_cameras), 'depth_range': depth_range,}

