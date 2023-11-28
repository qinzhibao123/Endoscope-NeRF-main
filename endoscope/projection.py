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


import torch
import torch.nn.functional as F


class Projector():
    def __init__(self, device):
        self.device = device

    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range  检查像素位置是否在有效范围内
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):  #h:378, w:504  pixel_locations:(12,426,64,2)
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]  #(1,1,2)
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  #(12,426,64,2) [n_views, n_points, 2]  归一化到[-1,1]
        return normalized_pixel_locations  #(12,426,64,2)

    def compute_projections(self, xyz, train_cameras):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)      #(426*64,3)
        num_views = len(train_cameras) #12         train_cameras：(12,34)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  #(12,4,4) [n_views, 4, 4] 内参
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  #(12,4,4)  [n_views, 4, 4]      pose
        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  #(426*64,4)  [n_points, 4]  增加一行，组成世界坐标系下空间点的齐次坐标，便于和内参*pose的结果(4,4)矩阵相乘
        projections = train_intrinsics.bmm(torch.inverse(train_poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  #(12,4,426*64)  [n_views, 4, n_points] 采样点分别在源视图中的投影坐标=训练内参@训练的[R t]@采样点位置xyz_h的转置
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  #(12,426*64,2)  [n_views, n_points, 2]  像素的位置=(x,y)/z
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)  #(12,426*64,2)   将输入input张量每个元素的夹紧到区间（哪个区间）,超过或低于的赋值边界值
        mask = projections[..., 2] > 0   #(12,426*64)  a point is invalid if behind the camera  如果采样点在源视图相机后面，采样点是无效的，在源视图上的投影点也是无效的， 应该不存在吧？？？
        return pixel_locations.reshape((num_views, ) + original_shape + (2, )), \
               mask.reshape((num_views, ) + original_shape)  #(12,426,64,2) (12,426,64)

    def compute_angle(self, xyz, query_camera, train_cameras):
        '''
        :param xyz: [..., 3]
        :param query_camera: [34, ]
        :param train_cameras: [n_views, 34]
        :return: [n_views, ..., 4]; The first 3 channels are unit-length vector of the difference between  前3个通道是查询光线和目标光线方向差的单位长度向量
        query and target ray directions, the last channel is the inner product of the two directions.      最后一个通道是两个方向的内积(可以表示角度？)
        '''
        original_shape = xyz.shape[:2]  #（426,64）
        xyz = xyz.reshape(-1, 3)  #(426*64,3)
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  #(12,4,4) [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)  #(12,4,4) [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)) #(12,426*64,3) 查询位姿指目标视图相机的中心轴光线？查询位姿的距离原点的偏移减去采样点位置得到的是采样点相对目标光轴的偏移吗
        ray2tar_pose_q = ray2tar_pose
        # ray2tar_pose /= (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)  #(12,426*64,3)  归一化
        ray2tar_pose =ray2tar_pose/(torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0)) #(12,426*64,3)  得到的是采样点位置相对源视图光轴的位姿吗
        ray2train_pose_q = ray2train_pose
        # ray2train_pose /= (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)  #(12,426*64,3)
        ray2train_pose = ray2train_pose/(torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose - ray2train_pose   #(12,426*64,3)  是否指目标视图相对于源视图的位姿，也即量这个的射线差
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)   #(12,426*64,1) 求射线差的模长，  用于后面的归一化
        # ray2tar_dis = torch.norm(ray2tar_pose, dim=-1, keepdim=True)   #qin jia
        # ray2train = torch.norm(ray2train_pose, dim=-1, keepdim=True)  #qin jia

        ray_diff_dot = torch.sum(ray2tar_pose * ray2train_pose, dim=-1, keepdim=True)  #(12,426*64,1)  求两者的点积，在维度-1求和  ？？？
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)  #(12,426*64,3)   将ray_diff归一化

        #原始   权重和偏差的单位方向，以及两单位向量的点积有关
        # ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)  #(12,426*64,4)  通过采样点的目标射线和源射线偏差的方向和大小 ***********
        # ray_diff = ray_diff.reshape((num_views,) + original_shape + (4,))  # (12,426,64,4)

        #qin jia 1
        # ray_diff = torch.cat([ray_diff_direction, ray_diff_dot, ray_diff_norm, ray2tar_dis, ray2train], dim=-1)  #将上面修改了，加了变量  (12,426*64,7)
        # ray_diff = ray_diff.reshape((num_views, ) + original_shape + (7, ))  #(12,426,64,7)

        # #qin jia 2
        # a1 = torch.abs(query_pose[:, :3, 3].unsqueeze(1))   #(8,1,3)
        # a2 = torch.abs(train_poses[:, :3, 3].unsqueeze(1))  #(8,1,3)
        # a3 = torch.norm(torch.abs(a2 * a2 - a1 * a1), dim=-1, keepdim=True)
        # a = 1/a3  #(8,1,1)

        # # qin jia 3  权重和偏差的单位方向，以及距离的平方差有关
        # d3 = torch.abs(torch.norm(torch.abs(ray2tar_pose_q*ray2tar_pose_q-ray2train_pose_q*ray2train_pose_q), dim=-1, keepdim=True)) #(8,2944,1)
        # # print("d3_min:",torch.min(d3))
        # d = 1 / (d3+1e-6)  #(8,2944,1)
        # d/=torch.norm(d, dim=0, keepdim=True)
        # # print("d:", torch.max(d)-0.01)
        #
        # b=d*ray_diff_dot
        # c=b/(torch.max(b)+1e-6)
        # # print("c:",torch.max(c))

        # #qin4   权重和点积，以及距离的平方差的绝对值有关
        # d4 = torch.norm(ray2tar_pose_q * ray2tar_pose_q - ray2train_pose_q * ray2train_pose_q, dim=-1, keepdim=True)  # (8,2944,1)
        # ray_diff = torch.cat([ray_diff_dot, d4], dim=-1)


        # qin5   权重和点积，以及距离的平方差有关
        # t1=ray2tar_pose_q**2
        # t2=ray2tar_pose_q*ray2tar_pose_q
        tar=torch.norm(ray2tar_pose_q, dim=-1, keepdim=True)
        tra=torch.norm(ray2train_pose_q, dim=-1, keepdim=True)
        dd=tar**2-tra**2
        # dd/=torch.norm(dd).max(0, keepdim=True)[0]
        # dd2=1/dd
        # dd3 = dd2/torch.norm(dd2, dim=0, keepdim=True)

        dd = torch.exp(-torch.abs(dd))

        # ray_diff = torch.cat([ray_diff_dot, dd], dim=-1)   #1
        # ray_diff = torch.cat([ray_diff_direction, ray_diff_dot * dd], dim=-1)  # 1
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot, ray_diff_dot*dd], dim=-1)  # 2


        # ray_diff = torch.cat([ray_diff_direction, c], dim=-1)  #将上面修改了，加了变量  (12,426*64,7)
        ray_diff = ray_diff.reshape((num_views, ) + original_shape + (5, ))  #(12,426,64,7)

        return ray_diff

    def compute(self,  xyz, query_camera, train_imgs, train_cameras, featmaps):
        '''
        :param xyz: [n_rays, n_samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, n_views, h, w, 3]
        :param train_cameras: [1, n_views, 34]
        :param featmaps: [n_views, d, h, w]
        :return: rgb_feat_sampled: [n_rays, n_samples, 3+n_feat],
                 ray_diff: [n_rays, n_samples, 4],
                 mask: [n_rays, n_samples, 1]
        '''
        assert (train_imgs.shape[0] == 1) \
               and (train_cameras.shape[0] == 1) \
               and (query_camera.shape[0] == 1), 'only support batch_size=1 for now'

        train_imgs = train_imgs.squeeze(0)  # [n_views, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [n_views, 34]
        query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [n_views, 3, h, w]

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image  计算查询点到每个参考图像的投影
        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)  #(12,426,64,2) (12,426,64)     xyz:(426,64,3)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   #(12,426,64,2)  像素坐标归一化到[-1,1] [n_views, n_rays, n_samples, 2]

        # rgb sampling  rgb采样*****************
        rgbs_sampled = F.grid_sample(train_imgs, normalized_pixel_locations, align_corners=True) #(12,3,426,64)  按照由采样点投影到源视图中并归一化后的位置在原图像中取对应的rgb值
        rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  #(426,64,12,3) [n_rays, n_samples, n_views, 3]

        # deep feature sampling  为什么叫深度特征采样******   深度和特征图有什么联系？？？
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations, align_corners=True) #(12,32,426,64)   按照由采样点投影到源视图中并归一化后的位置在由原图像获得的特征图中取对应的特征值
        feat_sampled = feat_sampled.permute(2, 3, 0, 1)  #(426,64,12,32) [n_rays, n_samples, n_views, d]
        rgb_feat_sampled = torch.cat([rgb_sampled, feat_sampled], dim=-1)   #(426,64,12,35)   *****   [n_rays, n_samples, n_views, d+3]  渲染视角上的采样点在参考图像投影位置对应的rgb和feat进行拼接

        # mask
        inbound = self.inbound(pixel_locations, h, w) #检查像素位置是否在有效范围内, 意思是目标图像射线上的采样点投影到源视图中后，是否在源图像的尺寸范围内，如果超出，将投影点的值取False
        ray_diff = self.compute_angle(xyz, query_camera, train_cameras)  #(12,426,64,4) 通过采样点的目标(查询)射线和源射线偏差的方向(单位向量)和大小
        ray_diff = ray_diff.permute(1, 2, 0, 3)  #(426,64,12,4)
        mask = (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]   #(426,64,12,1)  采样点既在源视图的边界范围内，又在源视图的相机前方 [n_rays, n_samples, n_views, 1]
        return rgb_feat_sampled, ray_diff, mask   #(426,64,12,35) (426,64,12,4) (426,64,12,1)






