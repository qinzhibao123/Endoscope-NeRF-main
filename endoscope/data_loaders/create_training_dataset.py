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
from . import dataset_dict
from torch.utils.data import Dataset, Sampler
from torch.utils.data import DistributedSampler, WeightedRandomSampler
from typing import Optional
from operator import itemgetter
import torch

class DatasetFromSampler(Dataset):
    """Dataset to create indexes from `Sampler`.
    Args:sampler: PyTorch sampler """

    def __init__(self, sampler: Sampler):
        """Initialisation for DatasetFromSampler."""
        self.sampler = sampler
        self.sampler_list = None

    def __getitem__(self, index: int):
        """Gets element of the dataset.
        Args: index: index of the element in the dataset
        Returns: Single element by index
        """
        if self.sampler_list is None:
            self.sampler_list = list(self.sampler)
        return self.sampler_list[index]

    def __len__(self) -> int:
        """ Returns: int: length of the dataset"""
        return len(self.sampler)

class DistributedSamplerWrapper(DistributedSampler):
    """ Wrapper over `Sampler` for distributed training.
    Allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with
    `torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSamplerWrapper instance as a DataLoader
    sampler, and load a subset of subsampled data of the original dataset
    that is exclusive to it.
    .. note::
        Sampler is assumed to be of constant size.
    """

    def __init__(self, sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True,):
        """Args:
            sampler: Sampler used for subsampling
            num_replicas (int, optional): Number of processes participating in
              distributed training
            rank (int, optional): Rank of the current process
              within ``num_replicas``
            shuffle (bool, optional): If true (default),
              sampler will shuffle the indices
        """
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

def create_training_dataset(args):
    # parse args.train_dataset, "+" indicates that multiple datasets are used, for example "ibrnet_collect+llff+spaces"
    # otherwise only one dataset is used
    # args.dataset_weights should be a list representing the resampling rate for each dataset, and should sum up to 1

    print('training dataset: {}'.format(args.train_dataset))
    mode = 'train'
    if '+' not in args.train_dataset:
        train_dataset = dataset_dict[args.train_dataset](args, mode, scenes=args.train_scenes)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        train_dataset_names = args.train_dataset.split('+')  #['llff', 'ibrnet_collected']
        weights = args.dataset_weights  #[0.45,0.55]
        assert len(train_dataset_names) == len(weights)
        assert np.abs(np.sum(weights) - 1.) < 1e-6
        print('weights:{}'.format(weights))
        train_datasets = []  #
        train_weights_samples = []
        for training_dataset_name, weight in zip(train_dataset_names, weights):  #第一个训练集：'llff',0.45
            train_dataset = dataset_dict[training_dataset_name](args, mode, scenes=args.train_scenes,)#会进入***
            train_datasets.append(train_dataset)
            num_samples = len(train_dataset)
            weight_each_sample = weight / num_samples  #0.45/1025
            train_weights_samples.extend([weight_each_sample]*num_samples)

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)  #{2977}  2977=两个场景数据集的图片数相加（real_iconic_noface 1025 + ibrnet_collected 1952）
        train_weights = torch.from_numpy(np.array(train_weights_samples))  #权重个数上面一样，  每个图片占各自场景数据集图片数的比例，如real_iconic_noface(占0.45)中：每个图片占比=0.45/1025
        sampler = WeightedRandomSampler(train_weights, len(train_weights))  #{2977}  每一张图像随机采样
        train_sampler = DistributedSamplerWrapper(sampler) if args.distributed else sampler  #判断是否使用多个显卡，如果不是，值为sampler

    return train_dataset, train_sampler   #{2977}  {2977}  分别包括多个场景的每个图片的参数(训练和渲染的图片路径,pose,内参)  权重：每个图片占的权重



