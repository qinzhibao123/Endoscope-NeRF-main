## Slightly modified version of IBRNet code, see https://github.com/googleinterns/IBRNet

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

    def __init__(self, sampler, num_replicas: Optional[int] = None, rank: Optional[int] = None, shuffle: bool = True,):
        super(DistributedSamplerWrapper, self).__init__(DatasetFromSampler(sampler), num_replicas=num_replicas, rank=rank, shuffle=shuffle,)
        self.sampler = sampler

    def __iter__(self):
        self.dataset = DatasetFromSampler(self.sampler)
        indexes_of_indexes = super().__iter__()
        subsampler_indexes = self.dataset
        return iter(itemgetter(*indexes_of_indexes)(subsampler_indexes))

def create_training_dataset(args):
    print('training dataset: {}'.format(args.train_dataset))
    mode = 'train'
    if '+' not in args.train_dataset:
        train_dataset = dataset_dict[args.train_dataset](args, mode, scenes=args.train_scenes)
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
    else:
        train_dataset_names = args.train_dataset.split('+')
        weights = args.dataset_weights  #[0.45,0.55]
        assert len(train_dataset_names) == len(weights)
        assert np.abs(np.sum(weights) - 1.) < 1e-6
        print('weights:{}'.format(weights))
        train_datasets = []  #
        train_weights_samples = []
        for training_dataset_name, weight in zip(train_dataset_names, weights):
            train_dataset = dataset_dict[training_dataset_name](args, mode, scenes=args.train_scenes,)
            train_datasets.append(train_dataset)
            num_samples = len(train_dataset)
            weight_each_sample = weight / num_samples
            train_weights_samples.extend([weight_each_sample]*num_samples)

        train_dataset = torch.utils.data.ConcatDataset(train_datasets)
        train_weights = torch.from_numpy(np.array(train_weights_samples))
        sampler = WeightedRandomSampler(train_weights, len(train_weights))
        train_sampler = DistributedSamplerWrapper(sampler) if args.distributed else sampler

    return train_dataset, train_sampler



