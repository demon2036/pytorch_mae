from accelerate import Accelerator
from torchvision import transforms
from torchvision.transforms import Compose, ToTensor, Normalize

from autoaugment import CIFAR10Policy
from utils import setup_seed
import torchvision
import torch
from normal_utils import CIFAR10_MEAN, CIFAR10_STD
from torch.utils.data import DataLoader
from utils import get_obj_from_str, json_print
from functools import partial


class BaseTrainer:
    def __init__(self,
                 seed: int,
                 batch_size,
                 max_device_batch_size,
                 total_epoch,
                 mixed_precision,
                 use_aux_dataset=False,
                 unsup_fraction=0.9,
                 aux_data_filename='/home/jtitor/Downloads/1m.npz',
                 save_every=500,
                 transform=True,
                 ):
        setup_seed(int(seed))

        batch_size = batch_size
        load_batch_size = min(max_device_batch_size, batch_size)
        self.total_epoch = total_epoch

        assert batch_size % load_batch_size == 0
        self.steps_per_update = batch_size // load_batch_size

        if not use_aux_dataset:
            print('Using Common DataSet')
            train_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=True, download=True, transform=Compose([
                                                                                                                           transforms.RandomCrop(
                                                                                                                               32,
                                                                                                                               padding=4,
                                                                                                                               fill=128),
                                                                                                                           transforms.RandomHorizontalFlip(),
                                                                                                                           CIFAR10Policy(),
                                                                                                                           ToTensor(), ]) if transform else
                                                                                                                       Compose([ToTensor()]))

            test_dataset = torchvision.datasets.CIFAR10('data/cifar10s', train=False, download=True,
                                                        transform=Compose(
                                                            [ToTensor(), ]))  # 0.5, 0.5

            train_dataloader = DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=8,
                                          prefetch_factor=2, drop_last=True, pin_memory=True, persistent_workers=True)
            test_dataloader = DataLoader(test_dataset, load_batch_size, shuffle=False, num_workers=8, prefetch_factor=4)

        else:
            from datasets import load_data
            print(f'Using Custom DataSet  With unsup_fraction:{unsup_fraction}')

            train_dataset, test_dataset, train_dataloader, test_dataloader = load_data('data/cifar10s',
                                                                                       batch_size=load_batch_size,
                                                                                       batch_size_test=load_batch_size,
                                                                                       num_workers=8,
                                                                                       use_augmentation='base',
                                                                                       use_consistency=False,
                                                                                       shuffle_train=True,
                                                                                       aux_data_filename=aux_data_filename,
                                                                                       unsup_fraction=unsup_fraction,
                                                                                       validation=False)

        self.train_dataset = train_dataset
        self.val_dataset = test_dataset

        self.train_dataloader = train_dataloader
        self.val_dataloader = test_dataloader
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        self.save_every = save_every

    @property
    def device(self):
        return 'cuda' if torch.cuda.is_available() else 'cpu'