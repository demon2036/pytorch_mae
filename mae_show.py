import os
import argparse
import math
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm


from baseline.model_test import *
from normal_utils import CIFAR10_MEAN, CIFAR10_STD
from utils import setup_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-bs', '--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=128)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.pth')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True,
                                                 transform=Compose([ToTensor(), ]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(),   ]))#Normalize(0.5, 0.5)
    dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    writer = SummaryWriter(os.path.join('logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = MAE_ViT_2_T(mask_ratio=args.mask_ratio).to(device)

    # summary(model, (1, 3, 32, 32), )

    model.load_state_dict(torch.load('baseline_aux09_tiny_7999.pt'),strict=False)
    # model = torch.load('vit-t-mae.pth')
    # model = torch.load('baseline_aux09_tiny_7999.pt')
    # print(model)
    # model = torch.load('model/vit_t_2000_l1_fp32.pth')
    # model.encoder.shuffle = PatchShuffle(0.1)
    # print(model.encoder.shuffle.ratio)

    # model = torch.load('vit-t-mae.pth')


    step_count = 0
    for e in range(args.total_epoch):
        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)

            val_img = Normalize(CIFAR10_MEAN, CIFAR10_STD)(val_img)

            predicted_val_img, mask = model(val_img)

            print(mask.sum()/torch.ones_like(mask).sum())

            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)

            print(img.shape)


            # img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            # writer.add_image('mae_image', (img + 1) / 2, global_step=e)
            torchvision.utils.save_image((img + 1) / 2, 'mae_img.png')
            # torchvision.utils.save_image(img, 'mae_img.png')
            # torchvision.utils.save_image()

        break

    # torch.save(model.state_dict(), 'mae_state.pt')
    torch.save(model.state_dict(), 'vit_s.pt')
