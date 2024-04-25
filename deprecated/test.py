import os
import argparse
import math

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Normalize
from torchvision.models import ViT_L_16_Weights
from tqdm import tqdm

from moe_model_fork import *
from mod_model import *
from utils import setup_seed
from accelerate import Accelerator
from autoaugment import CIFAR10Policy
from vit import ViT_2_T, ViT_2_B
from sam import SAM

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256 )
    parser.add_argument('--base_learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--total_epoch', type=int, default=100)
    parser.add_argument('--warmup_epoch', type=int, default=5)
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--output_model_path', type=str, default='vit-t-classifier-from_scratch.pth')
    parser.add_argument('--model', type=str, default='tiny')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('data', train=True, download=True, transform=Compose([
        transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
        transforms.RandomHorizontalFlip(), CIFAR10Policy(),
        ToTensor(), Normalize(0.5, 0.5)]))
    val_dataset = torchvision.datasets.CIFAR10('data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    train_dataloader = torch.utils.data.DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=4)
    val_dataloader = torch.utils.data.DataLoader(val_dataset, load_batch_size, shuffle=False, num_workers=4)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.pretrained_model_path is not None:
        model = torch.load(args.pretrained_model_path, map_location='cpu')
        writer = SummaryWriter(os.path.join('../logs', 'cifar10', 'pretrain-cls'))
    else:
        # model = MAE_ViT()
        # model=MAE_ViT_2_S()
        writer = SummaryWriter(os.path.join('../logs', 'cifar10', 'scratch-cls'))

        if args.model == 'tiny':
            model = MAE_ViT_2_T()
        elif args.model == 'small':
            model = MAE_ViT_2_S()
        elif args.model == 'mini':
            model = MAE_ViT_2_M()
        elif args.model == 'base':
            model = MAE_ViT_2_B()

        elif args.model == 'tiny_4':
            model = MAE_ViT_4_T()
        elif args.model == 'small_4':
            model = MAE_ViT_4_S()
        elif args.model == 'mini_4':
            model = MAE_ViT_4_M()
        elif args.model == 'base_4':
            model = MAE_ViT_4_B()
        elif args.model == 'tiny_moe':
            model = MAE_ViT_2_T_moe()
        elif args.model == 'small_moe':
            model = MAE_ViT_2_S_moe()
        elif args.model == 'mini_moe':
            model = MAE_ViT_2_M_moe()
        elif args.model == 'base_moe':
            model = MAE_ViT_2_B_moe()
        else:
            raise NotImplemented('')

    model = ViT_Classifier(model.encoder, num_classes=10).to(device)
    # model = ViT_2_B().to(device)

    if device == 'cuda':
        net = torch.nn.DataParallel(model)
    # loss_fn = torch.nn.CrossEntropyLoss()
    loss_fn = torch.nn.CrossEntropyLoss()
    acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())
    #
    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256,
                              betas=(0.9, 0.999), weight_decay=args.weight_decay)

    # optim = torch.optim.AdamW(model.parameters(), lr=1e-4 * args.batch_size / 256,
    #                           weight_decay=1e-5)

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    best_val_acc = 0
    step_count = 0
    optim.zero_grad()

    accelerator = Accelerator(mixed_precision='fp16')

    model, optim, train_dataloader,lr_scheduler = accelerator.prepare(model, optim, train_dataloader,lr_scheduler)

    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        train_step = len(train_dataloader)
        with tqdm(total=train_step, desc=f'Train Epoch {e + 1}/{args.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for img, label in iter(train_dataloader):
                optim.zero_grad()
                z_router_losses = []
                with accelerator.autocast():
                    step_count += 1
                    img = img.to(device)
                    label = label.to(device)
                    logits = model(img)
                    loss = loss_fn(logits, label)  # F.softmax(logits, -1)

                acc = acc_fn(logits, label)
                # loss.backward()
                accelerator.backward(loss)  # 1e-2 * z_router_losses

                accelerator.clip_grad_value_(model.parameters(), 1.0)


                optim.step()


                # if step_count % steps_per_update == 0:
                #     # accelerator.unscale_gradients(optimizer)
                #     accelerator.clip_grad_norm_(model.parameters(), 1.0)
                #
                #     optim.step()
                #     optim.zero_grad()
                losses.append(loss.item())
                acces.append(acc.item())

                pbar.set_postfix(**{'Train Loss': np.mean(losses),
                                    'Tran accs': np.mean(acces),
                                    # 'z_router_losses': np.mean(z_router_losses.item())
                                    })
                pbar.update(1)

        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

