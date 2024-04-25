import os
import argparse
import math
import numpy as np
import torch
import torchvision
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from base_trainer import BaseTrainer
from mod_fork_every2.mod_model_fork_every2 import *
from normal_utils import CIFAR10_MEAN, CIFAR10_STD, DeNormalize
from utils import get_obj_from_str, acc_fn, get_config, json_print, print_with_seperator

import yaml
import json
from torchinfo import summary
import torch.optim.adamw


def get_model_mae(model: str = None, mask_ratio=0.75):
    assert model is not None
    mae_model = get_obj_from_str(model)()
    return mae_model


def instant_optimizer(optimizer_configs, model_parameter, batch_size):
    target = optimizer_configs['target']
    configs = optimizer_configs['configs']

    optimizer = get_obj_from_str(target)
    print_with_seperator(f"Using Optimizer {optimizer} Optimizer Config ")
    json_print(configs)

    if 'blr' in configs:
        assert 'lr' not in configs, ("Since you Using base_learning_rate ,learning_rate should not predefine "
                                     "in optimizer config")
        effective_learning_rate = configs['blr'] * batch_size / 256
        configs.update({'lr': effective_learning_rate})

        print_with_seperator(
            f"Using base learning rate:{configs['blr']} batch size:{batch_size} , effective learning rate:blr*batch_size/256 = {effective_learning_rate} ")
        configs.pop('blr')
        json_print(configs)
    elif 'lr' in configs:
        pass
    else:
        raise NotImplemented()

    return optimizer(model_parameter, **configs)


class MaeTrainer(BaseTrainer):
    def __init__(self,
                 seed=2022,
                 batch_size=128,
                 max_device_batch_size=256,
                 base_learning_rate=1e-3,
                 weight_decay=0.05,
                 total_epoch=100,
                 warmup_epoch=5,
                 pretrained_model_path=None,
                 save_model_path=None,
                 model_instant_function=get_model_mae,
                 model_target: str = None,
                 save_model_name: str = None,
                 mask_ratio=0.75,
                 mixed_precision='fp16',
                 loss='l2',
                 use_aux_dataset=False,
                 unsup_fraction=0.9,
                 aux_data_filename='/home/jtitor/Downloads/1m.npz',
                 compile=False,
                 save_every=500,
                 optimizer='torch.optim.AdamW',
                 # optimizer_configs=

                 ):
        super().__init__(seed, batch_size, max_device_batch_size, total_epoch, mixed_precision, use_aux_dataset,
                         unsup_fraction, aux_data_filename, save_every=save_every, transform=False)
        self.compile = compile
        self.loss = loss
        self.model = model_instant_function(model_target, mask_ratio).to(self.device)
        # self.optim = torch.optim.AdamW(self.model.parameters(),
        #                                lr=base_learning_rate * batch_size / 256,
        #                                betas=(0.9, 0.999), weight_decay=weight_decay
        #                                )

        self.optim = instant_optimizer(optimizer, self.model.parameters(), batch_size)

        summary(self.model, (1, 3, 32, 32), )

        if compile:
            self.model = torch.compile(self.model, fullgraph=False, )  # mode='max-autotune'

        lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))

        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_func, verbose=True)
        self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-5, max_lr=3e-4, step_size_up=10,
                                                              step_size_down=10, cycle_momentum=False)

        self.model, \
            self.optim, \
            self.train_dataloader, \
            self.val_dataloader = self.accelerator.prepare(self.model, self.optim, self.train_dataloader,
                                                           self.val_dataloader,
                                                           )
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

        self.save_model_path = save_model_path
        self.save_model_name = save_model_name
        self.pretrained_model_path = pretrained_model_path
        self.epoch = 0

    def train(self):
        best_val_acc = 0
        step_count = 0
        self.optim.zero_grad()

        for e in range(self.total_epoch):
            self.epoch = e
            self.model.train()
            losses = []
            train_step = len(self.train_dataloader)
            with tqdm(total=train_step, desc=f'Epoch {e + 1}/{self.total_epoch}', postfix=dict,
                      mininterval=0.3) as pbar:
                for img, label in self.train_dataloader:

                    z_router_losses = []
                    with self.accelerator.autocast():
                        step_count += 1
                        img = Normalize(CIFAR10_MEAN, CIFAR10_STD)(img)
                        # predicted_img, mask = model.train_forward(img)
                        predicted_img, mask = self.model(img)

                        if self.loss == 'l2':
                            loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                        else:
                            loss = torch.mean(torch.abs(predicted_img - img) * mask) / args.mask_ratio

                        # for transformer in self.model.encoder.transformer:
                        #     if not transformer.skip:
                        #         z_router_losses.append(transformer.entropy_loss)
                        #
                        # z_router_losses = torch.stack(z_router_losses, dim=0).mean()

                        # model.encoder.shuffle = PatchShuffle(0.1)
                        # loss_adv = trades_loss(model, img, img, optim,perturb_steps=3,beta=5.0)
                        # model.encoder.shuffle = PatchShuffle(args.mask_ratio)

                        # adv_img = fgsm_attack_data(model, img, img, epsilon=16 / 255)
                        # predicted_img_adv, mask_adv = model(img)
                        # loss_adv = torch.mean((predicted_img - img) ** 2 * mask_adv) / args.mask_ratio
                    # accelerator.backward(loss + 1e-4 * z_router_losses)
                    self.accelerator.backward(loss)
                    # accelerator.backward(loss+loss_adv)
                    # accelerator.backward(loss_adv)
                    # loss.backward()
                    if step_count % self.steps_per_update == 0:
                        self.accelerator.wait_for_everyone()
                        # self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                        self.optim.step()
                        self.optim.zero_grad()

                        self.accelerator.wait_for_everyone()
                    losses.append(loss.item())
                    pbar.set_postfix(**{'Loss': np.mean(losses),
                                        # 'z_router_losses': np.mean(z_router_losses.item())
                                        # 'Adv_Loss': np.mean(loss_adv.item())
                                        }
                                     )
                    pbar.update(1)
            self.lr_scheduler.step()

            # print(self.lr_scheduler.get_lr())

            avg_loss = sum(losses) / len(losses)
            # print(f'In epoch {e}, average traning loss is {avg_loss}.')

            ''' save model '''

            # print(model)

            if (e + 1) % self.save_every == 0:
                print('eval!!!!!!')
                self.eval()
                self.save()

            # torch.save(model, args.output_model_path)

    def eval(self):
        ''' visualize the first 16 predicted images on val dataset'''
        self.model.eval()
        with torch.no_grad():
            val_img = torch.stack([self.val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(self.device)
            predicted_val_img, mask = self.model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            # img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            # writer.add_image('mae_image', (img + 1) / 2, global_step=e)
            os.makedirs('save_images', exist_ok=True)
            torchvision.utils.save_image(DeNormalize(CIFAR10_MEAN, CIFAR10_STD, )(img), f'save_images/mae_img_{1}.png')
            # torchvision.utils.save_image((img + 1) / 2, f'mae_img_{1}.png')
            # torchvision.utils.save_image(img, f'mae_img_{1}.png')

    def save(self):
        print('Now Save Model!')
        assert self.save_model_path is not None and self.save_model_name is not None
        os.makedirs(self.save_model_path, exist_ok=True)
        torch.save(self.accelerator.get_state_dict(self.model._orig_mod if self.compile else self.model),
                   f'{self.save_model_path}/{self.save_model_name}_{self.epoch}.pt')
        # torch.save(self.model, args.output_model_path)

    def load(self):
        print('Now load Model!')
        """
        if pretrained_model_path is not None:
            mae_model.load_state_dict(torch.load(pretrained_model_path))
        """
        pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, )  # default=42
    parser.add_argument('-bs', '--batch_size', type=int, )  # default=4096
    parser.add_argument('--max_device_batch_size', type=int, )  # default=1024
    parser.add_argument('--base_learning_rate', type=float, )  # default=1.5e-4
    parser.add_argument('--weight_decay', type=float, )  # default=0.05
    parser.add_argument('--mask_ratio', type=float, default=0.75)  # default=0.75
    parser.add_argument('--total_epoch', type=int, )  # default 2000
    parser.add_argument('--warmup_epoch', type=int, )  # default=200
    parser.add_argument('--loss', type=str, )  # default='l2'
    parser.add_argument('--yaml_path', type=str, default='configs/mae/test_baseline.yaml')
    parser.add_argument('--aux_data_filename', type=str, default='/home/jtitor/Downloads/1m.npz')
    parser.add_argument('--save_every', type=int, )
    parser.add_argument('--compile', action='store_true', default=None)
    args = parser.parse_args()

    yaml_data = get_config(args)
    trainer = MaeTrainer(**yaml_data)
    trainer.train()
    # trainer.save()

"""


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_device_batch_size', type=int, default=256)
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

    model, optim, train_dataloader, lr_scheduler = accelerator.prepare(model, optim, train_dataloader, lr_scheduler)

    for e in range(args.total_epoch):
        model.train()
        losses = []
        acces = []
        train_step = len(train_dataloader)
        with tqdm(total=train_step, desc=f'Train Epoch {e + 1}/{args.total_epoch}', postfix=dict,
                  mininterval=0.3) as pbar:
            for img, label in iter(train_dataloader):
                z_router_losses = []
                with accelerator.autocast():
                    step_count += 1
                    img = img.to(device)
                    label = label.to(device)
                    logits = model(img)
                    loss = loss_fn(logits, label)  # F.softmax(logits, -1)

                    for transformer in model.transformer:
                        if not transformer.skip:
                            z_router_losses.append(transformer.entropy_loss)

                    z_router_losses = torch.stack(z_router_losses, dim=0).mean()

                acc = acc_fn(logits, label)
                # loss.backward()
                # accelerator.backward(loss+1e-4 * z_router_losses)  #  +1e-2 * z_router_losses
                accelerator.backward(loss)

                if step_count % steps_per_update == 0:
                    # accelerator.unscale_gradients(optimizer)
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                    optim.step()
                    optim.zero_grad()
                losses.append(loss.item())
                acces.append(acc.item())

                pbar.set_postfix(**{'Train Loss': np.mean(losses),
                                    'Tran accs': np.mean(acces),
                                    'z_router_losses': np.mean(z_router_losses.item())
                                    })
                pbar.update(1)

        lr_scheduler.step()
        avg_train_loss = sum(losses) / len(losses)
        avg_train_acc = sum(acces) / len(acces)
        # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

        model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            val_step = len(val_dataloader)
            with tqdm(total=val_step, desc=f'Val Epoch {e + 1}/{args.total_epoch}', postfix=dict,
                      mininterval=0.3) as pbar2:
                for img, label in iter(val_dataloader):
                    img = img.to(device)
                    label = label.to(device)
                    logits = model(img)
                    loss = loss_fn(logits, label)
                    acc = acc_fn(logits, label)
                    losses.append(loss.item())
                    acces.append(acc.item())

                    pbar2.set_postfix(**{'Val Loss': np.mean(losses),
                                         'Val accs': np.mean(acces)})
                    pbar2.update(1)
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            # print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        if avg_val_acc > best_val_acc:
            best_val_acc = avg_val_acc
            print(f'saving best model with acc {best_val_acc} at {e} epoch!')
            torch.save(model, args.output_model_path)
"""
