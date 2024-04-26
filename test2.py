import os
import argparse
import math
import time

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
# import torch_xla.distributed.xla_multiprocessing as xmp
from accelerate import Accelerator


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
        self.accelerator = None
        self.compile = compile
        self.loss = loss
        self.model = model_instant_function(model_target, mask_ratio)
        # self.optim = torch.optim.AdamW(self.model.parameters(),
        #                                lr=base_learning_rate * batch_size / 256,
        #                                betas=(0.9, 0.999), weight_decay=weight_decay
        #                                )

        # self.optim = instant_optimizer(optimizer, self.model.parameters(), batch_size)

        self.optim = instant_optimizer(optimizer, self.model.parameters(), batch_size)
        # self.optim = torch.optim.AdamW(self.model.parameters(),lr=1e-3)

        # summary(self.model, (1, 3, 32, 32), )

        if compile:
            import torch_xla.core.xla_model as xm
            device = xm.xla_device()
            self.model = torchvision.models.resnet18().to(device)
            self.model = torch.compile(self.model, backend='torchxla_trace_once' )  # mode='max-autotune'

        lr_func = lambda epoch: min((epoch + 1) / (warmup_epoch + 1e-8),
                                    0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_func, verbose=True)
        # self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(self.optim, base_lr=1e-5, max_lr=3e-4, step_size_up=10,
        #                                                       step_size_down=10, cycle_momentum=False)

        self.save_model_path = save_model_path
        self.save_model_name = save_model_name
        self.pretrained_model_path = pretrained_model_path
        self.epoch = 0

        self.mask_ratio = mask_ratio

    def train(self):

        self.accelerator = Accelerator(mixed_precision='bf16', )
        print(self.accelerator.device,self.accelerator.mixed_precision)
        # self.model = torch.compile(self.model,fullgraph=True, backend='openxla')

        self.model, self.optim, \
            self.train_dataloader, \
            self.val_dataloader, self.lr_scheduler = self.accelerator.prepare(self.model, self.optim,
                                                                              self.train_dataloader,
                                                                              self.val_dataloader, self.lr_scheduler
                                                                              )

        # self.model, self.optim, \
        #     self.train_dataloader, \
        #     self.val_dataloader = self.accelerator.prepare(self.model, self.optim,
        #                                                    self.train_dataloader,
        #                                                    self.val_dataloader,
        #                                                    )

        # self.steps_per_update = self.batch_size // self.load_batch_size // self.accelerator.num_processes
        self.steps_per_update = self.batch_size // self.load_batch_size // self.accelerator.num_processes

        # self.steps_per_update = 1

        print(f'num_processes:{self.accelerator.num_processes} steps_per_update:{self.steps_per_update} ')
        # model=self.accelerator.prepare(model)

        best_val_acc = 0
        step_count = 0
        self.optim.zero_grad()

        for e in range(self.total_epoch):
            self.epoch = e
            self.model.train()
            losses = []
            train_step = len(self.train_dataloader)
            with tqdm(total=train_step, desc=f'Epoch {e + 1}/{self.total_epoch}', postfix=dict,
                      mininterval=0.3, disable=not self.accelerator.is_main_process) as pbar:
                for img, label in self.train_dataloader:

                    with self.accelerator.autocast():
                        # if self.accelerator.is_main_process:
                        #     print(img.shape)
                        # print(img.shape)
                        step_count += 1
                        img = Normalize(CIFAR10_MEAN, CIFAR10_STD)(img)


                        # predicted_img, mask = model.train_forward(img)
                        predicted_img, mask = self.model(img)

                        if self.loss == 'l2':
                            loss = torch.mean((predicted_img - img) ** 2 * mask) / self.mask_ratio
                        else:
                            loss = torch.mean(torch.abs(predicted_img - img) * mask) / self.mask_ratio


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

                    self.accelerator.backward(loss)#/ self.steps_per_update
                    # time.sleep(2)

                    # accelerator.backward(loss+loss_adv)
                    # accelerator.backward(loss_adv)
                    # loss.backward()
                    if step_count % self.steps_per_update == 0:
                        self.accelerator.wait_for_everyone()
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                        self.optim.step()
                        self.optim.zero_grad()

                        self.accelerator.wait_for_everyone()

                    losses.append(self.accelerator.gather(loss).detach().cpu())
                    if self.accelerator.is_main_process:
                        # losses=self.accelerator.ga

                        pbar.set_postfix(**{'Loss': np.mean(losses),
                                            # 'z_router_losses': np.mean(z_router_losses.item())
                                            # 'Adv_Loss': np.mean(loss_adv.item())
                                            }
                                         )
                        pbar.update(1)

            self.lr_scheduler.step()

            # print(self.lr_scheduler.get_lr())

            # avg_loss = sum(losses) / len(losses)
            # print(f'In epoch {e}, average traning loss is {avg_loss}.')

            ''' save model '''

            # print(model)

            # if (e + 1) % self.save_every == 0:
            #     print('eval!!!!!!')
            #     self.eval()
            #     self.save()

            # torch.save(model, args.output_model_path)

            """ """

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


def main():
    print('hi')

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
    parser.add_argument('--yaml_path', type=str, default='configs/mae/baseline/tiny.yaml')#'configs/mae/moe_soft/tiny.yaml'
    parser.add_argument('--aux_data_filename', type=str, default='/home/jtitor/Downloads/1m.npz')
    parser.add_argument('--save_every', type=int, )
    parser.add_argument('--compile', action='store_true', default=None)
    args = parser.parse_args()

    yaml_data = get_config(args)
    trainer = MaeTrainer(**yaml_data)

    trainer.train()

    # from accelerate import notebook_launcher, Accelerator
    #
    # notebook_launcher(trainer.train, num_processes=8)


if __name__ == "__main__":
    main()

# from accelerate import Accelerator
#
#
# def main():
#     # Accelerator instance.
#     accelerator = Accelerator()
#     print(accelerator.num_processes)
#
#     # Begin training
#     # train(opts, accelerator)
#
#     print("OMFG, training finished!")


# from accelerate import notebook_launcher, Accelerator

# notebook_launcher(trainer.train, num_processes=8)

# xmp.spawn(test,(trainer,))

# trainer.train()

# trainer.save()
