import os
import argparse
import math
import numpy as np
import torch
import torchvision
from accelerate import Accelerator
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from base_trainer import BaseTrainer
from normal_utils import CIFAR10_MEAN, CIFAR10_STD, DeNormalize
from utils import get_obj_from_str, acc_fn, get_config, json_print, print_with_seperator
import torch.nn as nn
import yaml
import json
from torchinfo import summary
import torch.optim.adamw
from at_helper import mae_feat_loss
import torch.nn.functional as F


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


"""
def adv_loss(model,
             x_natural, x_adv, loss_start=0
             ):
    criterion = nn.L1Loss()
    feats = model.forward_feat(x_natural)[loss_start:]

    losses = 0
    feats_adv = model.forward_feat(x_adv)[loss_start:]
    for feat, feat_adv in zip(feats, feats_adv):
        b, n, d = feat.shape
        feat_mean = feat.mean(dim=[-1]).reshape(b, n, 1)
        feat_std = feat.mean(dim=[-1]).reshape(b, n, 1)
        feat_normal = (feat - feat_mean) / feat_std
        feat_adv_normal = (feat_adv - feat_mean) / feat_std
        loss = criterion(feat_adv_normal, feat_normal)
        losses += loss / len(feats_adv)

    return losses
"""


def adv_loss(model,
             x_natural, x_adv, loss_start=0
             ):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    feats = model.forward_feat(x_natural)[loss_start:]

    losses = 0
    feats_adv = model.forward_feat(x_adv)
    feats_adv = feats_adv[loss_start:]
    for feat, feat_adv in zip(feats, feats_adv):
        loss = criterion_kl(torch.log_softmax(feat_adv, -1), torch.softmax(feat, -1))
        losses += loss / len(feats_adv)

    return losses, feats, feats_adv


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

                 ):
        super().__init__(seed, batch_size, max_device_batch_size, total_epoch, mixed_precision, use_aux_dataset,
                         unsup_fraction, aux_data_filename, save_every=save_every, transform=True)
        self.accelerator = None
        self.compile = compile
        self.loss = loss
        self.model = model_instant_function(model_target, mask_ratio)
        # self.optim = torch.optim.AdamW(self.model.parameters(),
        #                                lr=base_learning_rate * batch_size / 256,
        #                                betas=(0.9, 0.999), weight_decay=weight_decay
        #                                )

        self.optim = instant_optimizer(optimizer, self.model.parameters(), batch_size)

        summary(self.model, (1, 3, 32, 32), )

        if compile:
            self.model = torch.compile(self.model, fullgraph=True,
                                       mode='reduce-overhead')  # mode='max-autotune'  mode='reduce-overhead'

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
        self.accelerator = Accelerator(mixed_precision='fp16')
        print(self.accelerator.device, self.accelerator.mixed_precision)
        self.model, \
            self.optim, \
            self.train_dataloader, \
            self.val_dataloader = self.accelerator.prepare(self.model, self.optim, self.train_dataloader,
                                                           self.val_dataloader,
                                                           )
        self.lr_scheduler = self.accelerator.prepare(self.lr_scheduler)

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

                    with self.accelerator.autocast():
                        img_normal = Normalize(CIFAR10_MEAN, CIFAR10_STD)(img)
                        step_count += 1

                        loss_start = 0
                        x_adv = mae_feat_loss(self.model, x_natural=img, loss_start=loss_start, steps=10)

                        loss_adv, feats, feats_adv = adv_loss(self.model, img, x_adv, loss_start=loss_start)

                        feats = feats[-1]
                        feats_adv = feats_adv[-1]

                        feats = F.normalize(feats, dim=-1)
                        feats_adv = F.normalize(feats_adv, dim=-1)

                        labels = torch.arange(0, feats.shape[0], device=feats.device).long()

                        logit_scale = self.model.logit_scale.exp()
                        logits_per_ori = logit_scale * (feats @ feats_adv.T)
                        logits_per_adv = logit_scale * (feats_adv @ feats.T)
                        contrast_loss = (F.cross_entropy(logits_per_ori, labels) + F.cross_entropy(logits_per_adv,
                                                                                                   labels)) / 2
                        predicted_img, mask = self.model(img)
                        if self.loss == 'l2':
                            loss = torch.mean((predicted_img - img_normal) ** 2 * mask) / self.mask_ratio
                        else:
                            loss = torch.mean(torch.abs(predicted_img - img_normal) * mask) / self.mask_ratio

                    self.accelerator.backward((loss_adv + loss + contrast_loss) / self.steps_per_update)

                    if step_count % self.steps_per_update == 0:
                        self.accelerator.wait_for_everyone()
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                        self.optim.step()
                        self.optim.zero_grad()

                        self.accelerator.wait_for_everyone()
                    losses.append(loss.item())
                    pbar.set_postfix(**{
                        'Loss': np.mean(losses),
                        'Loss Adv': loss_adv.item(),
                        'contrast_loss': contrast_loss.item(),
                        'logit_scale': logit_scale.item()
                    }
                                     )
                    pbar.update(1)
            self.lr_scheduler.step()
            ''' save model '''
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
            val_img = val_img.to(self.accelerator.device)
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


def test(trainer):
    print('hi')

    trainer.train()


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
    parser.add_argument('--yaml_path', type=str, default='configs/mae/adv_feat_clip/tiny.yaml')
    parser.add_argument('--aux_data_filename', type=str, default='/home/jtitor/Downloads/1m.npz')
    parser.add_argument('--save_every', type=int, )
    parser.add_argument('--compile', action='store_true', default=None)
    args = parser.parse_args()

    yaml_data = get_config(args)
    trainer = MaeTrainer(**yaml_data)

    trainer.train()

    # from accelerate import notebook_launcher, Accelerator
    #
    # notebook_launcher(test,(trainer,) , num_processes=8)
    #
    # pass

    # trainer.train()

    # trainer.save()
