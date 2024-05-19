import collections
import functools
import os
import argparse
import math

import numpy as np
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.transforms import ToTensor, Compose, Normalize
from tqdm import tqdm

from base_trainer import BaseTrainer
from mod_fork_every2.mod_model_fork_every2 import *
from normal_utils import CIFAR10_MEAN, CIFAR10_STD
from utils import setup_seed, get_obj_from_str, acc_fn, get_config
from accelerate import Accelerator
from autoaugment import CIFAR10Policy
from torchinfo import summary


def get_model_finetune(model: str = None, pretrained_model_path: str = None, is_mae=False):
    assert model is not None
    print(model)
    mae_model = get_obj_from_str(model)()

    if is_mae:

        if pretrained_model_path is not None:
            data = torch.load(pretrained_model_path)
            # print(data)

            # for key in data.keys():
            #     print(key)
            # while True:
            #     pass

            # data= collections.OrderedDict([(k.replace('_orig_mod.',''), v)  for k, v in data.items()])
            # print(type(data))

            # while True:
            #     pass

            mae_model.load_state_dict(data)

        model = ViT_Classifier(mae_model.encoder, num_classes=10)
    else:
        model = mae_model
    return model


def lr_func(epoch, warmup_epoch, total_epoch):
    return min((epoch + 1) / (warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / total_epoch * math.pi) + 1))


class MaeFinetuneTrainer(BaseTrainer):
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
                 loss_fn=torch.nn.CrossEntropyLoss(),
                 model_instant_function=get_model_finetune,
                 model_target: str = None,
                 save_model_name: str = None,
                 mixed_precision='fp16',
                 save_every=500,
                 compile=False,
                 ):
        super().__init__(seed, batch_size, max_device_batch_size, total_epoch, mixed_precision, save_every=save_every)

        self.loss_fn = loss_fn
        self.model = model_instant_function(model_target, pretrained_model_path)

        summary(self.model, (1, 3, 32, 32), )
        if compile:
            self.model = torch.compile(self.model, fullgraph=False, )  # mode='max-autotune'

        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=functools.partial(lr_func,
        #                                                                                               warmup_epoch=warmup_epoch,
        #                                                                                               total_epoch=total_epoch),
        #                                                       verbose=True)

        self.save_model_path = save_model_path
        self.save_model_name = save_model_name
        # self.pretrained_model_path = pretrained_model_path

        self.base_learning_rate = base_learning_rate
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.warmup_epoch = warmup_epoch
        self.total_epoch = total_epoch

    def train(self):

        self.accelerator = Accelerator(mixed_precision='bf16')

        self.optim = torch.optim.AdamW(self.model.parameters(),
                                       lr= self.base_learning_rate * self.batch_size / 256,
                                       betas=(0.9, 0.999), weight_decay=self.weight_decay
                                       )
        # self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=functools.partial(lr_func,
        #                                                                                               warmup_epoch=self.warmup_epoch,
        #                                                                                               total_epoch=self.total_epoch),
        #                                                       verbose=True)

        # self.model, \
        #     self.optim, \
        #     self.train_dataloader, \
        #     self.val_dataloader, \
        #     self.lr_scheduler = self.accelerator.prepare(self.model, self.optim, self.train_dataloader,
        #                                                  self.val_dataloader,
        #                                                  self.lr_scheduler)

        self.model, \
            self.optim, \
            self.train_dataloader, \
            self.val_dataloader, \
            = self.accelerator.prepare(self.model, self.optim, self.train_dataloader,
                                                         self.val_dataloader,
                                                         )


        best_val_acc = 0
        step_count = 0
        self.optim.zero_grad()
        for e in range(self.total_epoch):
            self.model.train()
            losses = []
            acces = []
            train_step = len(self.train_dataloader)
            with tqdm(total=train_step, desc=f'Train Epoch {e + 1}/{self.total_epoch}', postfix=dict,
                      mininterval=0.3, disable=not self.accelerator.is_main_process) as pbar:
                for img, label in iter(self.train_dataloader):
                    z_router_losses = []
                    with self.accelerator.autocast():
                        step_count += 1
                        img = Normalize(CIFAR10_MEAN, CIFAR10_STD)(img)
                        logits = self.model(img)
                        loss = self.loss_fn(logits, label)  # F.softmax(logits, -1)

                        # for transformer in self.model.transformer:
                        #     if not transformer.skip:
                        #         z_router_losses.append(transformer.entropy_loss)
                        #
                        # z_router_losses = torch.stack(z_router_losses, dim=0).mean()

                    acc = acc_fn(logits, label)
                    # loss.backward()
                    # accelerator.backward(loss+1e-4 * z_router_losses)  #  +1e-2 * z_router_losses
                    self.accelerator.backward(8 *loss)

                    if step_count % self.steps_per_update == 0:
                        self.accelerator.wait_for_everyone()
                        self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                        self.optim.step()
                        self.optim.zero_grad()
                    losses.append(loss.item())
                    acces.append(acc.item())

                    pbar.set_postfix(**{'Train Loss': np.mean(losses),
                                        'Tran accs': np.mean(acces),
                                        # 'z_router_losses': np.mean(z_router_losses.item())
                                        })
                    pbar.update(1)

            # self.lr_scheduler.step()
            avg_train_loss = sum(losses) / len(losses)
            avg_train_acc = sum(acces) / len(acces)
            # print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

            avg_val_acc = self.eval(e)

            if avg_val_acc > best_val_acc:
                best_val_acc = avg_val_acc
                print(f'saving best model with acc {best_val_acc} at {e} epoch!')
                self.save()

    def eval(self, epoch):
        self.model.eval()
        with torch.no_grad():
            losses = []
            acces = []
            val_step = len(self.val_dataloader)
            with tqdm(total=val_step, desc=f'Val Epoch {epoch + 1}/{self.total_epoch}', postfix=dict,
                      mininterval=0.3) as pbar2:
                for img, label in iter(self.val_dataloader):
                    img = img
                    label = label
                    img = Normalize(CIFAR10_MEAN, CIFAR10_STD)(img)
                    logits = self.model(img)
                    loss = self.loss_fn(logits, label)
                    acc = acc_fn(logits, label)
                    losses.append(loss.item())
                    acces.append(acc.item())

                    pbar2.set_postfix(**{'Val Loss': np.mean(losses),
                                         'Val accs': np.mean(acces)})
                    pbar2.update(1)
            avg_val_loss = sum(losses) / len(losses)
            avg_val_acc = sum(acces) / len(acces)
            print(
                f'In epoch {epoch}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')
        return avg_val_acc

    def save(self):
        assert self.save_model_path is not None and self.save_model_name is not None
        os.makedirs(self.save_model_path, exist_ok=True)
        torch.save(self.accelerator.get_state_dict(self.model), f'{self.save_model_path}/{self.save_model_name}.pt')
        # torch.save(self.model, args.output_model_path)

    def load(self):
        """
        if pretrained_model_path is not None:
            mae_model.load_state_dict(torch.load(pretrained_model_path))
        """
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, )  # default=2022
    parser.add_argument('--batch_size', type=int, )  # default=128
    parser.add_argument('--max_device_batch_size', type=int, )  # default=256
    parser.add_argument('--base_learning_rate', type=float, )  # default=1e-3
    parser.add_argument('--weight_decay', type=float, )  # default=0.05
    parser.add_argument('--total_epoch', type=int, )  # default=100
    parser.add_argument('--warmup_epoch', type=int, )  # default=5
    parser.add_argument('--pretrained_model_path', type=str, )
    parser.add_argument('--save_every', type=int, )
    parser.add_argument('--yaml_path', type=str,
                        default='configs/vit/wideresnet/wideresnet_28_10.yaml')  # 'configs/vit/baseline/tiny.yaml'

    args = parser.parse_args()
    # print('Using Default Config From Yaml')
    # yaml_data = read_yaml(args.yaml_path)
    # yaml_data.update({'pretrained_model_path': 'mod_mae_custom.pt'})

    yaml_data = get_config(args)

    trainer = MaeFinetuneTrainer(**yaml_data, )
    trainer.train()


if __name__ == "__main__":
    main()
