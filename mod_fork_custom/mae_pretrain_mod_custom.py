import os
import argparse
import math
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms import ToTensor, Compose, Normalize
from torch.utils.data import DataLoader
import torch
from mod_model_fork_custom import MAE_ViT_2_T, MAE_ViT_2_S, MAE_ViT_2_M, MAE_ViT_2_B
from utils import setup_seed
from accelerate import Accelerator
from tqdm import tqdm
import numpy as np

accelerator = Accelerator(mixed_precision='fp16', )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('-bs', '--batch_size', type=int, default=4096)
    parser.add_argument('--max_device_batch_size', type=int, default=1024)
    parser.add_argument('--base_learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--total_epoch', type=int, default=2000)  # default 2000
    parser.add_argument('--warmup_epoch', type=int, default=200)
    parser.add_argument('--model_path', type=str, default='vit-t-mae.py')
    parser.add_argument('--loss', type=str, default='l2')
    parser.add_argument('--model', type=str, default='tiny')

    args = parser.parse_args()

    setup_seed(args.seed)

    batch_size = args.batch_size
    load_batch_size = min(args.max_device_batch_size, batch_size)

    assert batch_size % load_batch_size == 0
    steps_per_update = batch_size // load_batch_size

    train_dataset = torchvision.datasets.CIFAR10('../data', train=True, download=True,
                                                 transform=Compose(
                                                     [ToTensor(), Normalize(0.5, 0.5)]))  # Normalize(0.5, 0.5)
    val_dataset = torchvision.datasets.CIFAR10('../data', train=False, download=True,
                                               transform=Compose([ToTensor(), Normalize(0.5, 0.5)]))
    dataloader = DataLoader(train_dataset, load_batch_size, shuffle=True, num_workers=8, persistent_workers=True,
                            pin_memory=True)
    writer = SummaryWriter(os.path.join('../logs', 'cifar10', 'mae-pretrain'))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.model == 'tiny':
        model = MAE_ViT_2_T(mask_ratio=args.mask_ratio)
    elif args.model == 'small':
        model = MAE_ViT_2_S(mask_ratio=args.mask_ratio)
    elif args.model == 'mini':
        model = MAE_ViT_2_M(mask_ratio=args.mask_ratio)
    elif args.model == 'base':
        model = MAE_ViT_2_B(mask_ratio=args.mask_ratio)
    else:
        raise NotImplemented('')

    model = model.to(device)

    # model = torch.compile(model)

    optim = torch.optim.AdamW(model.parameters(), lr=args.base_learning_rate * args.batch_size / 256, betas=(0.9, 0.95),
                              weight_decay=args.weight_decay)

    lr_func = lambda epoch: min((epoch + 1) / (args.warmup_epoch + 1e-8),
                                0.5 * (math.cos(epoch / args.total_epoch * math.pi) + 1))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

    optim, dataloader, lr_scheduler = accelerator.prepare(optim, dataloader, lr_scheduler)
    model = accelerator.prepare_model(model)

    unwarp_model=accelerator.unwrap_model(model)
    accelerator.save(unwarp_model, args.model_path)



    step_count = 0
    optim.zero_grad()
    for e in range(args.total_epoch):
        model.train()
        losses = []
        train_step = len(dataloader)
        with tqdm(total=train_step, desc=f'Epoch {e + 1}/{args.total_epoch}', postfix=dict, mininterval=0.3) as pbar:
            for img, label in iter(dataloader):
                step_count += 1
                img = img.to(device)
                z_router_losses = []
                with accelerator.autocast():
                    # predicted_img, mask = model.train_forward(img)
                    predicted_img, mask = model(img)

                    if args.loss == 'l2':

                        loss = torch.mean((predicted_img - img) ** 2 * mask) / args.mask_ratio
                    else:
                        loss = torch.mean(torch.abs(predicted_img - img) * mask) / args.mask_ratio

                    for transformer in model.encoder.transformer:
                        if not transformer.skip:
                            z_router_losses.append(transformer.entropy_loss)

                    z_router_losses = torch.stack(z_router_losses, dim=0).mean()

                    # model.encoder.shuffle = PatchShuffle(0.1)
                    # loss_adv = trades_loss(model, img, img, optim,perturb_steps=3,beta=5.0)
                    # model.encoder.shuffle = PatchShuffle(args.mask_ratio)

                    # adv_img = fgsm_attack_data(model, img, img, epsilon=16 / 255)
                    # predicted_img_adv, mask_adv = model(img)
                    # loss_adv = torch.mean((predicted_img - img) ** 2 * mask_adv) / args.mask_ratio
                accelerator.backward(loss + 1e-4 * z_router_losses)
                # accelerator.backward(loss+loss_adv)
                # accelerator.backward(loss_adv)
                # loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                losses.append(loss.item())
                pbar.set_postfix(**{'Loss': np.mean(losses),
                                    'z_router_losses': np.mean(z_router_losses.item())
                                    # 'Adv_Loss': np.mean(loss_adv.item())
                                    }
                                 )
                pbar.update(1)
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        writer.add_scalar('mae_loss', avg_loss, global_step=e)
        # print(f'In epoch {e}, average traning loss is {avg_loss}.')

        ''' visualize the first 16 predicted images on val dataset'''
        model.eval()
        with torch.no_grad():
            val_img = torch.stack([val_dataset[i][0] for i in range(16)])
            val_img = val_img.to(device)
            predicted_val_img, mask = model(val_img)
            predicted_val_img = predicted_val_img * mask + val_img * (1 - mask)
            img = torch.cat([val_img * (1 - mask), predicted_val_img, val_img], dim=0)
            # img = rearrange(img, '(v h1 w1) c h w -> c (h1 h) (w1 v w)', w1=2, v=3)
            # writer.add_image('mae_image', (img + 1) / 2, global_step=e)
            torchvision.utils.save_image((img + 1) / 2, f'mae_img_{1}.png')
            # torchvision.utils.save_image(img, f'mae_img_{1}.png')
        ''' save model '''

        # print(model)

        # torch.save(model, args.output_model_path)
        torch.save(accelerator.get_state_dict(model), args.model_path)

