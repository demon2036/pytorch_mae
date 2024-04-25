from typing import Optional, List

import torch
import torch.nn.functional as F
import timm
import numpy as np

from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from timm.layers import Mlp, DropPath

# 这里可以用两个timm模型进行构建我们的结果
from timm.models.layers import trunc_normal_
# from timm.models.vision_transformer import Block
from functools import partial
import torch.nn as nn
from timm.models.vision_transformer import Attention, LayerScale


# class Block(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             init_values: Optional[float] = None,
#             drop_path: float = 0.,
#             act_layer: nn.Module = nn.GELU,
#             norm_layer: nn.Module = nn.LayerNorm,
#             mlp_layer: nn.Module = Mlp,
#     ) -> None:
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         self.norm2 = norm_layer(dim)
#         self.mlp = mlp_layer(
#             in_features=dim,
#             hidden_features=int(dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=proj_drop,
#         )
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts: int = 8, num_experts_per_tok: int = 2):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts

    def forward(self, inputs: torch.Tensor):
        inputs_shape = inputs.shape
        inputs = inputs.view(-1, inputs_shape[-1])

        gate_logits = self.gate(inputs)
        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)
        weights = F.softmax(weights, dim=1, dtype=torch.float).to(inputs.dtype)
        results = torch.zeros_like(inputs)
        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )
        # return results
        return results.view(inputs_shape)


class BlockMoE(nn.Module):
    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 4.,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            proj_drop: float = 0.,
            attn_drop: float = 0.,
            init_values: Optional[float] = None,
            drop_path: float = 0.,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.LayerNorm,
            mlp_layer: nn.Module = Mlp,
    ) -> None:
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.norm2 = norm_layer(dim)

        num_experts = 8

        self.mlp = MoeLayer(experts=[mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        ) for _ in range(num_experts)], gate=nn.Linear(dim, num_experts),)

        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


Block=BlockMoE
# class Block(nn.Module):
#     def __init__(
#             self,
#             dim: int,
#             num_heads: int,
#             mlp_ratio: float = 4.,
#             qkv_bias: bool = False,
#             qk_norm: bool = False,
#             proj_drop: float = 0.,
#             attn_drop: float = 0.,
#             init_values: Optional[float] = None,
#             drop_path: float = 0.,
#             act_layer: nn.Module = nn.GELU,
#             norm_layer: nn.Module = nn.LayerNorm,
#             mlp_layer: nn.Module = Mlp,
#     ) -> None:
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         self.attn = Attention(
#             dim,
#             num_heads=num_heads,
#             qkv_bias=qkv_bias,
#             qk_norm=qk_norm,
#             attn_drop=attn_drop,
#             proj_drop=proj_drop,
#             norm_layer=norm_layer,
#         )
#         self.ls1 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path1 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#         self.norm2 = norm_layer(dim)
#         self.mlp = mlp_layer(
#             in_features=dim,
#             hidden_features=int(dim * mlp_ratio),
#             act_layer=act_layer,
#             drop=proj_drop,
#         )
#         self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
#         self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
#         x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
#         return x


def random_indexes(size: int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)  # 打乱index
    backward_indexes = np.argsort(forward_indexes)  # 得到原来index的位置，方便进行还原
    return forward_indexes, backward_indexes


def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))


class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio) -> None:
        super().__init__()
        self.ratio = ratio

    def forward(self, patches: torch.Tensor):
        T, B, C = patches.shape  # length, batch, dim
        remain_T = int(T * (1 - self.ratio))

        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(
            patches.device)

        patches = take_indexes(patches, forward_indexes)  # 随机打乱了数据的patch，这样所有的patch都被打乱了
        patches = patches[:remain_T]  # 得到未mask的pacth [T*0.25, B, C]

        return patches, forward_indexes, backward_indexes


class MAE_Encoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=12,
                 num_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2, 1, emb_dim))

        # 对patch进行shuffle 和 mask
        self.shuffle = PatchShuffle(mask_ratio)

        # 这里得到一个 (3, dim, patch, patch)
        self.patchify = torch.nn.Conv2d(3, emb_dim, patch_size, patch_size)

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        # ViT的laynorm
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    # 初始化类别编码和向量编码
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes


class MAE_Decoder(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 num_layer=4,
                 num_head=3,
                 ) -> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((image_size // patch_size) ** 2 + 1, 1, emb_dim))

        self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])

        self.head = torch.nn.Linear(emb_dim, 3 * patch_size ** 2)
        self.patch2img = Rearrange('(h w) b (c p1 p2) -> b c (h p1) (w p2)', p1=patch_size, p2=patch_size,
                                   h=image_size // patch_size)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        backward_indexes = torch.cat(
            [torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat(
            [features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)],
            dim=0)
        features = take_indexes(features, backward_indexes)
        features = features + self.pos_embedding  # 加上了位置编码的信息

        features = rearrange(features, 't b c -> b t c')
        features = self.transformer(features)
        features = rearrange(features, 'b t c -> t b c')
        features = features[1:]  # remove global feature 去掉全局信息，得到图像信息

        patches = self.head(features)  # 用head得到patchs
        mask = torch.zeros_like(patches)
        mask[T:] = 1  # mask其他的像素全部设为 1
        mask = take_indexes(mask, backward_indexes[1:] - 1)
        img = self.patch2img(patches)  # 得到 重构之后的 img
        mask = self.patch2img(mask)

        return img, mask


class MAE_ViT(torch.nn.Module):
    def __init__(self,
                 image_size=32,
                 patch_size=2,
                 emb_dim=192,
                 encoder_layer=12,
                 encoder_head=3,
                 decoder_layer=4,
                 decoder_head=3,
                 mask_ratio=0.75,
                 ) -> None:
        super().__init__()

        self.encoder = MAE_Encoder(image_size, patch_size, emb_dim, encoder_layer, encoder_head, mask_ratio)
        self.decoder = MAE_Decoder(image_size, patch_size, emb_dim, decoder_layer, decoder_head)

    def forward(self, img):
        features, backward_indexes = self.encoder(img)
        predicted_img, mask = self.decoder(features, backward_indexes)
        return predicted_img, mask


class ViT_Classifier(torch.nn.Module):
    def __init__(self, encoder: MAE_Encoder, num_classes=10) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes,dtype=torch.float32)

    def forward(self, img):

        # print(img.max(),img.min())

        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


MAE_ViT_2_T_moe = partial(MAE_ViT, emb_dim=192, encoder_head=3, decoder_head=3, decoder_layer=2)
MAE_ViT_2_S_moe = partial(MAE_ViT, emb_dim=384, encoder_head=6, decoder_head=6, decoder_layer=2)
MAE_ViT_2_M_moe = partial(MAE_ViT, emb_dim=512, encoder_head=8, decoder_head=8, decoder_layer=2)
MAE_ViT_2_B_moe = partial(MAE_ViT, emb_dim=768, encoder_head=12, decoder_head=12, decoder_layer=2)

if __name__ == '__main__':
    shuffle = PatchShuffle(0.75)
    a = torch.rand(16, 2, 10)
    b, forward_indexes, backward_indexes = shuffle(a)
    print(b.shape)

    img = torch.rand(2, 3, 32, 32)
    encoder = MAE_Encoder()
    decoder = MAE_Decoder()
    features, backward_indexes = encoder(img)
    print(forward_indexes.shape)
    predicted_img, mask = decoder(features, backward_indexes)
    print(predicted_img.shape)
    loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)

    # torch.load('vit-t-mae.pth')
