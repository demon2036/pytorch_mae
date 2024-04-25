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
from torch.cuda.amp import autocast

from torch.distributions import Categorical


def entropy(output):
    one_minus_output = 1 - output
    return -(output * torch.log2(output + 1e-7) + one_minus_output * torch.log2(one_minus_output + 1e-7)).mean()


class Router(nn.Module):
    def __init__(self, n_embed, num_experts):
        super().__init__()
        # layer for router logits
        self.topkroute_linear = nn.Linear(n_embed, num_experts, dtype=torch.float)

    def forward(self, mh_output):
        with torch.cuda.amp.autocast(enabled=False, dtype=torch.float32):
            # mh_ouput is the output tensor from multihead self attention block
            logits = self.topkroute_linear(mh_output)

        return logits


class MoeLayer(nn.Module):
    def __init__(self, experts: List[nn.Module], gate: nn.Module, num_experts: int = 8, num_experts_per_tok: int = 2,
                 ):
        super().__init__()
        assert len(experts) > 0
        self.experts = nn.ModuleList(experts)
        self.gate = gate
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.entropy_loss = None

    def forward(self, inputs: torch.Tensor, ):
        inputs_shape = inputs.shape
        inputs = inputs.view(-1, inputs_shape[-1])

        gate_logits = self.gate(inputs)

        weights, selected_experts = torch.topk(gate_logits, self.num_experts_per_tok)

        weights = F.softmax(weights, dim=1, dtype=torch.float).to(torch.float16)
        # weights=weights.to(inputs.dtype)

        results = torch.zeros_like(inputs)

        for i, expert in enumerate(self.experts):
            batch_idx, nth_expert = torch.where(selected_experts == i)
            results[batch_idx] += weights[batch_idx, nth_expert, None] * expert(
                inputs[batch_idx]
            )

        if self.training:
            gate_logits = F.softmax(gate_logits, dim=1, dtype=torch.float)
            dist = Categorical(gate_logits)
            loss = -dist.entropy().mean()
            # loss=F.mse_loss(gate_logits,torch.zeros_like(gate_logits)+1/8)
            self.entropy_loss = loss

        return results.view(inputs_shape)


def softmax(x: torch.Tensor, dim: int | tuple[int, ...]) -> torch.Tensor:
    """
    Compute the softmax along the specified dimensions.
    This function adds the option to specify multiple dimensions

    Args:
        x (torch.Tensor): Input tensor.
        dims (int or tuple[int]): The dimension or list of dimensions along which the softmax probabilities are computed.

    Returns:
        torch.Tensor: Output tensor containing softmax probabilities along the specified dimensions.
    """
    max_vals = torch.amax(x, dim=dim, keepdim=True)
    e_x = torch.exp(x - max_vals)
    sum_exp = e_x.sum(dim=dim, keepdim=True)
    return e_x / sum_exp


class SoftMoELayerWrapper(nn.Module):
    """
    A wrapper class to create a Soft Mixture of Experts layer.

    From "From Sparse to Soft Mixtures of Experts"
    https://arxiv.org/pdf/2308.00951.pdf
    """

    def __init__(
            self,
            dim: int,
            num_experts: int,
            slots_per_expert: int,
            experts,
            normalize: bool = True,

    ) -> None:
        """
        Args:
            dim (int): Dimensionality of input features.
            num_experts (int): Number of experts.
            slots_per_expert (int): Number of token slots per expert.
            normalize (bool): Normalize input and phi (sec. 2.3 from paper)
        """
        super().__init__()

        self.dim = dim
        self.num_experts = num_experts
        self.slots_per_expert = slots_per_expert
        self.normalize = normalize

        # Initialize phi and normalization scaling factor
        self.phi = nn.Parameter(torch.zeros(dim, num_experts, slots_per_expert))
        if self.normalize:
            self.scale = nn.Parameter(torch.ones(1))

        # Initialize phi using LeCun normal initialization
        # https://github.com/google-research/vmoe/blob/662341d007650d5bbb7c6a2bef7f3c759a20cc7e/vmoe/projects/soft_moe/router.py#L49C1-L49C1
        nn.init.normal_(self.phi, mean=0, std=1 / dim ** 0.5)

        # Create a list of expert networks
        self.experts = nn.ModuleList(experts)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the Soft-MoE layer (algorithm 1 from paper).

        Args:
            x (torch.Tensor): Input tensor of shape [batch_size, seq_len, input_dim].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, seq_len, input_dim].
        """
        assert (
                x.shape[-1] == self.dim
        ), f"Input feature dim of {x.shape[-1]} does not match layer dim of {self.dim}"
        assert (
                len(x.shape) == 3
        ), f"Input expected to have 3 dimensions but has {len(x.shape)}"

        phi = self.phi

        # Normalize input and phi
        if self.normalize:
            x = F.normalize(x, dim=2)  # [b, m, d]
            phi = self.scale * F.normalize(phi, dim=0)  # [d, n, p]

        # Compute dispatch and combine weights
        logits = torch.einsum("bmd,dnp->bmnp", x, phi)
        d = softmax(logits, dim=1)
        c = softmax(logits, dim=(2, 3))

        # Compute input slots as weighted average of input tokens using dispatch weights
        xs = torch.einsum("bmd,bmnp->bnpd", x, d)

        # Apply expert to corresponding slots
        ys = torch.stack(
            [f_i(xs[:, i, :, :]) for i, f_i in enumerate(self.experts)], dim=1
        )


        # Compute output tokens as weighted average of output slots using combine weights
        y = torch.einsum("bnpd,bmnp->bmd", ys, c, )

        # print(y.shape)

        return y


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
            seq_length=None
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

        num_experts = 16

        self.mlp = SoftMoELayerWrapper(experts=[mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        ).cuda() for _ in range(num_experts)], dim=dim, num_experts=num_experts,
                                       slots_per_expert=(seq_length // num_experts))

        # self.mlp = mlp_layer(
        #     in_features=dim,
        #     hidden_features=int(dim * mlp_ratio),
        #     act_layer=act_layer,
        #     drop=proj_drop,
        # )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, train=False) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x), )))
        return x


class Block(nn.Module):
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
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x


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

        seq_length = (image_size // patch_size) ** 2

        # self.transformer = torch.nn.Sequential(*[Block(emb_dim, num_head) for _ in range(num_layer)])
        self.transformer = torch.nn.ModuleList(
            [BlockMoE(emb_dim, num_head, seq_length=seq_length) for _ in range(num_layer)])

        # ViT的laynorm
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    # 初始化类别编码和向量编码
    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, img, train=False):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding

        patches, forward_indexes, backward_indexes = self.shuffle(patches)

        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        for transformer in self.transformer:
            patches = transformer(patches, train=train)

        features = self.layer_norm(patches)
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
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes, dtype=torch.float32)

    def forward(self, img, train=False):
        # print(img.max(),img.min())

        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        for transformer in self.transformer:
            patches = transformer(patches, train=train)

        features = self.layer_norm(patches)
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits


MAE_ViT_2_T_moe = partial(MAE_ViT, emb_dim=192, encoder_head=3, decoder_head=3, decoder_layer=2)
MAE_ViT_2_S_moe = partial(MAE_ViT, emb_dim=384, encoder_head=6, decoder_head=6, decoder_layer=2)
MAE_ViT_2_M_moe = partial(MAE_ViT, emb_dim=512, encoder_head=8, decoder_head=8, decoder_layer=2)
MAE_ViT_2_B_moe = partial(MAE_ViT, emb_dim=768, encoder_head=12, decoder_head=12, decoder_layer=2)

if __name__ == '__main__':
    # shuffle = PatchShuffle(0.75)
    # a = torch.rand(16, 2, 10)
    # b, forward_indexes, backward_indexes = shuffle(a)
    # print(b.shape)
    #
    # img = torch.rand(2, 3, 32, 32)
    # encoder = MAE_Encoder()
    # decoder = MAE_Decoder()
    # features, backward_indexes = encoder(img)
    # print(forward_indexes.shape)
    # predicted_img, mask = decoder(features, backward_indexes)
    # print(predicted_img.shape)
    # loss = torch.mean((predicted_img - img) ** 2 * mask / 0.75)

    # loss.backward()

    # torch.load('vit-t-mae.pth')
    pass

    # b,n,e,d=1,4,16,8
    # shape=(b,n,e,d)
    # model=nn.ModuleList([nn.Linear(d,d) for i in range(8) ])
    #
    # x=torch.rand(shape)
    #
    # for i,expert in enumerate(model):
    #     y=
