from functools import partial

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block

from wave_dynamic_layer import Dynamic_MLP_OFA


class DOFASegmentationViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.out_indices = out_indices or []
        self.img_size = img_size[0] if isinstance(img_size, tuple) else img_size
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128,
            inter_dim=128,
            kernel_size=patch_size,
            embed_dim=embed_dim,
        )
        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False
        )
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    drop_path=drop_path_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

    def forward(self, x, wave_list):
        wavelengths = torch.tensor(wave_list, device=x.device).float()
        x, _ = self.patch_embed(x, wavelengths)
        hw = self.img_size // self.patch_embed.kernel_size
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        outputs = []
        for index, block in enumerate(self.blocks):
            x = block(x)
            if index in self.out_indices:
                out = x[:, 1:]
                batch, _, channels = out.shape
                out = out.reshape(batch, hw, hw, channels).permute(0, 3, 1, 2).contiguous()
                outputs.append(out)
        return outputs


def vit_base_patch16(**kwargs):
    return DOFASegmentationViT(
        out_indices=[3, 5, 7, 11],
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_large_patch16(**kwargs):
    return DOFASegmentationViT(
        out_indices=[7, 11, 15, 23],
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )


def vit_huge_patch14(**kwargs):
    return DOFASegmentationViT(
        out_indices=[7, 15, 23, 31],
        patch_size=14,
        embed_dim=1280,
        depth=32,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
