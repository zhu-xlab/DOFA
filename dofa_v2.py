# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# --------------------------------------------------------

from functools import partial
from wave_dynamic_layer import (
    Dynamic_MLP_OFA,
)

import torch
import torch.nn as nn
from timm.models.vision_transformer import VisionTransformer
import math
import pdb



class DOFAViT(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""

    def __init__(
        self,
        img_size=224,
        patch_size=14,
        drop_rate=0.0,
        out_indices=None,
        drop_path_rate=0.0,
        embed_dim=768,
        depth=24,
        num_heads=16,
        wv_planes=128,
        num_classes=45,
        global_pool=True,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.wv_planes = wv_planes
        self.out_indices = out_indices

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = Dynamic_MLP_OFA(
            wv_planes=128, inter_dim=128, kernel_size=14, embed_dim=embed_dim
        )
        self.img_size = img_size
        if isinstance(img_size, tuple):
            self.img_size = self.img_size[0]

        self.num_patches = (self.img_size // patch_size) ** 2
        self.patch_embed.num_patches = self.num_patches
        model_args = dict(patch_size=patch_size, embed_dim=embed_dim,
                depth=depth, num_heads=num_heads, init_values=1e-5, num_classes=0, dynamic_img_size=True)
        self.model = VisionTransformer(**model_args)
        del self.model.patch_embed.proj
        self.dynamic_img_size = True
        self.waves = None
        self.norm = norm_layer(embed_dim)

    def forward_features(self, x, wave_list=None):
        # embed patches
        if wave_list is None:
            wave_list = [0.665,0.56,0.49]
            #set to RGB by default
        wavelist = torch.tensor(wave_list, device=x.device).float()
        self.waves = wavelist
        x, _ = self.patch_embed(x, self.waves)
        B,HW,C = x.shape
        hw = int(math.sqrt(HW))
        hw_shape = (hw, hw)
        if self.dynamic_img_size:
            x = x.view(B,hw,hw,C)
        x = self.model._pos_embed(x)
        x = self.model.patch_drop(x)
        x = self.model.norm_pre(x)
        out_features = []

        # apply Transformer blocks
        for i, blk in enumerate(self.model.blocks):
            x = blk(x)
            if i in self.out_indices:
                out = x[:, 1:]
                B, _, C = out.shape
                out = (
                    out.reshape(B, hw_shape[0], hw_shape[1], C)
                    .permute(0, 3, 1, 2)
                    .contiguous()
                )
                out_features.append(out)

        x = self.model.norm(x)
        return out_features

    def forward(self, x, wave_list=None):
        x = self.forward_features(x, wave_list)
        return x


def vit_base_patch14(**kwargs):
    model = DOFAViT(
        out_indices=[4, 6, 10, 11],
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


def vit_large_patch14(**kwargs):
    model = DOFAViT(
        out_indices=[5, 11, 17, 23],
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model


if __name__=='__main__':
    #"https://huggingface.co/earthflow/DOFA/resolve/main/dofav2_vit_base_e150.pth"
    #"https://huggingface.co/earthflow/DOFA/resolve/main/dofav2_vit_large_e150.pth"
    size = "base"
    checkpoint_path = "dofav2_vit_base_e150.pth" if size=='base' else "dofav2_vit_large_e150.pth"
    check_point = torch.load(checkpoint_path)
    vit_model = vit_base_patch14()
    vit_model.load_state_dict(check_point, strict=True)
    vit_model(torch.randn([1,3,256,256]))
    vit_model(torch.randn([1,3,512,512]))
