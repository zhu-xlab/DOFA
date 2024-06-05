# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial

import torch
import torch.nn as nn

from timm.models.vision_transformer import PatchEmbed, Block
from wave_dynamic_layer import Dynamic_MLP_OFA, Dynamic_MLP_Decoder, GaussianFourierFeatureTransform

from util.pos_embed import get_2d_sincos_pos_embed, get_1d_sincos_pos_embed_from_grid_torch
from util.misc import CLIPLoss
import timm
import random
import pdb


class MaskedAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=[2,9,3,202,4],
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, norm_pix_loss=False):
        super().__init__()
        self.in_chans = in_chans
        self.wv_planes = 128
        self.patch_size = (patch_size,patch_size)

        self.patch_embed = Dynamic_MLP_OFA(wv_planes=128, inter_dim=128, kernel_size=16, embed_dim=embed_dim)

        #num_patches = self.patch_embed_s1.num_patches
        self.num_patches = (img_size // patch_size) ** 2
        self.waves = None

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        # MAE decoder specifics
        # --------------------------------------------------------------------------
        self.teacher_alpha = 1.0
        self.cos = nn.CosineSimilarity(dim=1)
        self.teacher_avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.projector = torch.nn.Linear(embed_dim, embed_dim)
        # --------------------------------------------------------------------------

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = Dynamic_MLP_Decoder(wv_planes=128, inter_dim=128, kernel_size=16, decoder_embed=decoder_embed_dim)

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

        # load per-trained weights for the continoul pretraining
        self.teacher = timm.create_model('vit_base_patch16_224.augreg2_in21k_ft_in1k', pretrained=True)
        pre_state_dict = self.teacher.state_dict()
        del pre_state_dict['patch_embed.proj.weight']
        del pre_state_dict['patch_embed.proj.bias']
        wg_weights = torch.load('weight_generator_1000_0.01_er50k.pt')
        msg = self.load_state_dict(pre_state_dict, strict=False)
        self.patch_embed.weight_generator.load_state_dict(wg_weights['weight_generator'])
        self.patch_embed.fclayer.load_state_dict(wg_weights['fclayer'])
        print('**************************************************')
        n_parameters = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(n_parameters)
        print('**************************************************')



    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], imgs.shape[1], h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * imgs.shape[1]))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        in_chan = x.shape[2] / (p**2)
        x = x.reshape(shape=(x.shape[0], h, w, p, p, in_chan))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], in_chan, h * p, h * p))
        return imgs

    def random_select_channels(self, imgs, wave_list):
        """
        imgs: input origginal
        return: new_imgs with randomly selected channels, wave_list
        """
        batch_size, num_channels, height, width = imgs.shape
        num_selected_channels = random.randint(int(num_channels*0.75)+1,num_channels)
        selected_indices = torch.randperm(num_channels)[:num_selected_channels]
        selected_channels = imgs[:, selected_indices, :, :]
        nwave_list = [wave_list[int(it)] for it in selected_indices]
        return selected_channels, nwave_list


    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_zs(self, x, wave_list):
        waves = torch.tensor(wave_list, device=x.device).float()
        x,_ = self.patch_embed(x, waves)
        x = x + self.pos_embed[:, 1:, :]
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        for blk in self.blocks[:-1]:
            x = blk(x)
        #mx = x
        x = self.teacher_avgpool(x.transpose(1, 2))
        x = torch.flatten(x, 1)
        x = self.projector(x)

        return x

    def forward_encoder(self, x, mask_ratio, wave_list):
        # embed patches
        waves = torch.tensor(wave_list, device=x.device).float()

        x,waves = self.patch_embed(x, waves)

        self.waves = waves
        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]
        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.random_masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore, in_chan):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x, self.waves)

        # remove cls token
        x = x[:, 1:, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_zt(self, imgs, wave_list):
        with torch.no_grad():
            if imgs.size(1)==2:
                timgs = torch.cat([imgs, torch.zeros([imgs.shape[0],1,imgs.shape[2],imgs.shape[3]], device=imgs.device)],dim=1)
                twave_list = [wave_list[0]] * 3
            elif imgs.size(1)==4:
                timgs = imgs[:,:3,...]
                twave_list = wave_list[:3]
            elif imgs.size(1)==9:
                timgs = imgs[:,:3,...]
                twave_list = wave_list[:3]
            elif imgs.size(1)==202:
                timgs = torch.cat([imgs[:,47:48,...],imgs[:,29:30,...],imgs[:,15:16,...]],dim=1)
                twave_list = [wave_list[47], wave_list[29], wave_list[15]]
            else:
                timgs = imgs
                twave_list = wave_list
            assert timgs.size(1) == 3 and len(twave_list) == 3
            z_t = self.teacher._intermediate_layers(timgs,2)[0]
            z_t = self.teacher_avgpool(z_t.transpose(1, 2))
            z_t = torch.flatten(z_t, 1)
            return z_t, timgs, twave_list

    def forward(self, imgs, in_chans, wave_list, mask_ratio=0.75):
        z_t, timgs, twave_list = self.forward_zt(imgs, wave_list)
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, wave_list)
        z_s = self.forward_zs(timgs, twave_list)
        distill_loss = -(self.cos(z_s, z_t.detach()).mean()) * self.teacher_alpha
        pred = self.forward_decoder(latent, ids_restore, imgs.shape[1])  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask) + distill_loss
        torch.cuda.empty_cache()
        return loss, distill_loss, pred, mask

def mae_vit_small_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedAutoencoderViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
mae_vit_small_patch16 = mae_vit_small_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks
