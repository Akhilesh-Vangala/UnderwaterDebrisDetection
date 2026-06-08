import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math
import numpy as np


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        random_tensor = keep_prob + torch.rand(x.shape[0], 1, 1, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None,
                 out_features: Optional[int] = None, act_layer: nn.Module = nn.GELU,
                 drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LePE(nn.Module):
    def __init__(self, dim: int, resolution: int):
        super().__init__()
        self.pos_conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.resolution = resolution
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.pos_conv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class CSWinBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, split_size: int = 7, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None, dropout: float = 0.0,
                 attn_drop: float = 0.0, drop_path: float = 0.0, act_layer: nn.Module = nn.GELU,
                 norm_layer: nn.Module = nn.LayerNorm, last_stage: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.attn_drop = nn.Dropout(attn_drop)
        
        if last_stage:
            self.lepe = LePE(dim // num_heads, resolution=split_size)
        else:
            self.lepe1 = LePE(dim // num_heads, resolution=split_size)
            self.lepe2 = LePE(dim // num_heads, resolution=split_size)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim,
                      act_layer=act_layer, drop=dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = torch.chunk(x, self.branch_num, dim=1)
        x1 = x[0].contiguous().view(B, C // self.branch_num, H, W)
        x2 = x[1].contiguous().view(B, C // self.branch_num, H, W) if self.branch_num == 2 else None
        return x1, x2, H, W
    
    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = func(x, H, W)
        x = x.transpose(-2, -1).contiguous().view(B, N, C)
        return x
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
    
    def attn(self, x, H, W):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.branch_num == 2:
            x1, x2, H, W = self.im2cswin(x)
            q1, q2 = q.chunk(2, dim=2)
            k1, k2 = k.chunk(2, dim=2)
            v1, v2 = v.chunk(2, dim=2)
            
            attn1 = (q1 @ k1.transpose(-2, -1)) * (C // self.num_heads // 2) ** -0.5
            attn1 = attn1.softmax(dim=-1)
            attn1 = self.attn_drop(attn1)
            attn1 = attn1 @ v1
            attn1 = self.get_lepe(attn1, self.lepe1)
            
            attn2 = (q2 @ k2.transpose(-2, -1)) * (C // self.num_heads // 2) ** -0.5
            attn2 = attn2.softmax(dim=-1)
            attn2 = self.attn_drop(attn2)
            attn2 = attn2 @ v2
            attn2 = self.get_lepe(attn2, self.lepe2)
            
            attn = torch.cat([attn1, attn2], dim=2)
        else:
            attn = (q @ k.transpose(-2, -1)) * (C // self.num_heads) ** -0.5
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            attn = attn @ v
            attn = self.get_lepe(attn, self.lepe)
        
        attn = attn.transpose(1, 2).reshape(B, N, C)
        attn = self.proj(attn)
        attn = self.proj_drop(attn)
        return attn


class CSWinStage(nn.Module):
    def __init__(self, dim: int, depth: int, num_heads: int, split_size: int, mlp_ratio: float = 4.0,
                 qkv_bias: bool = False, qk_scale: Optional[float] = None, dropout: float = 0.0,
                 attn_drop: float = 0.0, drop_path: List[float] = None, norm_layer: nn.Module = nn.LayerNorm,
                 last_stage: bool = False):
        super().__init__()
        self.blocks = nn.ModuleList([
            CSWinBlock(dim=dim, num_heads=num_heads, split_size=split_size, mlp_ratio=mlp_ratio,
                      qkv_bias=qkv_bias, qk_scale=qk_scale, dropout=dropout, attn_drop=attn_drop,
                      drop_path=drop_path[i] if drop_path is not None else 0.0, norm_layer=norm_layer,
                      last_stage=last_stage and (i == depth - 1))
            for i in range(depth)
        ])
    
    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x, H, W)
        return x


class MergeBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, norm_layer: nn.Module = nn.LayerNorm):
        super().__init__()
        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim * 4, dim_out, bias=False)
    
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = self.norm(x)
        x = x.reshape(B, H, W, C)
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], -1)
        x = x.reshape(B, -1, C * 4)
        x = self.reduction(x)
        return x, H // 2, W // 2


class PatchEmbedding(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 96):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        patches_resolution = img_size // patch_size
        self.num_patches = patches_resolution ** 2
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size, W // self.patch_size
        return x, (H, W)


class CSWinTransformer(nn.Module):
    def __init__(self, img_size: int = 224, patch_size: int = 4, in_chans: int = 3,
                 num_classes: int = 8, embed_dim: int = 96, depths: List[int] = [2, 4, 32, 2],
                 split_sizes: List[int] = [1, 2, 7, 7], num_heads: List[int] = [2, 4, 8, 16],
                 mlp_ratio: float = 4.0, qkv_bias: bool = True, qk_scale: Optional[float] = None,
                 dropout: float = 0.0, attn_drop: float = 0.0, drop_path: float = 0.1,
                 norm_layer: nn.Module = nn.LayerNorm, use_chk: bool = False):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = len(depths)
        self.embed_dim = embed_dim
        self.use_chk = use_chk
        
        self.patch_embed = PatchEmbedding(img_size=img_size, patch_size=patch_size,
                                         in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.num_patches ** 0.5
        self.patches_resolution = patches_resolution
        
        self.pos_drop = nn.Dropout(dropout)
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, sum(depths))]
        cur = 0
        
        self.stages = nn.ModuleList()
        self.merges = nn.ModuleList()
        
        for i in range(self.num_stages):
            stage = CSWinStage(dim=int(embed_dim * 2 ** i), depth=depths[i],
                             num_heads=num_heads[i], split_size=split_sizes[i],
                             mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                             dropout=dropout, attn_drop=attn_drop, drop_path=dpr[cur:cur + depths[i]],
                             norm_layer=norm_layer, last_stage=(i == self.num_stages - 1))
            self.stages.append(stage)
            cur += depths[i]
            
            if i < self.num_stages - 1:
                merge = MergeBlock(dim=int(embed_dim * 2 ** i), dim_out=int(embed_dim * 2 ** (i + 1)),
                                  norm_layer=norm_layer)
                self.merges.append(merge)
        
        self.norm = norm_layer(int(embed_dim * 2 ** (self.num_stages - 1)))
        self.head = nn.Linear(int(embed_dim * 2 ** (self.num_stages - 1)), num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    
    def get_classifier(self):
        return self.head
    
    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
    
    def forward_features(self, x):
        x, (H, W) = self.patch_embed(x)
        x = self.pos_drop(x)
        
        for i, stage in enumerate(self.stages):
            x = stage(x, H, W)
            if i < len(self.merges):
                x, H, W = self.merges[i](x, H, W)
        
        x = self.norm(x)
        return x, (H, W)
    
    def forward(self, x):
        x, (H, W) = self.forward_features(x)
        x = x.mean(dim=1)
        x = self.head(x)
        return x


class CSWinTransformerWithAttentionPooling(nn.Module):
    def __init__(self, base_model: CSWinTransformer):
        super().__init__()
        self.base_model = base_model
        self.attention_pool = nn.MultiheadAttention(
            embed_dim=int(base_model.embed_dim * 2 ** (base_model.num_stages - 1)),
            num_heads=8,
            batch_first=True
        )
        self.norm_pool = nn.LayerNorm(int(base_model.embed_dim * 2 ** (base_model.num_stages - 1)))
    
    def forward(self, x):
        features, (H, W) = self.base_model.forward_features(x)
        features = self.norm_pool(features)
        pooled, _ = self.attention_pool(features, features, features)
        pooled = pooled.mean(dim=1)
        return self.base_model.head(pooled)


def build_cswin_small(num_classes: int = 8, img_size: int = 224) -> CSWinTransformer:
    return CSWinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=64,
        depths=[2, 4, 32, 2],
        split_sizes=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        use_chk=False
    )


def build_cswin_base(num_classes: int = 8, img_size: int = 224) -> CSWinTransformer:
    return CSWinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=96,
        depths=[2, 4, 32, 2],
        split_sizes=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.1,
        attn_drop=0.1,
        drop_path=0.1,
        norm_layer=nn.LayerNorm,
        use_chk=False
    )


def build_cswin_large(num_classes: int = 8, img_size: int = 224) -> CSWinTransformer:
    return CSWinTransformer(
        img_size=img_size,
        patch_size=4,
        in_chans=3,
        num_classes=num_classes,
        embed_dim=144,
        depths=[2, 4, 32, 2],
        split_sizes=[1, 2, 7, 7],
        num_heads=[4, 8, 16, 32],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        dropout=0.1,
        attn_drop=0.1,
        drop_path=0.2,
        norm_layer=nn.LayerNorm,
        use_chk=False
    )


class EnsembleCSWin(nn.Module):
    def __init__(self, models: List[CSWinTransformer], weights: Optional[List[float]] = None):
        super().__init__()
        self.models = nn.ModuleList(models)
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]
    
    def forward(self, x):
        outputs = []
        for model in self.models:
            outputs.append(model(x))
        outputs = torch.stack(outputs, dim=0)
        weighted_output = torch.sum(outputs * torch.tensor(self.weights, device=x.device).view(-1, 1, 1), dim=0)
        return weighted_output
