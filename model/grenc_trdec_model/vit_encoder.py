"""
inspired from https://github.com/jankrepl/mildlyoverfitted/blob/master/github_adventures/vision_transformer/custom.py
"""

import torch 
import torch.nn as nn
from model.grenc_trdec_model.position_encoding import add_positional_features

class PatchEmbed(nn.Module):
    """
    This class will do the patching
    and will embed the final results.
    """
    def __init__(self,
                 img_size,
                 patch_size=10,
                 in_channels=3,
                 emb_dim=256,
                 ):
        
        # since the image is rectangle with 
        # w,h = 500,100, we need to chose the 
        # patch_size accordingly.

        super(PatchEmbed, self).__init__()

        error = "choose patch size such that \
                image width and height should be \
                multiple of the patch_size."
        
        assert img_size[0] % patch_size == 0, error
        assert img_size[1] % patch_size == 0, error 

        self.patch_size = patch_size 
        self.in_chn = in_channels
        self.emb_dim = emb_dim
        
        # patch and embed
        self.patch_embed = nn.Conv2d(
            in_channels,
            emb_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        
    def forward(self,
                img,
                ):
        
        # (N, in_chn, H, W) --> (N, emb_dim, H/patch_size, W/patch_size)
        img = self.patch_embed(img) 

        # n_patches = H/patch_size * W/patch_size
        # (N, n_patches, emb_dim)
        img = img.view(img.shape[0], img.shape[1], -1).permute(0,2,1)

        return img

class Attention(nn.Module):
    def __init__(self, 
                 dim, 
                 n_heads=12, 
                 qkv_bias=True, 
                 attn_p=0., 
                 proj_p=0.
                ):
            
            super().__init__()
            self.n_heads = n_heads
            self.dim = dim
            self.head_dim = dim // n_heads
            self.scale = self.head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_p)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_p)

    def forward(self, img):

        n_samples, n_tokens, dim = img.shape  # n_tokens = n_patches + 1

        assert dim == self.dim 

        qkv = self.qkv(img)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        img = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        img = self.proj_drop(img)  # (n_samples, n_patches + 1, dim)

        return img
    
class MLP(nn.Module):
    
    def __init__(self, 
                 in_features, 
                 hidden_features, 
                 out_features, 
                 p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, img):
        
        # img: (n_smaples, n_patches+1, in_chns)
        img = self.fc1(
                img
        ) # (n_samples, n_patches + 1, hidden_features)
        img = self.act(img)  # (n_samples, n_patches + 1, hidden_features)
        img = self.drop(img)  # (n_samples, n_patches + 1, hidden_features)
        img = self.fc2(img)  # (n_samples, n_patches + 1, out_features)
        img = self.drop(img)  # (n_samples, n_patches + 1, out_features)

        return img

class Block(nn.Module):
    def __init__(self, 
                 dim, 
                 n_heads, 
                 mlp_ratio=4.0, 
                 qkv_bias=True, 
                 p=0., 
                 attn_p=0.):
        
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x

class VisionTransformer(nn.Module):

    def __init__(
            self,
            img_size,
            patch_size,
            in_chns,
            embed_dim,
            depth,
            n_heads,
            mlp_ratio,
            qkv_bias,
            p=0.,
            attn_p=0.,
    ):
        super().__init__()

        self.patch_embed = PatchEmbed(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chns,
                embed_dim=embed_dim,
        )
        
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, x):
        # x: (N, in_chns, H, W)
        x = self.patch_embed(x)    # (N, emb_dim, n_patches)
        x = x.permute(0,2,1)   # (N, n_patches, emb_dim)
        x = x + add_positional_features(x) # (n_samples, n_patches, embed_dim)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        return x