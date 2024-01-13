# https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/config.py

import torch
from torch import nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        vit_emb_dim,
        xfmer_emb_dim,
        projection_dim,
        dropout,
    ):
        super().__init__()
        self.img_projection = nn.Linear(vit_emb_dim, projection_dim)
        self.text_projection = nn.Linear(xfmer_emb_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x, img=True):
        print("x shpe: ", x.shape)
        if img:
            projected = self.img_projection(x)
        else:
            projected = self.text_projection(x)
            
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x