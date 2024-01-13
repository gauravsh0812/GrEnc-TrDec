import math
import torch
import torch.nn as nn
from model.grenc_trdec_model.position_encoding import (
    PositionalEncoding,
)

class Transformer_Encoder(nn.Module):
    def __init__(
        self,
        emb_dim,
        hid_dim,
        nheads,
        dropout,
        device,
        max_len,
        n_xfmer_encoder_layers,
        dim_feedfwd,
    ):
        super(Transformer_Encoder, self).__init__()
        self.hid_dim = hid_dim
        self.device = device
        self.pos = PositionalEncoding(emb_dim, dropout, max_len)
        self.change_length = nn.Linear(emb_dim, hid_dim)

        """
        NOTE:
        nn.TransformerDecoderLayer doesn't have 'batch_first' argument anymore.
        Therefore, the sequences will be in the shape of (max_len, B)
        """
        xfmer_enc_layer = nn.TransformerEncoderLayer(
            d_model=hid_dim,
            nhead=nheads,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
        )

        self.xfmer_encoder = nn.TransformerEncoder(
            xfmer_enc_layer, num_layers=n_xfmer_encoder_layers
        )

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask
    
    def forward(self, text):
        # text: (B, max_len, emb_dim)
        # change the L=H*W to max_len
        print("text size: ", text.shape)
        text = self.change_length(
            text
        )  # (B, max_len, hid_dim)
        text = text.permute(
            1, 0, 2
        )  # (max_len, B, hid_dim)

        # embedding + normalization
        """
        no need to embed as src from cnn already has hid_dim as the 3rd dim
        """
        text *= math.sqrt(
            self.hid_dim
        )  # (max_len, B, hid_dim)

        # adding positoinal encoding
        pos_src = self.pos(text)  # (max_len, B, hid_dim)

        # xfmer encoder
        self.generate_square_subsequent_mask(pos_src.shape[0]).to(
            self.device
        )
        xfmer_enc_output = self.xfmer_encoder(
            src=pos_src, mask=None
        )  # (max_len, B, hid_dim)

        return xfmer_enc_output
