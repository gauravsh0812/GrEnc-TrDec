import math
import torch.nn as nn
from skema.img2mml.utils.utils import generate_square_subsequent_mask
from skema.img2mml.models.encoding.positional_encoding_for_xfmer import (
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
        self.pos = PositionalEncoding(hid_dim, dropout, max_len)

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

    def forward(self, text):
        # text: (B, L, emb_dim)
        # change the L=H*W to max_len
        text = text.permute(0, 2, 1)  # (B, emb_dim, L)
        text = self.change_length(
            text
        )  # (B, emb_dim, max_len)
        text = text.permute(
            2, 0, 1
        )  # (max_len, B, emb_dim)

        # embedding + normalization
        """
        no need to embed as src from cnn already has hid_dim as the 3rd dim
        """
        text *= math.sqrt(
            self.hid_dim
        )  # (max_len, B, emb_dim)

        # adding positoinal encoding
        pos_src = self.pos(text)  # (max_len, B, emb_dim)

        # xfmer encoder
        generate_square_subsequent_mask(pos_src.shape[0]).to(
            self.device
        )
        xfmer_enc_output = self.xfmer_encoder(
            src=pos_src, mask=None
        )  # (max_len, B, emb_dim)

        return xfmer_enc_output