import torch
import math
import torch.nn as nn
from model.grenc_trdec_model.position_encoding import PositionalEncoding


class Transformer_Decoder(nn.Module):
    def __init__(
        self,
        tr_enc_hid_dim,
        dec_emb_dim,
        dec_hid_dim,
        nheads,
        output_dim,
        # n_patches,       # n_patches = img_w//patche_size * img_h//patch_size
        dropout,
        max_len,
        n_xfmer_decoder_layers,
        dim_feedfwd,
        device,
    ):
        super(Transformer_Decoder, self).__init__()
        self.device = device
        self.output_dim = output_dim
        self.emb_dim = dec_emb_dim
        self.embed = nn.Embedding(output_dim, dec_emb_dim)
        self.pos = PositionalEncoding(dec_emb_dim, dropout, max_len)
        self.change_dim = nn.Linear(tr_enc_hid_dim, dec_hid_dim)

        """
        NOTE:
        updated nn.TransformerDecoderLayer doesn't have 'batch_first' argument anymore.
        Therefore, the sequences will be in the shape of (max_len, B)
        """
        xfmer_dec_layer = nn.TransformerDecoderLayer(
            d_model=dec_hid_dim,
            nhead=nheads,
            dim_feedforward=dim_feedfwd,
            dropout=dropout,
        )

        self.xfmer_decoder = nn.TransformerDecoder(
            xfmer_dec_layer, num_layers=n_xfmer_decoder_layers
        )

        self.modify_dimension = nn.Linear(dec_emb_dim, dec_hid_dim)
        self.final_linear = nn.Linear(dec_hid_dim, output_dim)
        self.init_weights()

    def init_weights(self):
        self.modify_dimension.bias.data.zero_()
        self.modify_dimension.weight.data.uniform_(-0.1, 0.1)
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.final_linear.bias.data.zero_()
        self.final_linear.weight.data.uniform_(-0.1, 0.1)

    def create_pad_mask(
        self, matrix: torch.tensor, pad_token: int
    ) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return matrix == pad_token

    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask


    def forward(
        self,
        trg,
        enc_output,
        sos_idx,
        pad_idx,
        is_test=False,
    ):
        # enc_output: (max_len, B, xfmer-enc hid_dim)
        # trg: (B, max_len)
        """
        we provide input: [<sos>, x1, x2, ...]
        we get output: [x1, x2, ..., <eos>]
        So we have to add <sos> in  the final preds

        for inference
        trg: sequnece containing total number of token that has been predicted.
        xfmer_enc_output: input from encoder
        """
        
        (B, max_len) = trg.shape
        print("device: ", self.device)
        _preds = torch.zeros(max_len, B)#.to(self.device)  # (max_len, B)
        trg = trg.permute(1, 0)  # (max_len, B)
        trg = trg[:-1, :]  # (max_len-1, B)

        sequence_length = trg.shape[0]
       
        trg_attn_mask = self.generate_square_subsequent_mask(
                                    sequence_length).to(self.device)  # (max_len-1, max_len-1)

        trg_padding_mask = self.create_pad_mask(
                                    trg,pad_idx).permute(1,0)  # (B, max_len-1)

        trg = self.embed(trg) * math.sqrt(
            self.emb_dim
        )  # (max_len-1, B, dec_emb_dim)

        pos_trg = self.pos(trg)  # (max_len-1, B, emb_dim)
        pos_trg = self.modify_dimension(pos_trg)  # (max_len-1, B, dec_hid_dim)

        # changing n_patches to max_len
        enc_output = self.change_dim(enc_output) # (max_len, B, dec_hid_dim)

        # outputs: (max_len-1,B, dec_hid_dim)
        xfmer_dec_outputs = self.xfmer_decoder(
            tgt=pos_trg,
            memory=enc_output,
            tgt_mask=trg_attn_mask,
            tgt_key_padding_mask=trg_padding_mask,
        )

        xfmer_dec_outputs = self.final_linear(
            xfmer_dec_outputs
        )  # (max_len-1,B, output_dim)
        
        # preds
        _preds[0, :] = torch.full(_preds[0, :].shape, sos_idx)
        if is_test:
            for i in range(xfmer_dec_outputs.shape[0]):
                top1 = xfmer_dec_outputs[i, :, :].argmax(1)  # (B)
                _preds[i + 1, :] = top1

        # xfmer_dec_outputs: (max_len-1, B, output_dim); _preds: (max_len, B)
        # permute them to make "Batch first"
        return xfmer_dec_outputs.permute(1, 0, 2), _preds.permute(1, 0)