import torch
import torch.nn as nn


class Grenc_Trdec_Model(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 Gr_ENC=None,
                 Vit_ENC=None,
                 Tr_DEC=None,
                 ):
        """
        :param encoder: encoders CNN and XFMER
        :param decoder: decoder
        :param device: device to use for model: cpu or gpu
        """
        super(Grenc_Trdec_Model, self).__init__()

        self.gr_enc = Gr_ENC
        self.xfmer_encoder = Vit_ENC
        self.xfmer_decoder = Tr_DEC
        self.vocab = vocab
        self.device = device

    def forward(
        self,
        imgs=None,
        graphs=None,
        mml=None,
        is_test=False,
        SOS_token=None,
        EOS_token=None,
        PAD_token=None,
    ):  

        # running the graph encoder 
        gr_output = self.gr_enc(graphs)

        # run the encoder --> get flattened FV of images
        # for inference Batch(B)=1
        cnn_enc_output = self.cnn_encoder(src)  # (B, L, dec_hid_dim)
        xfmer_enc_output = self.xfmer_encoder(
            cnn_enc_output
        )  # (max_len, B, dec_hid_dim)

        # xfmer_enc_output = self.linear(cnn_enc_output.permute(0,2,1)).permute(2,0,1)

        # normal training and testing part
        # we will be using torchtext.vocab object
        # while inference, we will provide them
        SOS_token = self.vocab.stoi["<sos>"]
        EOS_token = self.vocab.stoi["<eos>"]
        PAD_token = self.vocab.stoi["<pad>"]

        xfmer_dec_outputs, preds = self.xfmer_decoder(
            trg, xfmer_enc_output, SOS_token, PAD_token, is_test=is_test,
        )

        return xfmer_dec_outputs, preds