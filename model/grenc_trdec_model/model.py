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
        self.vit_enc = Vit_ENC
        self.xfmer_dec = Tr_DEC
        self.vocab = vocab
        self.device = device

    def forward(
        self,
        imgs=None,
        batch = None,
        # features_list=None,
        # edge_list=None,
        mml=None,
        is_test=False,
    ):  

        # running the Vit
        vit_output = self.vit_enc(imgs)  # (n_samples, n_patches, embed_dim)
        
        # running the graph encoder 
        features_list = batch.x.float()
        edge_list = batch.edge_index.long()
        gr_output = self.gr_enc(features_list, edge_list, vit_output)  # (n_samples, n_patches, gr_hidden*8)
        
        # normal training and testing part
        # we will be using torchtext.vocab object
        # while inference, we will provide them
        SOS_token = self.vocab.stoi["<sos>"]
        PAD_token = self.vocab.stoi["<pad>"]

        xfmer_dec_outputs, preds = self.xfmer_dec(
            mml, gr_output, SOS_token, PAD_token, is_test=is_test,
        )

        return xfmer_dec_outputs, preds