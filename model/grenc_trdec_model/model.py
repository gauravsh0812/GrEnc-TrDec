import torch
import torch.nn as nn


class Grenc_Trdec_Model(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 Gr_ENC=None,
                 Vit_ENC=None,
                 Tr_DEC=None,
                 isGraph=False,
                 isVitPixel=True,
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

        # for pixel information
        self.isGraphPixel = isGraph
        self.isVitPixel = isVitPixel

    def forward(
        self,
        imgs=None,
        batch = None,
        mml=None,
        is_test=False,
    ):  

        # running the Vit for Patch information
        vit_patch_output = self.vit_enc(imgs)  # (n_samples, n_patches, embed_dim)
        print("vit_patch_output: ", vit_patch_output.shape)

        if self.isVitPixel:
            enc_output = self.vit_enc(imgs, 
                                      vit_patch_output, 
                                      isVitPixel=True)  # (n_samples, n_pixels, embed_dim)
        
        # for pixel information
        elif self.isGraphPixel:
            # running the graph encoder 
            features_list = batch.x.float()
            edge_list = batch.edge_index.long()
            enc_output = self.gr_enc(features_list, edge_list, vit_patch_output)  # (n_samples, n_patches, emb_dim)
        
        # normal training and testing part
        # we will be using torchtext.vocab object
        # while inference, we will provide them
        SOS_token = self.vocab.stoi["<sos>"]
        PAD_token = self.vocab.stoi["<pad>"]

        xfmer_dec_outputs, preds = self.xfmer_dec(
            mml, enc_output, SOS_token, PAD_token, is_test=is_test,
        )

        return xfmer_dec_outputs, preds