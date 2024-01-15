import torch
import torch.nn as nn

class DecodingModel(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 Vit_ENC=None,
                 Tr_DEC=None,
                 isVitPixel=True,
                 ):
        """
        :param encoder: encoders CNN and XFMER
        :param decoder: decoder
        :param device: device to use for model: cpu or gpu
        """
        super(DecodingModel, self).__init__()

        self.vit_enc = Vit_ENC
        self.xfmer_dec = Tr_DEC
        self.vocab = vocab
        self.device = device

        # for pixel information
        self.isVitPixel = isVitPixel

    def forward(
        self,
        enc_output=None,
        mml=None,
        is_test=False,
    ):  
        
        # running the Vit for Patch information
        # vit_patch_output = self.vit_enc(imgs)  # (n_samples, n_patches, embed_dim)

        # if self.isVitPixel:
        #     vit_patch_output = self.vit_enc(imgs, 
        #                               vit_patch_output, 
        #                               isVitPixel=True)  # (n_samples, n_pixels, embed_dim)
            
        # normal training and testing part
        # we will be using torchtext.vocab object
        # while inference, we will provide them
        SOS_token = self.vocab.stoi["<sos>"]
        PAD_token = self.vocab.stoi["<pad>"]

        xfmer_dec_outputs, preds = self.xfmer_dec(
            mml, enc_output, SOS_token, PAD_token, is_test=is_test,
        )
        
        return xfmer_dec_outputs, preds