import torch
import torch.nn as nn
from utils.clip_projection import ProjectionHead

class ClipModel(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 vit_emb_dim,
                 xfmer_emb_dim,
                 projection_dim,
                 dropout,
                 Vit_ENC=None,
                 isVitPixel=True,
                 Xfmer_ENC=None,
                 ):
        """
        :param encoder: encoders CNN and XFMER
        :param decoder: decoder
        :param device: device to use for model: cpu or gpu
        """
        super(ClipModel, self).__init__()

        self.vit_enc = Vit_ENC
        self.Xfmer_ENC = Xfmer_ENC
        self.vocab = vocab
        self.device = device
        self.output_dim = len(vocab)

        # for pixel information
        self.isVitPixel = isVitPixel

        # self.embed_img = nn.Embeddding(self.output_dim, vit_emb_dim)
        self.embed_text = nn.Embedding(self.output_dim, xfmer_emb_dim)

        self.projection = ProjectionHead(
            vit_emb_dim,
            xfmer_emb_dim,
            projection_dim,
            dropout,
        )

    def forward(
        self,
        imgs=None,
        batch = None,
        mml=None,
        is_test=False,
    ):  
        # ENCODING IMAGES
        vit_enc_output = self.vit_enc(imgs)  # (n_samples, n_patches, embed_dim)

        if self.isVitPixel:
            vit_enc_output = self.vit_enc(imgs, 
                                      vit_enc_output, 
                                      isVitPixel=True)  # (n_samples, n_pixels, embed_dim)
       
        # ENCODING TEXTS
        embedded_mml = self.embed_text(mml)   # (B, max_len, emb_dim)
        xfmer_enc_output = self.Xfmer_ENC(embedded_mml)  # (max_len, B, emb_dim)
        xfmer_enc_output = xfmer_enc_output.permute(1,0,2)  # (B, max_len, emb_dim)

        # CLIP 
        projected_img = self.projection(vit_enc_output)
        projected_mml = self.projection(xfmer_enc_output)
        