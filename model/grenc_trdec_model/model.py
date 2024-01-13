import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clip_projection import ProjectionHead

class ClipModel(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 vit_emb_dim,
                 xfmer_emb_dim,
                 xfmer_hid_dim,
                 projection_dim,
                 dropout,
                 temperature,
                 Vit_ENC=None,
                 Xfmer_ENC=None,
                 isVitPixel=True,                 
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
        self.temperature = temperature

        # for pixel information
        self.isVitPixel = isVitPixel

        # self.embed_img = nn.Embeddding(self.output_dim, vit_emb_dim)
        self.embed_text = nn.Embedding(self.output_dim, xfmer_emb_dim)

        self.change_length = nn.Linear(75,350)

        self.projection = ProjectionHead(
            vit_emb_dim,
            xfmer_emb_dim,
            xfmer_hid_dim,
            projection_dim,
            dropout,
        )

    def forward(
        self,
        imgs=None,
        mml=None,
        only_img=False,
    ):  
        # ENCODING IMAGES
        vit_enc_output = self.vit_enc(imgs)  # (B, n_patches, embed_dim)

        if self.isVitPixel:
            vit_enc_output = self.vit_enc(imgs, 
                                      vit_enc_output, 
                                      isVitPixel=True)  # (B, n_pixels, embed_dim)
        if only_img:
            return vit_enc_output
            
        # ENCODING TEXTS
        else:
            embedded_mml = self.embed_text(mml)   # (B, max_len, emb_dim)
            xfmer_enc_output = self.Xfmer_ENC(embedded_mml)  # (max_len, B, emb_dim)
            xfmer_enc_output = xfmer_enc_output.permute(1,0,2)  # (B, max_len, emb_dim)

            # CLIP 
            vit_enc_output = self.change_length(
                vit_enc_output.permute(0,2,1)).permute(0,2,1) # (B, max_len, emb_dim)
            projected_img = self.projection(vit_enc_output, img=True)
            projected_mml = self.projection(xfmer_enc_output, img=False)
        
            
            # https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/config.py
            # Calculating the Loss
            # print("pmml and pmmlT: ", projected_mml.shape, projected_mml.T.shape)
            # print("pimg and pimgT: ", projected_img.shape, projected_img.T.shape)

            logits = (projected_mml @ projected_img.permute(0,2,1)) / self.temperature
            images_similarity = projected_img @ projected_img.permute(0,2,1)
            texts_similarity = projected_mml @ projected_mml.permute(0,2,1)
            targets = F.softmax(
                (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
            )
            
            # training or validation
            print("logits tgts: ", logits.shape, targets.view(-1, targets.shape[-1]).shape)
            texts_loss = nn.CrossEntropyLoss(logits, targets.view(-1, targets.shape[-1]))
            images_loss = nn.CrossEntropyLoss(logits.T, targets.T.view(-1, targets.shape[-1]))
            loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            return loss.mean()