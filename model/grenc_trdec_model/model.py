import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clip_projection import ProjectionHead

class ClipModel(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 n_patches, 
                 decoder_emb_dim,   # trying
                 vit_emb_dim,
                 xfmer_emb_dim,
                 xfmer_hid_dim,
                 projection_dim,
                 max_len,
                 dropout,
                 temperature,
                 Vit_ENC=None,
                 Xfmer_ENC=None,
                 xfmer_DEC=None,
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
        self.Xfmer_DEC = xfmer_DEC
        self.vocab = vocab
        self.device = device
        self.output_dim = len(vocab)
        self.decoder_emb_dim = decoder_emb_dim
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(
                ignore_index=self.vocab.stoi["<pad>"]
                )

        # for pixel information
        self.isVitPixel = isVitPixel
        self.lin = nn.Linear(vit_emb_dim, max_len)
        self.lin2 = nn.Linear(n_patches, xfmer_emb_dim)
        self.embed_text = nn.Embedding(self.output_dim, xfmer_emb_dim)

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
        train_dec=False,
    ):  
        # ENCODING IMAGES
        vit_enc_output = self.vit_enc(imgs)  # (B, n_patches, embed_dim)

        if self.isVitPixel:
            vit_enc_output = self.vit_enc(imgs, 
                                      vit_enc_output, 
                                      isVitPixel=True)  # (B, n_patches, embed_dim)

        if not train_dec:
            # CLIP 
            # ENCODING TEXTS
            embedded_mml = self.embed_text(mml)   # (B, max_len, emb_dim)
            xfmer_enc_output = self.Xfmer_ENC(embedded_mml)  # (max_len, B, emb_dim)
            xfmer_enc_output = xfmer_enc_output.permute(1,0,2)  # (B, max_len, emb_dim)

            # reshaping the tensors fom 3D to 2D - (B,-1). 
            batch_size = vit_enc_output.shape[0]
            vit_enc_output = vit_enc_output.reshape(batch_size, -1)
            xfmer_enc_output = xfmer_enc_output.reshape(batch_size, -1)

            # projection head - both will be (B, proj_dim)
            projected_img = self.projection(vit_enc_output, img=True)
            projected_mml = self.projection(xfmer_enc_output, img=False)
            
            # https://github.com/moein-shariatnia/OpenAI-CLIP/blob/master/config.py
            # Calculating the Loss
            logits = (projected_mml @ projected_img.T) / self.temperature
            images_similarity = projected_img @ projected_img.T
            texts_similarity = projected_mml @ projected_mml.T
            targets = F.softmax(
                (images_similarity + texts_similarity) / 2 * self.temperature, dim=-1
            )

            # training or validation
            texts_loss = self.crossEntropyLoss(logits, targets)
            images_loss = self.crossEntropyLoss(logits.T, targets.T)
            loss =  (images_loss + texts_loss) / 2.0 # shape: (batch_size)
            return loss.mean()
        
        else:
            # Train Dec
            # vit_enc_output: (B, n_pathes/pixels, emb_dim)
            vit_enc_output = self.lin(vit_enc_output).permute(0,2,1) # (B, max_len, n)
            vit_enc_output = self.lin2(vit_enc_output)   # (B, max_len, xfmer_enc_emb_dim)
            xfmer_enc_output = self.Xfmer_ENC(vit_enc_output)  # (max_len, B, hid_dim)

            xfmer_dec_output = self.Xfmer_DEC(mml,
                                              xfmer_enc_output,
                                              self.vocab.stoi["<sos>"],
                                              self.vocab.stoi["<pad>"])
            
            print("decoder shape==========: ", xfmer_dec_output.shape)

            loss = self.criterion(xfmer_dec_output, mml)
            
            return loss


    def crossEntropyLoss(self, preds, targets):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        return loss