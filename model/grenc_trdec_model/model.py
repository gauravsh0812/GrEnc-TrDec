import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.clip_projection import ProjectionHead

class ClipModel(nn.Module):
    def __init__(self,  
                 vocab, 
                 device,
                 decoder_emb_dim,   # trying
                 cnn_hid_dim,
                 xfmer_emb_dim,
                 xfmer_hid_dim,
                 projection_dim,
                 max_len,
                 dropout,
                 temperature,
                 Cnn_ENC=None,
                 Xfmer_ENC=None,
                 xfmer_DEC=None,
                 ):
        """
        :param encoder: encoders CNN and XFMER
        :param decoder: decoder
        :param device: device to use for model: cpu or gpu
        """
        super(ClipModel, self).__init__()

        self.cnn_enc = Cnn_ENC
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
        L = 930
        self.lin = nn.Linear(cnn_hid_dim, max_len)
        self.lin2 = nn.Linear(L, xfmer_emb_dim)
        self.embed_text = nn.Embedding(self.output_dim, xfmer_emb_dim)

        self.projection = ProjectionHead(
            projection_dim,
            dropout,
        )

    def forward(
        self,
        imgs=None,
        mml=None,
        is_test=False,
        train_dec=False,

    ):  
        # ENCODING IMAGES
        cnn_enc_output = self.cnn_enc(imgs)  # (B, L, enc_hid_dim)

        if not train_dec:
            # CLIP 
            # ENCODING TEXTS
            embedded_mml = self.embed_text(mml)   # (B, max_len, emb_dim)
            xfmer_enc_output = self.Xfmer_ENC(embedded_mml)  # (max_len, B, emb_dim)
            xfmer_enc_output = xfmer_enc_output.permute(1,0,2)  # (B, max_len, emb_dim)

            # reshaping the tensors fom 3D to 2D - (B,-1). 
            batch_size = cnn_enc_output.shape[0]
            cnn_enc_output = cnn_enc_output.reshape(batch_size, -1)
            xfmer_enc_output = xfmer_enc_output.reshape(batch_size, -1)

            # projection head - both will be (B, proj_dim)
            projected_img = self.projection(cnn_enc_output, img=True)
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
            # cnn_enc_output: (B, n_pathes/pixels, emb_dim)
            cnn_enc_output = self.lin(cnn_enc_output).permute(0,2,1) # (B, max_len, L)
            cnn_enc_output = self.lin2(cnn_enc_output)   # (B, max_len, xfmer_enc_emb_dim)
            xfmer_enc_output = self.Xfmer_ENC(cnn_enc_output)  # (max_len, B, hid_dim)

            xfmer_dec_outputs, preds = self.Xfmer_DEC(mml,
                                              xfmer_enc_output,
                                              self.vocab.stoi["<sos>"],
                                              self.vocab.stoi["<pad>"],
                                              is_test=is_test)   # (B, max_len-1, output_dim)
            # calculate loss for training only
            if not is_test:
                output_dim = xfmer_dec_outputs.shape[-1]
                xfmer_dec_outputs = xfmer_dec_outputs.contiguous().view(-1, output_dim)
                mml = mml[:, 1:].contiguous().view(-1)
                loss = self.criterion(xfmer_dec_outputs, mml)
                return xfmer_dec_outputs, preds, loss
            else:
                return xfmer_dec_outputs, preds, []

    def crossEntropyLoss(self, preds, targets):
        log_softmax = nn.LogSoftmax(dim=-1)
        loss = (-targets * log_softmax(preds)).sum(1)
        return loss