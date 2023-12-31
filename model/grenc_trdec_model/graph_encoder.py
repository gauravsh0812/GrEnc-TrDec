import torch 
import torch.nn as nn
from torch_geometric.nn import GCNConv, BatchNorm
from model.grenc_trdec_model.position_encoding import Positional_features

class Graph_Encoder(nn.Module):

    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 vit_embed_dim,
                 n_patches,
                 n_pixels,
                 dropout=0.1,
                 ):

        super(Graph_Encoder, self).__init__()
    
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels*2)
        self.conv3 = GCNConv(hidden_channels*2, hidden_channels*4)
        self.conv4 = GCNConv(hidden_channels*4, hidden_channels*8)
        
        self.bn1 = BatchNorm(hidden_channels*2)
        self.bn2 = BatchNorm(hidden_channels*8)
        
        self.pixel2patch = nn.Linear(n_pixels, n_patches)
        self.linear = nn.Linear(vit_embed_dim+hidden_channels*8, vit_embed_dim)
        
        self.relu = nn.ReLU()
        self.p = nn.Dropout(p=dropout)
        
        self.init_weights()

    
    def init_weights(self):
        """
        initializing the model wghts with values
        drawn from normal distribution.
        else initialize them with 0.
        """
        for layer in [self.conv1,
                      self.conv2,
                      self.conv3,
                      self.conv4,
                      self.bn1,
                      self.bn2]:
            
            for name, param in layer.named_parameters():
                if "GCNConv" in name:
                    if "weight" in name:
                        nn.init.normal_(param.data, mean=0, std=0.1)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)
                elif "BatchNorm" in name:
                    if "weight" in name:
                        nn.init.constant_(param.data, 1)
                    elif "bias" in name:
                        nn.init.constant_(param.data, 0)

    def forward(self, 
                x,
                edge_index,
                # features_list, 
                # edges_list,
                vit_output,
                ):
        
        # final_nodes_list = list()
        
        # for i, data in enumerate(graphs_list):
        # assert len(features_list) == len(edges_list)

        # for i, (x, edge_index) in enumerate(zip(features_list, edges_list)):
            # edge_index = edge_index.long()
            
            # has_nan = torch.isnan(edge_index).any()
            # has_inf = torch.isinf(edge_index).any()
            # isgt = edge_index.max() > x.size(0)

            # print('\n Contains NaN:', has_nan.item())
            # print('\n Contains inf:', has_inf.item())
            # print('\n Contains isgt:', isgt, edge_index.max())


        # node embedding
        x = self.relu(self.conv1(self.p(x), edge_index))  # in_chn --> hid
        x = self.relu(self.bn1(self.conv2(self.p(x), edge_index)))  # hid --> hid*2
        x = self.relu(self.conv3(self.p(x), edge_index))  # hid*2 --> hid*4
        x = self.relu(self.bn2(self.conv4(self.p(x), edge_index)))  # hid*4 --> hid*8

        batch_size = vit_output.shape[0]   # [n_samples, n_patches, emb_dim]
        x = x.reshape(batch_size, -1, x.shape[-1])  # [n_samples, n_pixels, hid_dim*8]

        # graph embedding + vit output concat
        # vit_output: (n_samples, n_patches, emb_dim.    
        # _vit = vit_output[i,:,:] # (n_patches,emb_dim)
        _vit_1 = vit_output.shape[1]  # n_patches
        _x_1 = x.shape[1]  # n_pixels

        x = self.pixel2patch(x.permute(0,2,1)).permute(0,2,1) # (n_samples, n_patch, emb)
        x = torch.cat((vit_output, x), dim=2)   # (n_samples, n_patch, emb_dim + hid*8)
        x = self.linear(x)  # (n_samples, n_patches, vit_embed_dim)

        return x
    
        # final_nodes_list.append(x)

        # return torch.stack(final_nodes_list)