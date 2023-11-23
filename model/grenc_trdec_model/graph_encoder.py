import torch 
import torch.nn as nn
from torch_geometric.nn import GCNConv, BatchNorm

class Graph_Encoder(nn.Module):
    def __init__(self, 
                 in_channels, 
                 hidden_channels, 
                 out_channels,
                 dropout=0.1,
                 ):

        super(Graph_Encoder).__init__()
        
        self.conv1 = GCNConv(in_channels, hidden_channels/8)
        self.conv2 = GCNConv(hidden_channels/8, hidden_channels/4)
        self.conv3 = GCNConv(hidden_channels/4, hidden_channels/2)
        self.conv4 = GCNConv(hidden_channels/2, hidden_channels)
        self.conv5 = GCNConv(hidden_channels, out_channels)
        
        self.bn1 = BatchNorm(hidden_channels/4)
        self.bn2 = BatchNorm(hidden_channels/2)
        
        self.relu = nn.ReLU()
        self.p = nn.Dropout(p=dropout)

        self.init_weights()

    
    def init_weights(self):
        """
        initializing the model wghts with values
        drawn from normal distribution.
        else initialize them with 0.
        """
        for name, param in self.gcn.named_parameters():
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
                features, 
                edge_index, 
                ):
        
        x = self.relu(self.conv1(self.p(features), edge_index))  # in_chn --> hid/8
        x = self.relu(self.bn1(self.conv2(self.p(x), edge_index)))  # hid/8 --> hid/4
        x = self.relu(self.conv3(self.p(x), edge_index))  # hid/4 --> hid/2
        x = self.relu(self.bn2(self.conv4(self.p(x), edge_index)))  # hid/2 --> hid 
        x = self.relu(self.conv5(self.p(x), edge_index))  # hid --> output_chn
        
        return x