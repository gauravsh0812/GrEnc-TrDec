import torch
from torch_geometric.utils import from_networkx

G = torch.load("data/image_graphs/1.pt")
print(G)

# data = from_networkx(G)
# print(data)

#f = data.x
#e = data.edge_index

#print(f.shape, e.shape)