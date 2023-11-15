import numpy as np
import cv2
import os
import yaml
import torch
import multiprocessing as mp
import networkx as nx
from sklearn.feature_extraction import image
from torch_geometric.data import Data
from torch_geometric.utils.convert import to_networkx
import matplotlib.pyplot as plt


with open('configs/config.yaml') as f:
	cfg = yaml.safe_load(f)

def main(im):

    im = np.array(cv2.imread(im, cv2.COLOR_GRAY2RGB))

    adj = image.img_to_graph(im)
    edge_index = torch.tensor(np.vstack((adj.row, adj.col)))
    edge_weight = adj.data

    im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) * 255
    feature_matrix = torch.tensor(im.flatten().reshape((-1,3)))
    
    # Make a data object to store graph informaiton
    data = Data(x=feature_matrix, edge_index=edge_index, edge_attr=edge_weight)

    # build graph
    G = to_networkx(data)
    # nx.draw_networkx(G)
    # plt.savefig('graph.png', dpi=300, bbox_inches='tight')
    # plt.show()

if __name__ == "__main__":
    args = cfg["building_graph"]
    imgs = [os.path.join(args["path_to_images"], i) 
            for i in os.listdir(args["path_to_images"]) if ".png" in i]
    _adj_path = os.path.join(
            os.path.dirname(args["path_to_images"]),
            "adj_matrices")
    if not os.path.exists(_adj_path):
        os.mkdir(_adj_path)

    with mp.Pool(args["ncpus"]) as pool:
        result = pool.map(main, imgs)