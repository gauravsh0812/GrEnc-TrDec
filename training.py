from sklearn.feature_extraction import image
from PIL import Image
import cv2
import numpy as np
import torch 
import sys

np.set_printoptions(threshold=np.inf)

im =  Image.open("data/images/13.png")
# im = np.array(cv2.imread("data/images/13.png", cv2.COLOR_GRAY2RGB))
im = np.array(im)
im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB) * 255

adj = image.img_to_graph(im)

# The graph is a scipy.sparse.coo_matrix object
# The row and column indices correspond to the edge indices
edge_index = np.vstack((adj.row, adj.col))

# The data in the graph corresponds to the edge weights
edge_weights = adj.data

print("Edge Index Matrix:")
print(torch.tensor(edge_index).shape)

# The feature matrix can be the flattened image array (pixel intensities)

feature_matrix = im.flatten().reshape((-1,3))

print("Feature Matrix:")
print(torch.tensor(feature_matrix).shape)
