"""
base script inspired by  https://github.com/harvardnlp/im2markup/blob/master/scripts/utils/image_utils.py
"""

import numpy as np
import cv2
import os
import yaml
import torch
import multiprocessing as mp
import networkx as nx
from sklearn.feature_extraction import image
from torch_geometric.data import Data
from torch_geometric.utils import grid
from torchvision import transforms
from PIL import Image


with open('configs/config.yaml') as f:
	cfg = yaml.safe_load(f)
args = cfg["building_graph"]
path_to_data = cfg["preprocessing"]["path_to_data"]

def crop_image(im, reject=False):
    # converting to np array
    image_arr = np.asarray(im, dtype=np.uint8)

    # find where the data lies
    indices = np.where(image_arr != 255)

    # see if image is not blank
    # if both arrays of indices are null: blank image
    # if either is not null: it is only line either horizontal or vertical
    # In any case, thse image will be treated as garbage and will be discarded.

    if (len(indices[0]) == 0) or (len(indices[1]) == 0):
        reject = True

    else:
        # get the boundaries
        x_min = np.min(indices[1])
        x_max = np.max(indices[1])
        y_min = np.min(indices[0])
        y_max = np.max(indices[0])

        # crop the image
        im = im.crop((x_min, y_min, x_max, y_max))
        # im = im[y_min:y_max, x_min:x_max]
        # print("after crop: ", im.size)
    return im, reject


def resize_image(im, resize_factor):
    im = im.resize(
        (
            int(im.size[0] * resize_factor),
            int(im.size[1] * resize_factor),
        ),
        Image.Resampling.LANCZOS,
    )
    return im


def pad_image(im):
    pad = args["padding"]
    width = args["preprocessed_image_width"]
    hgt = args["preprocessed_image_height"]
    new_im = Image.new("RGB", (width, hgt), (255, 255, 255))
    new_im.paste(im, (pad, pad))
    return new_im


def bucket(im):
    """
    selecting the bucket based on the width, and hgt
    of the image. This will provide us the appropriate
    resizing factor.
    """
    # [width, hgt, resize_factor]
    buckets = [
        [820, 86, 0.6],
        [615, 65, 0.8],
        [492, 52, 1],
        [410, 43, 1.2],
        [350, 37, 1.4],
    ]
    # current width, hgt
    # print("bucket: ", im.size)
    crop_width, crop_hgt = im.size[0], im.size[1]

    # find correct bucket
    resize_factor = args["resizing_factor"]
    for b in buckets:
        w, h, r = b
        if crop_width <= w and crop_hgt <= h:
            resize_factor = r

    return resize_factor


def downsampling(im):
    """
    if the image is too large and we won't be
    able to do bucketing, in that case, we need
    to dowmsample the image first and then
    will proceed with the preprocessing.
    It will be helpful if some random images
    will be send as input.
    """
    w, h = im.size
    max_h = args["max_input_hgt"]
    # we have come up with this number
    # from the buckets dimensions
    if h >= max_h:
        # need to calculate the ratio
        resize_factor = max_h / h

    im = im.resize(
        (
            int(im.size[0] * resize_factor),
            int(im.size[1] * resize_factor),
        ),
        Image.Resampling.LANCZOS,
    )
    return im
    
def create_image_tensors(im, img_name):

    # bucketing
    resize_factor = bucket(im)

    # resize
    im = resize_image(im, resize_factor)

    # padding
    im = pad_image(im)

    # convert to tensor
    convert = transforms.ToTensor()
    _im = convert(im)

    # save them as tensors
    torch.save(
        _im,
        f"{path_to_data}/image_tensors/{img_name}.pt",
    )

    return im

def create_image_graphs(im, img_name):
    ######### GETTING GRAPH AT PIXEL LEVEL ###########
            
    # im = np.array(im)
    
    # adj = image.img_to_graph(im)
    # edge_index = torch.tensor(np.vstack((adj.row, adj.col)))
    # edge_weight = adj.data
    
    w,h = im.size
    # print(w,h)
    edge_index, _ = grid(height=h, width=w)

    # already RGB -- did while padding
    im = np.array(im)
    im = im.flatten().reshape((-1,3))
    feature_matrix = torch.tensor(im.flatten().reshape((-1,3)))

    # Make a data object to store graph informaiton
    data = Data(x=feature_matrix, 
                edge_index=edge_index
            )
    
    # saving the graph
    torch.save(data, os.path.join(
            path_to_data, 
            f"image_graphs/{img_name}.pt")
    )

    # print(data)
    # print(data.edge_index.max() > data.x[0])

def main(img):
    im = Image.open(img).convert("L")

    # checking the size of the image
    w, h = im.size
    
    if h >= args["max_input_hgt"]:
        im = downsampling(im)

    # crop the image
    im, reject = crop_image(im)

    img_name = os.path.basename(img).split('.')[0]
    if not reject:
        im = create_image_tensors(im, img_name)
        create_image_graphs(im, img_name)
    else:
        return img

if __name__ == "__main__":

    path_to_images = os.path.join(path_to_data, "images")
    imgs = [os.path.join(path_to_images, f"{i}.png") 
            for i in range(len(os.listdir(path_to_images)))]
    
    # imgs = ["data/images/13.png"]  

    _graph_path = os.path.join(path_to_data, "image_graphs")
    _tnsr_path = os.path.join(path_to_data, "image_tensors")

    # main(imgs[0])

    for _p in [_graph_path, 
               _tnsr_path, 
               "./logs"]:
        if not os.path.exists(_p):
            os.mkdir(_p)

    with mp.Pool(args["ncpus"]) as pool:
        result = pool.map(main, imgs)
    
    blank_images = [i for i in result if i is not None]
    
    with open("logs/blank_images.lst", "w") as out:
        out.write("\n".join(str(item) for item in blank_images))
    
    tnsrs = sorted(
        [
            int(i.split(".")[0])
            for i in os.listdir(_tnsr_path)
        ]
    )
    os.chdir(path_to_data)
    for t in range(len(tnsrs)):
        os.rename(f"image_tensors/{tnsrs[t]}.pt", f"image_tensors/{t}.pt")
        os.rename(f"image_graphs/{tnsrs[t]}.pt", f"image_graphs/{t}.pt")
    