"Main script to train the model."

import os
import sys
import random
import yaml
import numpy as np
import time
import math
import torch
import wandb
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from src.training import train
from src.testing import evaluate
from model.preprocessing import preprocess_data
from model.grenc_trdec_model.model import Grenc_Trdec_Model
from model.grenc_trdec_model.graph_encoder import Graph_Encoder
from model.grenc_trdec_model.vit_encoder import VisionTransformer
from model.grenc_trdec_model.decoder import Transformer_Decoder
from box import Box

# opening training_args file
with open('configs/config.yaml') as f:
	cfg = Box(yaml.safe_load(f))
buiding_graph_args = cfg["building_graph"]
training_args = cfg["training"]
preprocessing_args = cfg["preprocessing"]
graph_args = cfg["model"]["graph_model"]
vit_args = cfg.model.vit
xfmer_args = cfg["model"]["decoder_transformer"]

cmd = "python model/grenc_trdec_model/decoder.py"
os.system(cmd)