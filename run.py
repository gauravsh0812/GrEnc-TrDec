# -*- coding: utf-8 -*-
"Main script to train the model."

import os
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
from box import Box
from src.training import train
from src.testing import evaluate
from model.preprocessing.preprocess_data import preprocess_dataset
from model.grenc_trdec_model.model import Grenc_Trdec_Model
from model.grenc_trdec_model.graph_encoder import Graph_Encoder
from model.grenc_trdec_model.vit_encoder import VisionTransformer
from model.grenc_trdec_model.decoder import Transformer_Decoder


# opening training_args file
with open('configs/config.yaml') as f:
	cfg = Box(yaml.safe_load(f))
buiding_graph_args = cfg.building_graph
training_args = cfg.training
preprocessing_args = cfg.preprocessing
graph_args = cfg.model.graph_model
vit_args = cfg.model.vit
xfmer_args = cfg.model.decoder_transformer

# for deterministic results, make it False.
# to optimize performance, make it True, but that 
# might affect the results a bit at atomic level.
torch.backends.cudnn.enabled = False

def set_random_seed(SEED):
    # set up seed
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)


def define_model(vocab, device):
    """
    defining the model
    initializing encoder, decoder, and model
    """

    print("defining model...")

    isGraphEnc = cfg.model.isGraphEnc
    isVitEnc = cfg.model.isVitEnc
    gr_dropout = graph_args.dropout
    
    assert isGraphEnc or isVitEnc, "Need to select either one of the encoder or both of them."

    if isGraphEnc:
        Gr_ENC = Graph_Encoder(in_channels=graph_args.input_channels,
                            hidden_channels=graph_args.hid_dim,
                            vit_embed_dim=vit_args.emb_dim,
                            dropout=gr_dropout,
                            )

    if isVitEnc:
        image_w = buiding_graph_args.preprocessed_image_width
        image_h = buiding_graph_args.preprocessed_image_height

        Vit_ENC = VisionTransformer(
                        img_size=[image_w,image_h],
                        patch_size=vit_args.patch_size,
                        in_chns=graph_args.input_channels,
                        embed_dim=vit_args.emb_dim,
                        depth=vit_args.depth,
                        n_heads=vit_args.nheads,
                        mlp_ratio=vit_args.mlp_ratio,
                        qkv_bias=vit_args.qkv_bias,
                        p=gr_dropout,
                        attn_p=gr_dropout,
                        )

    assert cfg.building_graph.preprocessed_image_width % cfg.model.vit.patch_size == 0
    assert cfg.building_graph.preprocessed_image_height % cfg.model.vit.patch_size == 0

    n_patches = (
        cfg.building_graph.preprocessed_image_width // cfg.model.vit.patch_size
        ) * (
        cfg.building_graph.preprocessed_image_height // cfg.model.vit.patch_size
        )

    Tr_DEC = Transformer_Decoder(
        emb_dim=xfmer_args.emb_dim,
        dec_hid_dim=xfmer_args.dec_hid_dim,
        gr_hid_dim=graph_args.hid_dim,
        nheads=xfmer_args.nheads,
        output_dim=len(vocab),
        n_patches=n_patches,
        dropout=gr_dropout,
        max_len=xfmer_args.max_len,
        n_xfmer_decoder_layers=xfmer_args.n_xfmer_decoder_layers,
        dim_feedfwd=xfmer_args.dim_feedfwd,
        device=device,
    )

    model = Grenc_Trdec_Model(vocab, 
                            device,
                            Gr_ENC, 
                            Vit_ENC,
                            Tr_DEC, 
                            )

    return model


def init_weights(m):
    """
    initializing the model wghts with values
    drawn from normal distribution.
    else initialize them with 0.
    """
    for name, param in m.named_parameters():
        if "weight" in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)


def count_parameters(model):
    """
    counting total number of parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def epoch_time(start_time, end_time):
    """
    epoch timing
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_model(rank=None,):
    # parameters
    epochs = training_args["epochs"]
    batch_size = training_args["batch_size"]
    learning_rate = training_args["learning_rate"]
    weight_decay = training_args["weight_decay"]
    [beta1, beta2] = training_args["betas"]
    clip = training_args["clip"]
    seed = training_args["seed"]
    ddp = training_args["ddp"]
    gpus = training_args["gpus"]
    load_trained_model_for_testing = training_args["load_trained_model_for_testing"]
    early_stopping_counts = training_args.early_stopping
    
    if training_args.wandb:
        # initiate the wandb    
        wandb.init()
        wandb.config.update(cfg)

    # set_random_seed
    set_random_seed(seed)

    # to save trained model and logs
    folders = ["trained_models", "logs"]
    for f in folders:
        if not os.path.exists(f):
            os.mkdir(f)

    # to log losses
    loss_file = open("logs/loss_file.txt", "w")

    # defining model using DataParallel
    preprocessing_args["batch_size"] = training_args.batch_size
    preprocessing_args["max_len"] = xfmer_args.max_len
    preprocessing_args["ddp"] = training_args.ddp
    if torch.cuda.is_available():
        if ddp:
            # add a few args for temporarily purpose
            # this is to avoid replicating in config file
            # create default process group
            dist.init_process_group("nccl", rank=rank, world_size=len(gpus))
            # add rank to training_args
            training_args["rank"] = rank
            device = f"cuda:{rank}"
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(preprocessing_args)

            model = define_model(vocab, rank)
            model = DDP(
                model.to(f"cuda:{rank}"),
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
            device = torch.device("cuda")
            
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(preprocessing_args)
            model = define_model(vocab, device).to("cuda")

    else:
        import warnings

        warnings.warn("No GPU input has provided. Falling back to CPU. ")
        device = torch.device("cpu")
        (
            train_dataloader,
            test_dataloader,
            val_dataloader,
            vocab,
        ) = preprocess_data(preprocessing_args)
        model = define_model(training_args, vocab, device).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # optimizer
    optimizer = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    
    # multistep_lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=[training_args.scheduler_step_size],
        gamma=training_args.scheduler_gamma,
        last_epoch=-1,
        verbose=False
    )

    best_valid_loss = float("inf")
    
    if training_args.wandb:
        wandb.watch(model)

    # raw data paths
    img_tnsr_path = f"{preprocessing_args.path_to_data}/image_tensors"
    img_graph_path = f"{preprocessing_args.path_to_data}/image_graphs"

    if not load_trained_model_for_testing:
        count_es = 0
        for epoch in range(epochs):
            if count_es <= early_stopping_counts:
                start_time = time.time()

                # training and validation
                train_loss = train(
                    model,
                    img_tnsr_path,
                    img_graph_path,
                    train_dataloader,
                    optimizer,
                    criterion,
                    clip,
                    device,
                    isGraphEnc=cfg["model"]["isGraphEnc"],
                    isVitEnc=cfg["model"]["isVitEnc"],
                    ddp=ddp,
                    rank=rank,
                )

                if training_args.wandb:
                    wandb.log({"train_loss": train_loss})

                val_loss = evaluate(
                    model,
                    img_tnsr_path,
                    img_graph_path,
                    batch_size,
                    val_dataloader,
                    criterion,
                    device,
                    vocab,
                    isGraphEnc=cfg["model"]["isGraphEnc"],
                    isVitEnc=cfg["model"]["isVitEnc"],
                    ddp=ddp,
                    rank=rank,
                )

                if training_args.wandb:
                    wandb.log({"val_loss": val_loss})

                end_time = time.time()
                # total time spent on training an epoch
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
                scheduler.step()
                        
                # saving the current model for transfer learning
                if (not ddp) or (ddp and rank == 0):
                    torch.save(
                        model.state_dict(),
                        f"trained_models/{training_args['markup']}_latest.pt",
                    )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    count_es = 0
                    if (not ddp) or (ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{training_args['markup']}_best.pt",
                        )
                        if training_args.wandb:
                            wandb.save(f"trained_models/{training_args['markup']}_best.pt")

                else:
                    count_es += 1

                # logging
                if (not ddp) or (ddp and rank == 0):
                    print(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s"
                    )
                    print(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}"
                    )
                    print(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}"
                    )

                    loss_file.write(
                        f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s\n"
                    )
                    loss_file.write(
                        f"\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}\n"
                    )
                    loss_file.write(
                        f"\t Val. Loss: {val_loss:.3f} |  Val. PPL: {math.exp(val_loss):7.3f}\n"
                    )

            else:
                print(
                    f"Terminating the training process as the validation \
                    loss hasn't been reduced from last {early_stopping_counts} epochs."
                )
                break

        print(
            "best model saved as:  ",
            f"trained_models/{training_args['markup']}_best.pt",
        )

    if ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/{training_args['markup']}_best.pt",
    )
    try:
        # loading pre_tained_model
        model.load_state_dict(
            torch.load(
                f"trained_models/{training_args['markup']}_best.pt"
            )
        )

    except:
        try:
            # removing "module." from keys
            pretrained_dict = {
                key.replace("module.", ""): value
                for key, value in model.state_dict().items()
            }
        except:
            # adding "module." in keys
            pretrained_dict = {
                f"module.{key}": value
                for key, value in model.state_dict().items()
            }

        model.load_state_dict(pretrained_dict)


    test_loss = evaluate(
        model,
        img_tnsr_path,
        img_graph_path,
        batch_size,
        test_dataloader,
        criterion,
        device,
        vocab,
        isGraphEnc=cfg["model"]["isGraphEnc"],
        isVitEnc=cfg["model"]["isVitEnc"],
        is_test=True,
        ddp=ddp,
        rank=rank,
    )

    if (not ddp) or (ddp and rank == 0):
        print(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
        )
        loss_file.write(
            f"| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |"
        )

    # stopping time
    print(time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()))


# for DDP
def ddp_main(world_size, gpus):    
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    mp.spawn(train_model, args=(), nprocs=world_size, join=True)

if __name__ == "__main__":
    if training_args["ddp"]:
        gpus = training_args["gpus"]
        world_size = len(gpus)
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29890"
        ddp_main(world_size, gpus)
    else:
        train_model()