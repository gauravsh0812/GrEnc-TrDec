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
from model.grenc_trdec_model.model import ClipModel
# from model.grenc_trdec_model.decoding_model import DecodingModel
from model.grenc_trdec_model.vit_encoder import VisionTransformer
from model.grenc_trdec_model.xfmer_encoder import Transformer_Encoder
from model.grenc_trdec_model.xfmer_decoder import Transformer_Decoder

# opening training_args file
with open('configs/config.yaml') as f:
	cfg = Box(yaml.safe_load(f))
buiding_graph_args = cfg.building_graph
training_args = cfg.training
preprocessing_args = cfg.preprocessing
vit_args = cfg.model.vit
xfmer_args = cfg.model.xfmer_enc
xfmer_dec_args = cfg.model.xfmer_dec

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

    isVitPixel = cfg.model.isVitPixel
    dropout = cfg.model.dropout
    
    # assert isGraphPixel or isVitPixel, "Need to select either one of the encoder or both of them."
    if (not isVitPixel):
        print(" NO PIXEL ENCODER IS PRESENT!!")
    
    image_w = buiding_graph_args.preprocessed_image_width
    image_h = buiding_graph_args.preprocessed_image_height
    
    assert image_w % cfg.model.vit.patch_size == 0
    assert image_h % cfg.model.vit.patch_size == 0
    
    Vit_ENC = VisionTransformer(
        img_size=[image_w,image_h],
        patch_size=vit_args.patch_size,
        pixel_patch_size=vit_args.pixel_patch_size,
        in_chns=vit_args.input_channels,
        embed_dim=vit_args.emb_dim,
        depth=vit_args.depth,
        n_heads=vit_args.nheads,
        mlp_ratio=vit_args.mlp_ratio,
        qkv_bias=vit_args.qkv_bias,
        p=dropout,
        attn_p=dropout,
        isVitPixel=isVitPixel,
    )

    n_patches = (
        image_w // cfg.model.vit.patch_size
        ) * (
        image_h // cfg.model.vit.patch_size
    )

    Tr_ENC = Transformer_Encoder(
        # vit_emb_dim=vit_args.emb_dim,
        hid_dim=xfmer_args.hid_dim,
        nheads=xfmer_args.nheads,
        n_patches = n_patches,
        dropout=dropout,
        device=device,
        max_len=xfmer_args.max_len,
        n_xfmer_encoder_layers=xfmer_args.n_xfmer_encoder_layers,
        dim_feedfwd=xfmer_args.dim_feedfwd,
    ) 
    
    Tr_DEC = Transformer_Decoder(
        tr_enc_hid_dim=xfmer_args.hid_dim,
        dec_emb_dim=xfmer_dec_args.emb_dim,
        dec_hid_dim=xfmer_dec_args.hid_dim,
        nheads=xfmer_dec_args.nheads,
        output_dim=len(vocab),
        # n_patches=n_patches,
        dropout=dropout,
        max_len=xfmer_args.max_len,
        n_xfmer_decoder_layers=xfmer_dec_args.n_xfmer_decoder_layers,
        dim_feedfwd=xfmer_dec_args.dim_feedfwd,
        device=device,
    )

    # decoding_model = DecodingModel(
    #     vocab, 
    #     device,
    #     Vit_ENC,
    #     Tr_DEC, 
    #     isVitPixel=isVitPixel,
    # )


    model = ClipModel(
        vocab, 
        device,
        xfmer_dec_args.emb_dim,  # trying
        vit_args.emb_dim,  
        xfmer_args.emb_dim, 
        xfmer_args.hid_dim,
        cfg.model.projection_dim,
        xfmer_args.max_len,
        dropout,
        cfg.model.temperature,
        Vit_ENC,
        Tr_ENC, 
        Tr_DEC,
        isVitPixel=isVitPixel,
    )

    return model#, decoding_model


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
    world_size = training_args["world_size"]
    load_trained_model_for_testing = training_args["load_trained_model_for_testing"]
    early_stopping_counts = training_args.early_stopping
    
    if (training_args.wandb):
        if (not ddp) or (ddp and rank == 0): 
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
    preprocessing_args["world_size"] = world_size
    preprocessing_args["rank"] = rank

    if torch.cuda.is_available():
        if ddp:
            # add a few args for temporarily purpose
            # this is to avoid replicating in config file
            # create default process group
            dist.init_process_group("nccl", rank=rank, world_size=world_size)
            # add rank to training_args
            training_args["rank"] = rank
            device = f"cuda:{rank}"
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(preprocessing_args)

            # model, decoding_model = define_model(vocab, rank)
            model = define_model(vocab, rank)
            model = DDP(
                model.to(device),
                device_ids=[rank],
                output_device=rank,
                find_unused_parameters=True,
            )
            # decoding_model = DDP(
            #     decoding_model.to(device),
            #     device_ids=[rank],
            #     output_device=rank,
            #     find_unused_parameters=True,
            # )

        else:
            print(f"using gpu {str(gpus)}...")
            # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpus)
            device = torch.device(f"cuda:{gpus}")
            
            (
                train_dataloader,
                test_dataloader,
                val_dataloader,
                vocab,
            ) = preprocess_dataset(preprocessing_args)
            # model,decoding_model = define_model(vocab, device).to("cuda")
            model = define_model(vocab, device).to("cuda:0")

    else:
        import warnings

        warnings.warn("No GPU input has provided. Falling back to CPU. ")
        device = torch.device("cpu")
        (
            train_dataloader,
            test_dataloader,
            val_dataloader,
            vocab,
        ) = preprocess_dataset(preprocessing_args)
        # model,decoding_model = define_model(vocab, device).to(device)
        model = define_model(vocab, rank).to(device)

    print("MODEL: ")
    print(f"The model has {count_parameters(model)} trainable parameters")

    # intializing loss function
    criterion = torch.nn.CrossEntropyLoss(ignore_index=vocab.stoi["<pad>"])

    # optimizer
    optimizer_clip = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )
    
    optimizer_dec = torch.optim.AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(beta1, beta2),
    )

    # multistep_lr scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer_clip,
        # step_size=training_args.scheduler_step_size,
        milestones=[50],
        gamma=training_args.scheduler_gamma,
    )

    best_valid_loss = float("inf")
    
    if training_args.wandb:
        if (not ddp) or (ddp and rank == 0):
            wandb.watch(model)

    # raw data paths
    img_tnsr_path = f"{preprocessing_args.path_to_data}/image_tensors"

    # # model only for decoding part while testing or validating
    # decoder_model_path = cfg.model.decoder_model_path
    # # loading pre_tained_model for decoding
    # decoding_model.load_state_dict(torch.load(decoder_model_path))

    if not load_trained_model_for_testing:
        count_es = 0
        for epoch in range(epochs):
            if count_es <= early_stopping_counts:
                start_time = time.time()

                # training and validation
                train_loss = train(
                    model,
                    img_tnsr_path,
                    train_dataloader,
                    optimizer_clip,
                    optimizer_dec,
                    clip,
                    device,
                    ddp=ddp,
                    rank=rank,
                )

                val_loss = evaluate(
                    model,
                    # decoding_model,
                    img_tnsr_path,
                    criterion,
                    val_dataloader,
                    device,
                    vocab,
                )

                if training_args.wandb:
                    if (not ddp) or (ddp and rank == 0):
                        wandb.log({"train_loss": train_loss})
                        wandb.log({"val_loss": val_loss})

                end_time = time.time()
                # total time spent on training an epoch
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
        
                scheduler.step()
                        
                # saving the current model for transfer learning
                if (not ddp) or (ddp and rank == 0):
                    torch.save(
                        model.state_dict(),
                        f"trained_models/{preprocessing_args['markup']}_latest.pt",
                    )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    count_es = 0
                    if (not ddp) or (ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{preprocessing_args['markup']}_best.pt",
                        )
                        if training_args.wandb:
                            wandb.save(f"trained_models/{preprocessing_args['markup']}_best.pt")

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
            f"trained_models/{preprocessing_args['markup']}_best.pt",
        )

    if ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/{preprocessing_args['markup']}_best.pt",
    )
    # loading pre_tained_model
    model.load_state_dict(
        torch.load(
            f"trained_models/{preprocessing_args['markup']}_best.pt"
        )
    )

    test_loss = evaluate(
        model,
        # decoding_model,
        img_tnsr_path,
        criterion,        
        test_dataloader,
        device,
        vocab,
        is_test=True,
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
def ddp_main(world_size,):    
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    mp.spawn(train_model, args=(), nprocs=world_size, join=True)

if __name__ == "__main__":
    if training_args["ddp"]:
        gpus = training_args["gpus"]
        world_size = training_args["world_size"]
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29800"
        ddp_main(world_size)
    else:
        train_model()