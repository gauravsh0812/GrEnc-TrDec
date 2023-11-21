# -*- coding: utf-8 -*-
"Main script to train the model."

import os
import random
import yaml
import numpy as np
import time
import json
import math
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from model.preprocessing import preprocess_data
from src.training import train
from src.testing import evaluate
from model.grenc_trdec_model.encoder import VIT_Encoder, Graph_Encoder
from model.grenc_trdec_model.decoder import Transformer_Decoder


# opening training_args file
with open('configs/config.yaml') as f:
	cfg = yaml.safe_load(f)
training_args = cfg["training"]
preprocessing_args = cfg["preprocessing_args"]
model_args = cfg["model"]

# torch.backends.cudnn.enabled = False

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

    INPUT_CHANNELS = model_args["input_channels"]
    OUTPUT_DIM = len(vocab)
    EMB_DIM = model_args["emb_dim"]
    HID_DIM = model_args["dec_dim"]
    DROPOUT = model_args["dropout"]
    MAX_LEN = model_args["max_len"]

    Gr_ENC = Graph_Encoder(INPUT_CHANNELS,
                        HID_DIM,
                        OUTPUT_DIM,
                        DROPOUT)

    Vit_ENC = VIT_Encoder()

    Tr_DEC = Transformer_Decoder()
    
    model = Image2MathML_Xfmer(ENC, DEC, VOCAB, DEVICE)

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
    if torch.cuda.is_available():
        if ddp:
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
            ) = preprocess_data(preprocessing_args)

            model = define_model(training_args, vocab, rank)
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
            ) = preprocess_data(preprocessing_args)
            model = define_model(training_args, vocab, device).to("cuda")

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

    # raw data paths
    img_tnsr_path = f"{preprocessing_args.data_path}/image_tensors"

    if not load_trained_model_for_testing:
        count_es = 0
        for epoch in range(epochs):
            if count_es <= early_stopping_counts:
                start_time = time.time()

                # training and validation
                train_loss = train(
                    model,
                    model_type,
                    img_tnsr_path,
                    train_dataloader,
                    optimizer,
                    criterion,
                    CLIP,
                    device,
                    ddp=ddp,
                    rank=rank,
                    scheduler=scheduler,
                    isScheduler=isScheduler,
                    whichScheduler=whichScheduler,
                )


                val_loss = evaluate(
                    model,
                    model_type,
                    img_tnsr_path,
                    batch_size,
                    val_dataloader,
                    criterion,
                    device,
                    vocab,
                    ddp=ddp,
                    rank=rank,
                    g2p=g2p,
                )

                end_time = time.time()
                # total time spent on training an epoch
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)

                if isScheduler:
                    if whichScheduler == "reduce_lr":
                        scheduler.step(val_loss)
                    else:
                        scheduler.step()

                # saving the current model for transfer learning
                if (not ddp) or (ddp and rank == 0):
                    torch.save(
                        model.state_dict(),
                        f"trained_models/{model_type}_{dataset_type}_{training_args['markup']}_latest.pt",
                    )

                if val_loss < best_valid_loss:
                    best_valid_loss = val_loss
                    count_es = 0
                    if (not ddp) or (ddp and rank == 0):
                        torch.save(
                            model.state_dict(),
                            f"trained_models/{model_type}_{dataset_type}_{training_args['markup']}_best.pt",
                        )

                elif early_stopping:
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
            f"trained_models/{model_type}_{dataset_type}_{training_args['markup']}_best.pt",
        )

    if ddp:
        dist.destroy_process_group()

    time.sleep(3)

    print(
        "loading best saved model: ",
        f"trained_models/{model_type}_{dataset_type}_{training_args['markup']}_best.pt",
    )
    try:
        # loading pre_tained_model
        model.load_state_dict(
            torch.load(
                f"trained_models/{model_type}_{dataset_type}_{training_args['markup']}_best.pt"
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

    epoch = "test_0"
    if training_args["beam_search"]:
        beam_params = [beam_k, alpha, min_length_bean_search_normalization]
    else:
        beam_params = None

    """
    bin comparison
    """
    if training_args["bin_comparison"]:
        print("comparing bin...")
        from bin_testing import bin_test_dataloader

        test_dataloader = bin_test_dataloader(
            training_args,
            vocab,
            device,
            start=training_args["start_bin"],
            end=training_args["end_bin"],
            length_based_binning=training_args["length_based_binning"],
            content_based_binning=training_args["content_based_binning"],
        )

    test_loss = evaluate(
        model,
        model_type,
        img_tnsr_path,
        batch_size,
        test_dataloader,
        criterion,
        device,
        vocab,
        beam_params=beam_params,
        is_test=True,
        ddp=ddp,
        rank=rank,
        g2p=g2p,
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