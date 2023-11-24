# -*- coding: utf-8 -*-
import torch

# new addition
import math
import random

from tqdm.auto import tqdm
from src.testing import evaluate

def train(
    model,
    img_tnsr_path,
    train_dataloader,
    optimizer,
    criterion,
    clip,
    device,
    isGraphEnc=True,
    ddp=False,
    rank=None,
    scheduler=None,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    tset = tqdm(iter(train_dataloader))

    for i, (img, mml) in enumerate(tset):
        # mml: (B, max_len)
        # img: (B, in_channel, H, W)
        mml = mml.to(device, dtype=torch.long)
        imgs = list()
        graphs = list()
        for im in img:
            imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.txt"))
            if isGraphEnc:
                graphs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.txt"))
        img = torch.stack(imgs).to(device)
        if isGraphEnc:
            graphs = torch.stack(graphs).to(device)

        # setting gradients to zero
        optimizer.zero_grad()

        outputs, _ = model(img, mml, graphs)  # (B, max_len, output_dim)
        output_dim = outputs.shape[-1]

        # avoiding <sos> token while Calculating loss
        mml = mml[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, output_dim)

        loss = criterion(outputs, mml)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss