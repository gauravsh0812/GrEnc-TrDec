# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm

def train(
    model,
    img_tnsr_path,
    train_dataloader,
    optimizer_clip,
    optimizer_dec,
    clip,
    device,
    ddp=False,
    rank=None,
):
    # train mode is ON i.e. dropout and normalization tech. will be used
    model.train()

    epoch_loss = 0

    tset = tqdm(iter(train_dataloader))

    for i, (img, mml) in enumerate(tset):
        # mml: (B, max_len)
        # img: (B, in_channel, H, W)
        mml = mml.to(device, dtype=torch.long)
        
        _imgs = list()
        for im in img:
            # for vit patch encoder 
            _imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.pt"))

        imgs = torch.stack(_imgs).to(device)
        
        # =========== training CLIP ================ #

        # setting gradients to zero
        optimizer_clip.zero_grad()
        loss_clip = model(
            imgs,
            mml,
            train_dec=False,
        )

        loss_clip.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer_clip.step()

        # ================= training Decoder ================= #
        
        optimizer_dec.zero_grad()
        _, _, loss_dec = model(
            imgs, 
            mml, 
            train_dec=True,
        )

        loss_dec.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer_dec.step()

        if (not ddp) or (ddp and rank == 0):
            tset.set_postfix(train_loss=f'CLIP: {loss_clip.item()}, DEC: {loss_dec.item()}', lr=optimizer_clip.param_groups[0]['lr'])

        epoch_loss += loss_clip.item() + loss_dec.item()

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss