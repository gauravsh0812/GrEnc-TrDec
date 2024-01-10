# -*- coding: utf-8 -*-
import torch
from tqdm.auto import tqdm

def train(
    model,
    img_tnsr_path,
    train_dataloader,
    optimizer,
    criterion,
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
        _data_list = list()
        for im in img:
            # for vit patch encoder 
            _imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.pt"))

        imgs = torch.stack(_imgs).to(device)
        
        # setting gradients to zero
        optimizer.zero_grad()

        loss = model(
            imgs,
            mml,
        )
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

        if (not ddp) or (ddp and rank == 0):
            desc = 'Loss: %.4f - Learning Rate: %.6f' % (loss.item(), optimizer.param_groups[0]['lr'])
            tset.set_description(desc)

    net_loss = epoch_loss / len(train_dataloader)
    return net_loss