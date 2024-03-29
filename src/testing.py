# -*- coding: utf-8 -*-

import torch
from utils.garbage_to_pad import garbage2pad
from torch_geometric.data import Data, Batch

def evaluate(
    model,
    img_tnsr_path,
    img_graph_path,
    batch_size,
    test_dataloader,
    criterion,
    device,
    vocab,
    isGraphPixel=True,
    is_test=False,
):
    model.eval()
    epoch_loss = 0

    if is_test:
        mml_seqs = open("logs/test_targets_100K.txt", "w")
        pred_seqs = open("logs/test_predicted_100K.txt", "w")

    with torch.no_grad():
        for i, (img, mml) in enumerate(test_dataloader):
            batch_size = mml.shape[0]
            mml = mml.to(device, dtype=torch.long)
            _imgs = list()
            
            _data_list = list()
            for im in img:
                # for vit patch encoder 
                _imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.txt"))

                # for pixel encoders
                # for vit pixel encoder, _imgs will be same
                if isGraphPixel:
                    G = torch.load(f"{img_graph_path}/{int(im.item())}.pt")
                    _data_list.append(G)
            
            if isGraphPixel:
                batch = Batch.from_data_list(_data_list).to(device)
            else:
                batch=None

            imgs = torch.stack(_imgs).to(device)

            """
            we will pass "mml" just to provide initial <sos> token.
            There will no teacher forcing while validation and testing.
            """
            outputs, preds = model(
                imgs,
                batch,  
                mml, 
                is_test=is_test,
            )  # O: (B, max_len, output_dim), preds: (B, max_len)

            if is_test:
                preds = garbage2pad(preds, vocab, is_test=is_test)
                output_dim = outputs.shape[-1]
                mml_reshaped = mml[:, 1:].contiguous().view(-1)
                outputs_reshaped = outputs.contiguous().view(
                    -1, output_dim
                )  # (B * max_len-1, output_dim)

            else:
                output_dim = outputs.shape[-1]            
                outputs_reshaped = outputs.contiguous().view(
                    -1, output_dim
                )  # (B * max_len-1, output_dim)
                mml_reshaped = mml[:, 1:].contiguous().view(-1)

            loss = criterion(outputs_reshaped, mml_reshaped)

            epoch_loss += loss.item()

            if is_test:
                for idx in range(batch_size):
                    # writing target eqn
                    mml_arr = [vocab.itos[imml] for imml in mml[idx, :]]
                    mml_seq = " ".join(mml_arr)
                    mml_seqs.write(mml_seq + "\n")

                    # writing pred eqn
                    pred_arr = [
                        vocab.itos[ipred] for ipred in preds.int()[idx, :]
                    ]
                    pred_seq = " ".join(pred_arr)
                    pred_seqs.write(pred_seq + "\n")

    net_loss = epoch_loss / len(test_dataloader)
    return net_loss