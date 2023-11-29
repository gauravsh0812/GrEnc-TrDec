# -*- coding: utf-8 -*-

import torch
from utils.garbage_to_pad import garbage2pad

def evaluate(
    model,
    img_tnsr_path,
    img_graph_path,
    batch_size,
    test_dataloader,
    criterion,
    device,
    vocab,
    isGraphEnc=True,
    isVitEnc=True,
    is_test=False,
    ddp=False,
    rank=None,
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
            _features_list = list()
            _edge_list = list()
            for im in img:
                if isGraphEnc:
                    G = torch.load(f"{img_tnsr_path}/{int(im.item())}.pt")
                    _features_list.append(G.x.float())
                    _edge_list.append(G.edge_index) 
                
                if isVitEnc:
                    _imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.txt"))
            
            if isGraphEnc:
                features_list = torch.stack(_features_list).to(device)
                edges_list = torch.stack(_edge_list).to(device)
            else:
                features_list = None
                edges_list = None

            if isVitEnc:
                imgs = torch.stack(_imgs).to(device)
            else:
                imgs = None

            """
            we will pass "mml" just to provide initial <sos> token.
            There will no teacher forcing while validation and testing.
            """
            outputs, preds = model(
                imgs, features_list, edges_list, mml, is_test=is_test
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