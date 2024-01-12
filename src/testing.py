# -*- coding: utf-8 -*-

import torch
from utils.garbage_to_pad import garbage2pad

def evaluate(
    model,
    decoding_model,
    img_tnsr_path,
    criterion,
    test_dataloader,
    device,
    vocab,
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
            for im in img:
                # for vit patch encoder 
                _imgs.append(torch.load(f"{img_tnsr_path}/{int(im.item())}.pt"))

            imgs = torch.stack(_imgs).to(device)

            """
            we will pass "mml" just to provide initial <sos> token.
            There will no teacher forcing while validation and testing.
            """
            encodded_img = model(
                imgs,
                only_img=True,
            )  # O: (B, max_len, output_dim)

            decoding_model(
                
            )

            epoch_loss += loss.item()

            # if is_test:
            #     for idx in range(batch_size):
            #         # writing target eqn
            #         mml_arr = [vocab.itos[imml] for imml in mml[idx, :]]
            #         mml_seq = " ".join(mml_arr)
            #         mml_seqs.write(mml_seq + "\n")

            #         # writing pred eqn
            #         pred_arr = [
            #             vocab.itos[ipred] for ipred in preds.int()[idx, :]
            #         ]
            #         pred_seq = " ".join(pred_arr)
            #         pred_seqs.write(pred_seq + "\n")

    net_loss = epoch_loss / len(test_dataloader)
    return net_loss