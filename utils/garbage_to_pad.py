def garbage2pad(preds, vocab, is_test=False):
    """
    all garbage tokens will be converted to <pad> token
    "garbage" tokens: tokens after <eos> token

    params:
    pred: predicted eqns (B, seq_len/max_len)

    return:
    pred: cleaned pred eqn
    """

    vocab.stoi["<pad>"]
    eos_idx = vocab.stoi["<eos>"]
    for b in range(preds.shape[0]):
        try:
            # cleaning pred
            eos_pos = (preds[b, :] == eos_idx).nonzero(as_tuple=False)[0]
            preds[b, :] = preds[b, : eos_pos + 1]  # pad_idx
        except:
            pass

    return preds