import torch

def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask

trg = torch.rand((350,20))
trg = trg[:-1, :]  # (max_len-1, B)
sequence_length = trg.shape[0]  
trg_attn_mask = generate_square_subsequent_mask(sequence_length).to("cuda:0")