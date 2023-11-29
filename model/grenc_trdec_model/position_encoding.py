import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, model_dimension, dropout, max_len):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, model_dimension)  # (max_len, model_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(
            1
        )  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, model_dimension, 2).float()
            * (-math.log(10000.0) / model_dimension)
        )  # ([model_dim//2])
        pe[:, 0::2] = torch.sin(position * div_term)  # (max_len, model_dim//2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (max_len, model_dim//2)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, model_dim)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.tensor) -> torch.tensor:
        # x: (max_len, B, embed_dim)
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
    

class Positional_features(nn.Module):
    def __init__(self,):
        self.min_timescale: float = 1.0
        self.max_timescale: float = 1.0e4
    
    def get_range_vector(self, size: int, device) -> torch.Tensor:
        return torch.arange(0, size, dtype=torch.long, device=device)


    def __call__(self,x):
        """
        Implements the frequency-based positional encoding described
        in `Attention is all you Need
        Parameters
        ----------
        tensor : ``torch.Tensor``
            a Tensor with shape (batch_size, timesteps, hidden_dim).
        min_timescale : ``float``, optional (default = 1.0)
            The largest timescale to use.
        Returns
        -------
        The input tensor augmented with the sinusoidal frequencies.
        """

        _, timesteps, hidden_dim = x.size()

        timestep_range = self.get_range_vector(timesteps, x.device).data.float()
        # We're generating both cos and sin frequencies,
        # so half for each.
        num_timescales = hidden_dim // 2
        timescale_range = self.get_range_vector(
            num_timescales, x.device
        ).data.float()

        log_timescale_increments = math.log(
            float(self.max_timescale) / float(self.min_timescale)
        ) / float(num_timescales - 1)
        inverse_timescales = self.min_timescale * torch.exp(
            timescale_range * -log_timescale_increments
        )

        # Broadcasted multiplication - shape (timesteps, num_timescales)
        scaled_time = timestep_range.unsqueeze(1) * inverse_timescales.unsqueeze(0)
        # shape (timesteps, 2 * num_timescales)
        sinusoids = torch.randn(
            scaled_time.size(0), 2 * scaled_time.size(1), device=x.device
        )
        sinusoids[:, ::2] = torch.sin(scaled_time)
        sinusoids[:, 1::2] = torch.cos(scaled_time)
        if hidden_dim % 2 != 0:
            # if the number of dimensions is odd, the cos and sin
            # timescales had size (hidden_dim - 1) / 2, so we need
            # to add a row of zeros to make up the difference.
            sinusoids = torch.cat(
                [sinusoids, sinusoids.new_zeros(timesteps, 1)], 1
            )
        return x + sinusoids.unsqueeze(0)