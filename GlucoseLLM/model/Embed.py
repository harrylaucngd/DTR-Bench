import torch
import torch.nn as nn
from torch import Tensor


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class ReplicationPad1d(nn.Module):
    def __init__(self, padding: int) -> None:
        super(ReplicationPad1d, self).__init__()
        self.padding = padding

    def forward(self, input: Tensor) -> Tensor:
        replicate_padding = input[:, :, -1].unsqueeze(-1).repeat(1, 1, self.padding)
        output = torch.cat([input, replicate_padding], dim=-1)
        return output


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, n_var, patch_len,  stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len*n_var, d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)  # (batch, n_vars, n_patches, patch_len)
        x = x.permute(0, 2, 1, 3)  # (batch, n_patches, n_vars, patch_len)
        x = torch.reshape(x, (x.shape[0], x.shape[1], x.shape[2] * x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)  # (batch, n_patches, d_model)
        return self.dropout(x), n_vars