import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dtype=torch.float32):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode='circular',
            dtype=dtype
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        self.dtype = dtype

    def forward(self, x):
        x = x.to(dtype=self.dtype)
        x = self.tokenConv(x)
        return x

class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout, dtype=torch.float32):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model, dtype=dtype)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        self.dtype = dtype

    def forward(self, x):
        # Convert input to the specified dtype
        x = x.to(dtype=self.dtype)
        # Do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x)
        return self.dropout(x), n_vars
