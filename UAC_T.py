<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, src_mask=None):
        # Self-attention layer
        src2, _ = self.self_attn(y, y, x, attn_mask=src_mask)
        src = x + self.dropout(src2)
        src = self.norm1(src)

        # Feed-forward layer
        src2 = F.relu(self.linear1(src))
        src2 = self.linear2(src2)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(num_heads, hidden_dim, mlp_dim, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, src_mask=None):
        x = x + self.pos_embedding
        y = y + self.pos_embedding
        x = self.dropout(x)
        y = self.dropout(y)
        for layer in self.layers:
            src = layer(x, y, src_mask)
        return self.ln(src)


class Uac(nn.Module):
    def __init__(self, image_size, patch_size_v, num_layers, num_heads, hidden_dim, mlp_dim, dropout, num_frames):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size_v
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.conv_proj = nn.Conv3d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size_v, stride=patch_size_v
            )
        seq_length = ((image_size // patch_size_v[1]) ** 2) * (num_frames // patch_size_v[0])
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
        )
        self.seq_length = seq_length
        self.conv1d = nn.Conv1d(in_channels=seq_length, out_channels=seq_length // (num_frames // patch_size_v[0]), kernel_size=2, stride=2)

    def process_input(self, x, y):
        b, c, h, w = x.shape
        _, _, n, _, _ = y.shape
        x = x.unsqueeze(2).repeat(1, 1, n, 1, 1)
        p = self.patch_size[1]
        p2 = self.patch_size[0]
        n_h = h // p
        n_w = w // p
        n_n = n // p2
        x = self.conv_proj(x)
        y = self.conv_proj(y)
        x = x.reshape(b, self.hidden_dim, n_h * n_w * n_n)
        x = x.permute(0, 2, 1)
        y = y.reshape(b, self.hidden_dim, n_h * n_w * n_n)
        y = y.permute(0, 2, 1)

        return x, y

    def forward(self, x, y):
        b, c, h, w = x.shape
        x1, y1 = self.process_input(x, y)
        feature = self.encoder(x1, y1)
        feature = self.conv1d(feature)
        feature = feature.flatten(1)
        feature = feature.reshape(b, c, h, w)

        return feature

# # Example usage:
# #
# uac_t = Uac(image_size=224, patch_size_v=(2, 14, 14), num_layers=1, num_heads=2, hidden_dim=1176, mlp_dim=1024, dropout=0.1, num_frames=16)
# x = torch.randn((16, 3, 224, 224))
# y = torch.randn((16, 3, 16, 224, 224))
# z = uac_t(x, y)
# print(z.shape)
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class EncoderLayer(nn.Module):
    def __init__(self, num_heads, hidden_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.self_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(hidden_dim, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, src_mask=None):
        # Self-attention layer
        src2, _ = self.self_attn(y, y, x, attn_mask=src_mask)
        src = x + self.dropout(src2)
        src = self.norm1(src)

        # Feed-forward layer
        src2 = F.relu(self.linear1(src))
        src2 = self.linear2(src2)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self, seq_length, num_layers, num_heads, hidden_dim, mlp_dim, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.pos_embedding = nn.Parameter(torch.empty(1, seq_length, hidden_dim).normal_(std=0.02))
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.ModuleList([EncoderLayer(num_heads, hidden_dim, mlp_dim, dropout) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(hidden_dim)

    def forward(self, x, y, src_mask=None):
        x = x + self.pos_embedding
        y = y + self.pos_embedding
        x = self.dropout(x)
        y = self.dropout(y)
        for layer in self.layers:
            src = layer(x, y, src_mask)
        return self.ln(src)


class Uac(nn.Module):
    def __init__(self, image_size, patch_size_v, num_layers, num_heads, hidden_dim, mlp_dim, dropout, num_frames):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size_v
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.conv_proj = nn.Conv3d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size_v, stride=patch_size_v
            )
        seq_length = ((image_size // patch_size_v[1]) ** 2) * (num_frames // patch_size_v[0])
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
        )
        self.seq_length = seq_length
        self.conv1d = nn.Conv1d(in_channels=seq_length, out_channels=seq_length // (num_frames // patch_size_v[0]), kernel_size=2, stride=2)

    def process_input(self, x, y):
        b, c, h, w = x.shape
        _, _, n, _, _ = y.shape
        x = x.unsqueeze(2).repeat(1, 1, n, 1, 1)
        p = self.patch_size[1]
        p2 = self.patch_size[0]
        n_h = h // p
        n_w = w // p
        n_n = n // p2
        x = self.conv_proj(x)
        y = self.conv_proj(y)
        x = x.reshape(b, self.hidden_dim, n_h * n_w * n_n)
        x = x.permute(0, 2, 1)
        y = y.reshape(b, self.hidden_dim, n_h * n_w * n_n)
        y = y.permute(0, 2, 1)

        return x, y

    def forward(self, x, y):
        b, c, h, w = x.shape
        x1, y1 = self.process_input(x, y)
        feature = self.encoder(x1, y1)
        feature = self.conv1d(feature)
        feature = feature.flatten(1)
        feature = feature.reshape(b, c, h, w)

        return feature

# # Example usage:
# #
# uac_t = Uac(image_size=224, patch_size_v=(2, 14, 14), num_layers=1, num_heads=2, hidden_dim=1176, mlp_dim=1024, dropout=0.1, num_frames=16)
# x = torch.randn((16, 3, 224, 224))
# y = torch.randn((16, 3, 16, 224, 224))
# z = uac_t(x, y)
# print(z.shape)
>>>>>>> 661c694 ('init')
