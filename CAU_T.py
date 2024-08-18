<<<<<<< HEAD
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        src2, _ = self.self_attn(x, x, y, attn_mask=src_mask)
        src = y + self.dropout(src2)
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


class Cau(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        seq_length = (image_size // patch_size) ** 2
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
        )
        self.seq_length = seq_length

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))

    def process_input(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(b, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, y):
        x = self.process_input(x)
        b, c, n, h, w = y.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        feature_list = []
        for i in range(n):
            frame_y = y[:, :, i:i+1, :, :]
            frame_y = frame_y.squeeze(2)
            frame_y = self.conv_proj(frame_y)
            frame_y = frame_y.reshape(b, self.hidden_dim, n_h * n_w)
            frame_y = frame_y.permute(0, 2, 1)
            feature = self.encoder(x, frame_y)
            feature = feature.flatten(1)
            feature = feature.reshape(b, c, h, w)
            feature = feature.unsqueeze(2)
            feature_list.append(feature)

        feature = torch.cat(feature_list, dim=2)

        return feature


# Example usage:
#
# cau_t = Cau(image_size=224, patch_size=14, num_layers=1, num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1)
# x = torch.randn((16, 3, 224, 224))
# y = torch.randn((16, 3, 16, 224, 224))
# z = cau_t(x, y)
# print(z.shape)
=======
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
        src2, _ = self.self_attn(x, x, y, attn_mask=src_mask)
        src = y + self.dropout(src2)
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


class Cau(nn.Module):
    def __init__(self, image_size, patch_size, num_layers, num_heads, hidden_dim, mlp_dim, dropout):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.mlp_dim = mlp_dim
        self.dropout = dropout

        self.conv_proj = nn.Conv2d(
                in_channels=3, out_channels=hidden_dim, kernel_size=patch_size, stride=patch_size
            )
        seq_length = (image_size // patch_size) ** 2
        self.encoder = Encoder(
            seq_length,
            num_layers,
            num_heads,
            hidden_dim,
            mlp_dim,
            dropout,
        )
        self.seq_length = seq_length

        if isinstance(self.conv_proj, nn.Conv2d):
            # Init the patchify stem
            fan_in = self.conv_proj.in_channels * self.conv_proj.kernel_size[0] * self.conv_proj.kernel_size[1]
            nn.init.trunc_normal_(self.conv_proj.weight, std=math.sqrt(1 / fan_in))

    def process_input(self, x):
        b, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        x = self.conv_proj(x)
        x = x.reshape(b, self.hidden_dim, n_h * n_w)
        x = x.permute(0, 2, 1)

        return x

    def forward(self, x, y):
        x = self.process_input(x)
        b, c, n, h, w = y.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        feature_list = []
        for i in range(n):
            frame_y = y[:, :, i:i+1, :, :]
            frame_y = frame_y.squeeze(2)
            frame_y = self.conv_proj(frame_y)
            frame_y = frame_y.reshape(b, self.hidden_dim, n_h * n_w)
            frame_y = frame_y.permute(0, 2, 1)
            feature = self.encoder(x, frame_y)
            feature = feature.flatten(1)
            feature = feature.reshape(b, c, h, w)
            feature = feature.unsqueeze(2)
            feature_list.append(feature)

        feature = torch.cat(feature_list, dim=2)

        return feature


# Example usage:
#
# cau_t = Cau(image_size=224, patch_size=14, num_layers=1, num_heads=2, hidden_dim=588, mlp_dim=1024, dropout=0.1)
# x = torch.randn((16, 3, 224, 224))
# y = torch.randn((16, 3, 16, 224, 224))
# z = cau_t(x, y)
# print(z.shape)
>>>>>>> 661c694 ('init')
