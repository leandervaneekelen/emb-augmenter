from cgi import test
import math

from torch import nn, Tensor
import torch
import torch.nn as nn
from torch.nn import ReLU, ELU


class GeneratorMLP(nn.Module):
    def __init__(self, n_tokens=1024, dropout=0.0) -> None:
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(n_tokens * 2, n_tokens),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens, n_tokens // 2),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens // 2, n_tokens // 4),
            ELU(),
            nn.AlphaDropout(dropout),
        )

        self.decoder = nn.Sequential(
            nn.Linear(n_tokens // 4, n_tokens // 2),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens // 2, n_tokens // 2),
            ELU(),
            nn.AlphaDropout(dropout),
            nn.Linear(n_tokens // 2, n_tokens),
        )

    def forward(self, x, z):
        """
        Forward pass.
            Args:
                - x: original feature [B, 1024 x 1]
                - z: random noise [B, 1024 x1]
            Returns:
                - x_aug: augmented version of x [B, 1024]
        """
        x_aug = torch.cat([x, z], dim=1).squeeze()
        x_aug = self.encoder(x_aug)
        x_aug = self.decoder(x_aug)

        return x_aug


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1024):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe
        return self.dropout(x)


class GeneratorTransformer(nn.Module):
    def __init__(
        self,
        n_tokens=1024,
        n_heads=1,
        emb_dim=8,
        dropout=0.0,
    ):
        super(GeneratorTransformer, self).__init__()

        self.augment_channel = nn.Linear(2, emb_dim)
        self.pos_encoding = PositionalEncoding(
            d_model=emb_dim, dropout=dropout, max_len=n_tokens
        )
        self.transformer = nn.TransformerEncoderLayer(
            d_model=emb_dim,
            dim_feedforward=emb_dim,
            nhead=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.reduce_channel = nn.Linear(emb_dim, 1)

    def forward(self, x, z):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: random noise [B x 1024 x1]
            Returns:
                - x_aug: augmented version of x [B x 1024]
        """
        x, z = x.unsqueeze(2), z.unsqueeze(2)
        x_aug = torch.cat([x, z], dim=2)  # concat original and noise
        x_aug = torch.permute(x_aug, (1, 0, 2))  # seq len first
        x_aug = self.augment_channel(x_aug)  # increast channel from 2 to emb_dim
        x_aug = self.pos_encoding(x_aug)  # position encoding
        x_aug = self.transformer(x_aug)  # apply transformer
        x_aug = self.reduce_channel(x_aug)  # reduce channel
        x_aug = torch.permute(x_aug.squeeze(dim=2), (1, 0))  # batch first
        return x_aug


class GeneratorIndependent(nn.Module):
    def __init__(
        self,
    ):
        super(GeneratorIndependent, self).__init__()

        self.all_mlps = []
        for _ in range(1024):
            mlp = nn.Sequential(nn.Linear(2, 4), nn.ReLU(), nn.Linear(4, 1))
            # mlp = nn.Sequential(
            #     nn.Linear(2, 1),
            # )
            self.all_mlps.append(mlp)
        self.all_mlps = nn.ModuleList(self.all_mlps)

    def forward(self, x, z):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: random noise [B x 1024 x1]
            Returns:
                - x_aug: augmented version of x [B x 1024]
        """
        x, z = x.unsqueeze(2), z.unsqueeze(2)
        data = torch.cat([x, z], dim=2)  # concat original and noise
        data = torch.permute(data, (1, 0, 2))  # seq len first
        augmentations = []
        for i in range(1024):
            o = self.all_mlps[i](data[i, :, :])
            augmentations.append(o)
        augmentations = torch.cat(augmentations, dim=1)
        return augmentations


class GeneratorIndependentFast(nn.Module):
    def __init__(
        self,
    ):
        super(GeneratorIndependentFast, self).__init__()

        # self.mlp = nn.Sequential(
        #     nn.Linear(2, 32),
        #     nn.ReLU(),
        #     nn.Linear(32, 8),
        #     nn.ReLU(),
        #     nn.Linear(8, 1)
        # )

        self.mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

        # self.mlp = nn.Sequential(
        #     nn.Linear(2, 1, bias=False),
        # )

    def forward(self, x, z):
        """
        Forward pass.
            Args:
                - x: original feature [B x 1024 x 1]
                - z: random noise [B x 1024 x1]
            Returns:
                - x_aug: augmented version of x [B x 1024]
        """
        x, z = x.unsqueeze(2), z.unsqueeze(2)
        data = torch.cat([x, z], dim=2)  # concat original and noise
        data = torch.permute(data, (1, 0, 2))  # seq len first
        # augmentations = []
        # for i in range(1024):
        #     o = self.mlp(data[i, :, :])
        #     augmentations.append(o)
        # augmentations = torch.cat(augmentations, dim=1)
        # print('Data', data.shape)
        augmentations = self.mlp(data)
        augmentations = torch.permute(augmentations, (1, 0, 2)).squeeze()
        # print('Augmentations', augmentations.shape)
        # exit()

        return augmentations


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":

    test_independent = True
    if test_independent:
        net = GeneratorIndependent().cuda()
        print("Number of parameters:", count_parameters(net))
        x = torch.randn(32, 1024).cuda()
        z = torch.randn(32, 1024).cuda()
        out = net(x, z)
        print("Out shape:", out.shape)

    test_mlp = False
    if test_mlp:
        net = GeneratorMLP().cuda()
        # print(net)
        print("Number of parameters:", count_parameters(net))
        x = torch.randn(32, 1024).cuda()
        z = torch.randn(32, 1024).cuda()
        out = net(x, z)
        print("Out shape:", out.shape)

    test_transformer = False
    if test_transformer:
        net = GeneratorTransformer(n_heads=4, emb_dim=64).cuda()
        # print("Net:", net)
        print("Number of parameters:", count_parameters(net))
        x = torch.randn(32, 1024).cuda()
        z = torch.randn(32, 1024).cuda()
        out = net(x, z)
        print("Out shape:", out.shape)
