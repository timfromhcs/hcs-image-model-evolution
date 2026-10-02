"""Compact Diffusion Transformer (DiT) architecture for student distillation."""


import torch
from torch import nn


class DiTBlock(nn.Module):
    """Standard Transformer block with multi-head self-attention and MLP."""

    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = self.norm1(x)
        attn_out, _ = self.attn(norm_x, norm_x, norm_x)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class CompactDiTStudent(nn.Module):
    """Compact DiT backbone for low-memory, fast-inference student models."""

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        patch_size: int = 2,
        hidden_size: int = 768,
        depth: int = 12,
        num_heads: int = 12,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.patch_size = patch_size
        self.hidden_size = hidden_size

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_size, kernel_size=patch_size, stride=patch_size
        )
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads) for _ in range(depth)
        ])

        self.norm_final = nn.LayerNorm(hidden_size)
        self.proj_out = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor | None = None) -> torch.Tensor:
        b, c, h, w = x.shape
        p = self.patch_size
        hp, wp = h // p, w // p

        # Embed patches
        patches = self.patch_embed(x) # (B, hidden, hp, wp)
        x_seq = patches.flatten(2).transpose(1, 2) # (B, hp*wp, hidden)

        if t is not None:
            t_emb = self.time_embed(t.view(-1, 1).to(x.dtype)).unsqueeze(1)
            x_seq = x_seq + t_emb

        for block in self.blocks:
            x_seq = block(x_seq)

        x_seq = self.norm_final(x_seq)
        out = self.proj_out(x_seq) # (B, hp*wp, p*p*out_channels)

        # Unpatchify
        out = out.view(b, hp, wp, p, p, self.out_channels)
        out = out.permute(0, 5, 1, 3, 2, 4).contiguous()
        out = out.view(b, self.out_channels, h, w)
        return out
