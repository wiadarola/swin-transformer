import functools
import math

import torch
from einops import rearrange
from torch import nn


class SwinTransformer(nn.Module):
    def __init__(
        self,
        patch_size: int,
        embed_dim: int,
        depths: tuple[int, int, int, int],
        window_size: int,
        num_heads: list[int],
        dropout: float,
        num_classes: int,
    ):
        """Creates a SwinTransformer class

        :param patch_size: Patch partition size
        :param embed_dim: Projection dimension (denoted as C in the paper)
        :param depths: Number of blocks in each of the four stages
        :param window_size: Attention patch size
        :param num_heads: Number of attention heads
        :param dropout: Dropout probability for the attention MLP
        :param num_classes: Output dimension of the model
        """
        super().__init__()
        Stage = functools.partial(SwinTransformerStage, M=window_size, p_drop=dropout)
        self.stage1 = Stage(3, embed_dim, patch_size, depths[0], num_heads[0])
        self.stage2 = Stage(embed_dim, 2 * embed_dim, 2, depths[1], num_heads[1])
        self.stage3 = Stage(embed_dim * 2, embed_dim * 4, 2, depths[2], num_heads[2])
        self.stage4 = Stage(embed_dim * 4, embed_dim * 8, 2, depths[3], num_heads[3])
        self.ln = nn.LayerNorm(embed_dim * 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(embed_dim * 8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)  # 'linear embedding' -> blocks
        x = self.stage2(x)  # 'patch merging' -> blocks
        x = self.stage3(x)
        x = self.stage4(x)

        x = rearrange(x, "B C H W -> B H W C")
        x = self.ln(x)
        x = rearrange(x, "B H W C -> B C H W")
        x = self.pool(x).flatten(1)
        x = self.head(x)

        return x


class SwinTransformerStage(nn.Module):
    def __init__(
        self,
        in_embed: int,
        out_embed: int,
        d_window: int,
        depth: int,
        nH: int,
        M: int,
        p_drop: float,
    ):
        super().__init__()
        assert depth % 2 == 0, "Depth must be a multiple of two"
        self.conv = nn.Conv2d(in_embed, out_embed, d_window, d_window)
        BlockPair = functools.partial(SwinTransformerBlockPair, out_embed, M, nH, p_drop)
        self.blocks = nn.Sequential(*[BlockPair() for _ in range(depth // 2)])

    def forward(self, x: torch.Tensor):
        x = self.conv(x)
        x = self.blocks(x)
        return x


class SwinTransformerBlockPair(nn.Module):
    def __init__(self, d_embed: int, M: int, nH: int, p_drop: float):
        super().__init__()
        self.block1 = SwinTransformerBlock(d_embed, WMSA(d_embed, M, nH), p_drop)
        self.block2 = SwinTransformerBlock(d_embed, SWMSA(d_embed, M, nH), p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        return x


class SwinTransformerBlock(nn.Module):
    def __init__(self, d_embed: int, msa_fn: nn.Module, p_drop: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_embed)
        self.msa = msa_fn
        self.ln2 = nn.LayerNorm(d_embed)
        self.mlp = MLP(d_embed, p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip1 = x
        x = rearrange(x, "B C H W -> B H W C")
        x = self.ln1(x)
        x = rearrange(x, "B H W C -> B C H W")
        x = self.msa(x)
        x = x + skip1

        skip2 = x
        x = rearrange(x, "B C H W -> B H W C")
        x = self.ln2(x)
        x = self.mlp(x)
        x = rearrange(x, "B H W C -> B C H W")
        x = x + skip2

        return x


class WMSA(nn.Module):
    def __init__(self, d_embed: int, M: int, nH: int):
        super().__init__()
        self.M = M
        self.nH = nH
        self.partition = nn.Unfold(kernel_size=M, stride=M)
        self.qkv = nn.Linear(d_embed, 3 * d_embed)
        self.d = math.sqrt(d_embed**2 / nH)
        self.bias = RelativePositionBias(M, nH)
        self.out = nn.Linear(d_embed, d_embed)
        self.cached_input_shape = torch.Size()
        self.cached_mask = torch.empty(0)

    def get_mask(self, shape: torch.Size) -> torch.Tensor:
        """Returns an identity attention mask"""
        if self.cached_input_shape == shape:
            return self.cached_mask

        mask = torch.zeros(self.M**2)
        self.cached_mask = mask
        self.cached_input_shape = shape
        self.register_buffer("mask", mask)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, c, h, w = x.shape
        assert h % self.M == 0 and w % self.M == 0, "Patches must be divisible by window size"
        windows = self.partition(x)
        windows = rearrange(windows, "B (C MM) nW -> B nW MM C", C=c)
        QKV = self.qkv(windows)
        Q, K, V = rearrange(QKV, "B nW MM (nH T H) -> T B nW nH MM H", T=3, nH=self.nH).unbind()
        mask = self.get_mask(x.shape).to(x.device)  # nW 1 MM MM
        attn = (self.bias((Q @ K.transpose(-1, -2)).div(self.d)) + mask).softmax(-1) @ V
        stack = rearrange(attn, "B nW nH MM H -> B nW MM (nH H)")
        x = self.out(stack)
        x = rearrange(x, "B nW MM C -> B C (nW MM)")
        x = rearrange(x, "B C (H W) -> B C H W", H=h)
        return x


class SWMSA(WMSA):
    """Shifted Window Self-Attention"""

    def __init__(self, d_embed: int, M: int, nH: int):
        super().__init__(d_embed, M, nH)

    def get_mask(self, shape: torch.Size) -> torch.Tensor:
        """Returns a mask to disable shifted pixels that shouldn't attend to each other"""
        if self.cached_input_shape == shape:
            return self.cached_mask

        *_, H, W = shape
        img_mask = torch.zeros((H, W))  # C H W
        slices = (slice(0, -self.M), slice(-self.M, -self.M // 2), slice(-self.M // 2, None))

        for i, h in enumerate(slices):
            for j, w in enumerate(slices):
                img_mask[h, w] = i * len(slices) + j

        mask_windows = self.partition(img_mask.unsqueeze(0)).T
        mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        mask = mask.masked_fill(mask != 0.0, -torch.inf).unsqueeze(1)

        self.cached_mask = mask
        self.cached_input_shape = shape
        self.register_buffer("mask", mask)
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.roll(x, (self.M // 2, self.M // 2), dims=(2, 3))
        x = super().forward(x)
        x = torch.roll(x, (-self.M // 2, -self.M // 2), dims=(2, 3))
        return x


class RelativePositionBias(nn.Module):
    def __init__(self, M: int, nH: int):
        super().__init__()
        self.MM = M * M
        self.nH = nH
        self.bias_table = nn.Parameter(torch.zeros(((2 * M - 1) ** 2, nH)))
        nn.init.trunc_normal_(self.bias_table, std=0.02)

        coords = torch.cartesian_prod(torch.arange(M), torch.arange(M))  # MM 2
        relative_coords = coords[:, None, :] - coords[None, :, :]  # MM MM 2
        relative_coords += M - 1  # shift to >= 0
        relative_coords[..., 0] *= 2 * M - 1

        self.index: torch.Tensor
        self.register_buffer("index", relative_coords.sum(-1).reshape(-1))  # MMMM

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        flat = torch.index_select(self.bias_table, 0, self.index)  # MMMM nH
        bias = flat.unflatten(0, (self.MM, self.MM))
        return x + bias


class MLP(nn.Module):
    def __init__(self, d_embed: int, p_drop: float):
        super().__init__()
        self.fc1 = nn.Linear(d_embed, d_embed)
        self.drop1 = nn.Dropout(p_drop)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(d_embed, d_embed)
        self.drop2 = nn.Dropout(p_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.drop1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x
