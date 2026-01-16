from typing import Tuple

import torch
import torch.nn.functional as Fn
from einops.layers.torch import Rearrange
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        use_flash_attn=True,
        return_last_block_attn: bool = False,
    ):
        super().__init__()

        inner_dim = dim_head * heads
        self.use_flash_attn = use_flash_attn
        self.p_dropout = dropout
        self.scale = dim_head**-0.5
        no_project = heads == 1 and dim_head == dim
        self.return_last_block_attn = return_last_block_attn

        self.ln1 = nn.LayerNorm(dim)
        self.wq = nn.Linear(dim, inner_dim, bias=False)
        self.wk = nn.Linear(dim, inner_dim, bias=False)
        self.wv = nn.Linear(dim, inner_dim, bias=False)
        self.split_emb_dim = Rearrange("b n (h d) -> b h n d", h=heads)
        self.drop = nn.Dropout(p=self.p_dropout)
        self.merge_emb_dim = Rearrange("b h n d -> b n (h d)", h=heads)
        self.project = (
            nn.Sequential(
                nn.Linear(inner_dim, dim),
                nn.Dropout(p=dropout),
            )
            if not no_project
            else nn.Identity()
        )

    def flash_attn(self, q, k, v):
        with sdpa_kernel(
            [
                SDPBackend.MATH,
                SDPBackend.EFFICIENT_ATTENTION,
                SDPBackend.FLASH_ATTENTION,
                SDPBackend.CUDNN_ATTENTION,
            ]
        ):
            out = Fn.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.p_dropout,
                is_causal=False,
                scale=self.scale,
            )

        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, E)
        :return: (B, N, E)
        """
        B, N, E = x.shape

        x = self.ln1(x)  # (B, N, E)

        # I = heads * dim_head, inner dimension before splitting for heads
        q, k, v = self.wq(x), self.wk(x), self.wv(x)  # q, k, v: (B, N, I)

        q, k, v = map(
            lambda t: self.split_emb_dim(t), [q, k, v]
        )  # q, k, v: (B, H, N, I / H)

        last_block_attn = None
        if self.use_flash_attn and not self.return_last_block_attn:
            out = self.flash_attn(q, k, v)  # (B, H, N, I / H)
        else:
            logits = torch.matmul(q, k.transpose(-1, -2)) * self.scale  # (B, H, N, N)
            attn = Fn.softmax(logits, dim=-1)  # (B, H, N, N)

            if self.return_last_block_attn:
                last_block_attn = attn  # (B, H, N, N)

            attn = self.drop(attn)  # (B, H, N, N)
            out = torch.matmul(attn, v)  # (B, H, N, I / H)

        out = self.merge_emb_dim(out)  # (B, N, I)
        out = self.project(out)  # (B, N, E)
        return out, last_block_attn


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.l1 = nn.Linear(dim, hidden_dim)
        self.gelu1 = nn.GELU()
        self.drop1 = nn.Dropout(p=dropout)
        self.l2 = nn.Linear(hidden_dim, dim)
        self.drop2 = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, N, E)
        :return: (B, N, E)
        """
        # H = hidden dim
        x = self.ln1(x)  # (B, N, E)
        x = self.l1(x)  # (B, N, H)
        x = self.gelu1(x)  # (B, N, H)
        x = self.drop1(x)  # (B, N, H)
        x = self.l2(x)  # (B, N, E)
        x = self.drop2(x)  # (B, N, E)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        return_last_block_attn: bool = False,
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth - 1):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(
                            dim=dim,
                            heads=heads,
                            dim_head=dim_head,
                            dropout=dropout,
                            use_flash_attn=use_flash_attn,
                            return_last_block_attn=False,  # we only want the [CLS] attn from the last block
                        ),
                        FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                    ]
                )
            )
        self.layers.append(
            nn.ModuleList(
                [
                    Attention(
                        dim=dim,
                        heads=heads,
                        dim_head=dim_head,
                        dropout=dropout,
                        use_flash_attn=use_flash_attn,
                        return_last_block_attn=return_last_block_attn,
                    ),
                    FeedForward(dim=dim, hidden_dim=mlp_dim, dropout=dropout),
                ]
            )
        )
        self.ln1 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, E)
        :return: (B, N, E)
        """
        last_block_attn = None
        for attn, ff in self.layers:
            x_attn, last_block_attn = attn(
                x
            )  # x_attn: (B, N, E), last_block_attn: (B, H, N, N)
            x = x_attn + x  # (B, N, E)
            x = ff(x) + x  # (B, N, E)
        x = self.ln1(x)  # (B, N, E)
        return x, last_block_attn


class PatchEmbedding(nn.Module):
    def __init__(self, patch_size: Tuple[int, int], channels: int, dim: int):
        """
        patch_size: (PH, PW) of each patch in pixels.
        dim: Embedding dimension E.
        """
        super().__init__()
        ph, pw = patch_size[0], patch_size[1]
        self.patch_dim = ph * pw * channels

        self.patchify = Rearrange(
            "b f c (h ph) (w pw) -> b f (h w) (ph pw c)", ph=ph, pw=pw
        )
        self.ln1 = nn.LayerNorm(self.patch_dim)
        self.l1 = nn.Linear(self.patch_dim, dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, F, C, H, W)
        :return: (B, H / P1, W / P2, E)
        """
        # N = H / PH * W / PW, the number of patches
        x = self.patchify(x)  # (B, F, N, PH * PW * C)
        x = self.ln1(x)  # (B, F, N, PH * PW * C)
        x = self.l1(x)  # (B, F, N, E)
        x = self.ln2(x)  # (B, F, N, E)

        return x  # (B, F, N, E)


class ViViT(nn.Module):
    """
    Video Vision Transformer without frame patching.
    """

    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        frames: int,
        channels: int,
        n_classes: int,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        return_cls_attn: bool = False,
    ):
        """
        image_size: (H, W) of the image in pixels.
        patch_size: (PH, PW) of each patch in pixels.
        channels: Image channels C.
        dim: Embedding dimension E.
        """
        super().__init__()

        ih, iw = image_size
        ph, pw = patch_size
        n_patches = ih // ph * iw // pw

        self.patch_emb = PatchEmbedding(patch_size, channels, dim)
        self.pos_enc = nn.Parameter(torch.randn(1, frames, n_patches, dim))
        self.flatten_frames = Rearrange("b f n e -> b (f n) e")
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.transformer = Transformer(
            dim=dim,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            mlp_dim=mlp_dim,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            return_last_block_attn=return_cls_attn,
        )

        self.l1 = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, F, C, H, W)
        :return: [(B, num_classes), (B, Heads, F * N)]
        """
        B, F, C, H, W = x.shape

        # N = H / PH * W / PW * C, the number of patches
        x = self.patch_emb(x)  # (B, F, N, E)
        x = x + self.pos_enc  # (B, F, N, E)
        x = self.flatten_frames(x)  # (B, F * N, E)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, F * N + 1, E)
        x, last_block_attn = self.transformer(
            x
        )  # x: (B, F * N + 1, E), last_block_attn: (B, Heads, F * N + 1, F * N + 1)

        # fetching [CLS] token
        x = x[:, 0]  # (B, E)
        x = self.l1(x)  # (B, num_classes)

        cls_attn = last_block_attn[:, :, 0, :]  # (B, Heads, F * N + 1)
        cls_attn = cls_attn[:, :, 1:]  # (B, Heads, F * N)

        return x, cls_attn


class HierarchicalViT(nn.Module):
    def __init__(
        self,
        image_size: Tuple[int, int],
        patch_size: Tuple[int, int],
        frames: int,
        channels: int,
        n_classes: int,
        dim: int,
        depth1: int,
        depth2: int,
        heads1: int,
        heads2: int,
        dim_head1: int,
        dim_head2: int,
        mlp_dim1: int,
        mlp_dim2: int,
        dropout: float = 0.0,
        use_flash_attn: bool = True,
        return_cls_attn: bool = False,
    ):
        super().__init__()

        ih, iw = image_size
        ph, pw = patch_size
        n_patches = ih // ph * iw // pw

        self.patch_emb = PatchEmbedding(patch_size, channels, dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_enc = nn.Parameter(torch.randn(1, frames, n_patches + 1, dim))
        self.flatten_frames = Rearrange("b f n e -> b (f n) e")

        self.t1 = Transformer(
            dim=dim,
            depth=depth1,
            heads=heads1,
            dim_head=dim_head1,
            mlp_dim=mlp_dim1,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            return_last_block_attn=return_cls_attn,
        )

        self.unflatten_attn = Rearrange(
            "b heads (fq nq) (fk nk) -> b heads fq nq fk nk", fq=frames, fk=frames
        )
        self.unflatten_t1_out = Rearrange("b (f n) e -> b f n e", f=frames)

        self.t2 = Transformer(
            dim=dim,
            depth=depth2,
            heads=heads2,
            dim_head=dim_head2,
            mlp_dim=mlp_dim2,
            dropout=dropout,
            use_flash_attn=use_flash_attn,
            return_last_block_attn=False,  # [CLS] vs patch attention only exists in first transformer's attention layers
        )

        self.l1 = nn.Linear(dim, n_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, F, C, H, W)
        :return: [(B, num_classes), (B, Heads, F * N)]
        """
        B, F, C, H, W = x.shape

        # N = H / PH * W / PW * C, the number of patches
        x = self.patch_emb(x)  # (B, F, N, E)
        cls_tokens = self.cls_token.expand(B, F, 1, -1)
        x = torch.cat((cls_tokens, x), dim=2)  # (B, F, N + 1, E)
        x = x + self.pos_enc  # (B, F, N + 1, E)
        x = self.flatten_frames(x)  # (B, F * (N + 1), E)

        x, last_block_attn = self.t1(
            x
        )  # x: (B, F * (N + 1), E), last_block_attn: (B, Heads, F * (N + 1), F * (N + 1))

        cls_attn = self.unflatten_attn(
            last_block_attn
        )  # (B, Heads, F, N + 1, F, N + 1)
        cls_attn = cls_attn[:, :, :, 0, :, :]  # (B, Heads, F, F, N + 1)
        cls_attn = cls_attn[:, :, :, :, 1:]  # (B, Heads, F, F, N)
        cls_attn = cls_attn.diagonal(dim1=2, dim2=3)  # (B, Heads, N, F)
        cls_attn = cls_attn.permute(0, 1, 3, 2)  # (B, Heads, F, N)

        x = self.unflatten_t1_out(x)  # (B, F, N + 1, E)
        x = x[:, :, 0, :]  # (B, F, E)

        x, _ = self.t2(x)  # (B, F, E)
        x = x.mean(dim=1)  # (B, E)

        x = self.l1(x)
        return x, cls_attn
