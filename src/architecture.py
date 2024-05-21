import math

import torch


class PositionalEncoding(torch.nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        pe = torch.zeros(max_len, d_model)
        pe[:, : d_model // 2] = torch.sin(position * div_term)
        pe[:, d_model // 2 :] = torch.cos(position * div_term)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0)]
        return x


class MultiheadSelfAttention(torch.nn.Module):
    def __init__(self, embed_dim: int, num_heads: int = 8):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_heads = num_heads

        self.query = torch.nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
        )
        self.key = torch.nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
        )
        self.value = torch.nn.Linear(
            in_features=embed_dim,
            out_features=embed_dim,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.query(x).view(x.shape[0], self.num_heads, -1).transpose(0, 1)
        k = self.key(x).view(x.shape[0], self.num_heads, -1).permute(1, 2, 0)
        v = self.value(x).view(x.shape[0], self.num_heads, -1).transpose(0, 1)
        qk = torch.softmax(
            torch.matmul(q, k) / (self.embed_dim / self.num_heads) ** 0.5,
            dim=-1,
        )
        qkv = torch.matmul(qk, v).transpose(0, 1).reshape(x.shape[0], -1)
        return qkv


class Block(torch.nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, eps: float = 1e-6):
        super().__init__()

        self.ln1 = torch.nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.attn = MultiheadSelfAttention(embed_dim=d_model, num_heads=num_heads)
        self.ln2 = torch.nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.linear1 = torch.nn.Linear(in_features=d_model, out_features=d_model * 4)
        self.linear2 = torch.nn.Linear(in_features=d_model * 4, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln1 = self.ln1(x)
        attn = self.attn(ln1)
        ln2 = self.ln2(x + attn)
        mlp = self.linear2(torch.relu(self.linear1(ln2)))
        return mlp + x + attn


class Head(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        self.ln = torch.nn.LayerNorm(normalized_shape=d_model, eps=eps)
        self.linear1 = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.linear2 = torch.nn.Linear(in_features=d_model, out_features=d_model)
        self.tanh_layer = torch.nn.Linear(in_features=d_model * 2, out_features=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln = self.ln(x)
        mlp = torch.exp(self.linear2(torch.nn.functional.elu(self.linear1(ln))))
        res = torch.cat(
            [
                ln.sum(dim=0) / ln.shape[0],
                (mlp * ln).sum(dim=0) / mlp.sum(dim=0),
            ]
        )
        res = torch.tanh(self.tanh_layer(res))
        res /= (res**2).sum() ** 0.5
        res /= (res**2).sum() ** 0.5
        return res


class MUSE(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        d_model: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.d_model = d_model
        self.num_heads = num_heads
        self.eps = eps

        self.embedding = torch.nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
        )
        self.linear = torch.nn.Linear(
            in_features=embedding_dim,
            out_features=d_model,
        )
        self.pe = PositionalEncoding(
            d_model=d_model,
            max_len=512,  # TODO: remove hardcode
        )
        self.block0 = Block(d_model=d_model)
        self.block1 = Block(d_model=d_model)
        self.block2 = Block(d_model=d_model)
        self.block3 = Block(d_model=d_model)
        self.block4 = Block(d_model=d_model)
        self.block5 = Block(d_model=d_model)
        self.head = Head(d_model=d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedding(x)
        x = self.linear(x)
        x = self.pe(x)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.head(x)
        return x
