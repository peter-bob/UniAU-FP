import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFeatureMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_v1, x_v2):

        B, _, D = x_v1.shape
        _, M, _ = x_v2.shape

        Q = self.q_proj(x_v1)   # [B, 1, D]
        K = self.k_proj(x_v2)   # [B, M, D]
        V = self.v_proj(x_v2)   # [B, M, D]

        Q = Q.view(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, M, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, V)  # [B, H, 1, head_dim]

        out = out.transpose(1, 2).contiguous().view(B, 1, D)
        out = self.out_proj(out)

        return out


class CrossFeatureEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()

        self.cross_attn = CrossFeatureMultiHeadAttention(
            embed_dim, num_heads, dropout
        )
        self.norm1 = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x_v1, x_v2):

        attn_out = self.cross_attn(x_v1, x_v2)
        x_v1 = self.norm1(x_v1 + attn_out)

        ffn_out = self.ffn(x_v1)
        x_v1 = self.norm2(x_v1 + ffn_out)

        return x_v1


class CrossFeatureEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, dropout=0.1):
        super().__init__()

        self.layers = nn.ModuleList([
            CrossFeatureEncoderLayer(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x_v1, x_v2):
        for layer in self.layers:
            x_v1 = layer(x_v1, x_v2)
        return x_v1


if __name__ == "__main__":
    x_v1 = torch.randn(2, 1, 768)
    x_v2 = torch.randn(2, 6, 768)
    model = CrossFeatureEncoder(768, 8, 4)

    out = model(x_v1, x_v2)
    print(out.shape)