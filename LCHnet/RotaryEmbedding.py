import torch
from torch import nn

class DeepseekV2RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=1000, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings,
            device=self.inv_freq.device,
            dtype=torch.get_default_dtype(),
        )
        self.max_seq_len_cached = None

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(
            self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype
        )

        freqs = torch.outer(t, self.inv_freq.to(t.device))
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.max_seq_len_cached is None or seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)

        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]  # 取前半部分
    x2 = x[..., x.shape[-1] // 2:]   # 取后半部分
    return torch.cat((-x2, x1), dim=-1)  # 交换并拼接

def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)  # 提取对应的cos值并扩展维度
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)  # 提取对应的sin值并扩展维度

    # 对q进行处理，分成两部分后进行旋转
    b, h, s, d = q.shape  # 获取q的形状
    q = q.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)  # 分割并重排列

    # 对k进行类似的处理
    b, h, s, d = k.shape
    k = k.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)  # 分割并重排列

    # 应用旋转位置编码
    q_embed = (q * cos) + (rotate_half(q) * sin)  # 查询向量的旋转位置编码
    k_embed = (k * cos) + (rotate_half(k) * sin)  # 键向量的旋转位置编码

    return q_embed, k_embed