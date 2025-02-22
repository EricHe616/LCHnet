import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt, font_manager
import torchvision

# 加载训练集
train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]), download=True)

# 加载测试集
test_data_set = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]), download=True)

# 创建训练集数据加载器
train_data_load = DataLoader(train_data_set, batch_size=1000, shuffle=True, drop_last=True)
# 创建测试集数据加载器
test_data_load = DataLoader(test_data_set, batch_size=1000, shuffle=False, drop_last=True)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义 PatchEmbedding 类
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=768, img_size=32,):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 计算图像中 patch 的数量
        self.num_patches = (img_size // patch_size) ** 2
        # 卷积层来切分图像并将每个 patch 映射为一个嵌入向量
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 使用卷积操作来提取每个 patch 并映射为嵌入向量
        x = self.conv(x)  # 输出形状为 (batch_size, embed_dim, num_patches, num_patches)
        # 将空间维度展平为序列
        x = x.flatten(2)  # 展平后形状为 (batch_size, embed_dim, num_patches^2)
        # 转置成 (batch_size, num_patches^2, embed_dim)
        x = x.transpose(1, 2)
        return x  # 最终的形状为 (batch_size, num_patches^2, embed_dim)
class DeepseekV2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


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

class DeepseekConfig:
    hidden_size: int
    num_heads: int
    max_position_embeddings: int
    rope_theta: float
    attention_dropout: float

    q_lora_rank: int
    qk_rope_head_dim: int
    kv_lora_rank: int
    v_head_dim: int
    qk_nope_head_dim: int
    attention_bias: bool


class MLA(nn.Module):
    def __init__(self, config, ):
        super().__init__()

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_heads
        self.max_postion_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        # 对应 query 压缩的向量， 在 deepseek v3 中， hidden_size 7168
        # 但是压缩后的 kv d_c= 512，压缩比例 1/14
        # q 的压缩为 1536 压缩比例 1/4.7
        # rope 部分是 64
        self.q_lora_rank = config.q_lora_rank
        # 对应 query 和 key 进行 rope 的维度
        self.qk_rope_head_dim = config.qk_rope_head_dim
        # 对应 value 压缩的向量
        self.kv_lora_rank = config.kv_lora_rank
        # 对应 每一个 Head 的维度大小
        self.v_head_dim = config.v_head_dim
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.q_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim
        self.q_down_proj = nn.Linear(
            self.hidden_size,
            self.q_lora_rank,
            bias=config.attention_bias,
        )
        self.q_down_layernorm = DeepseekV2RMSNorm(self.q_lora_rank)
        self.q_up_proj = nn.Linear(
            self.q_lora_rank,
            self.num_heads * self.q_head_dim,
            # 最终还需要做切分（split），一部分是 nope，一部分需要应用 rope
            bias=False,
        )
        # 同理对于 kv 也是一样的
        self.kv_down_proj = nn.Linear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=config.attention_bias,
        )
        self.kv_down_layernorm = DeepseekV2RMSNorm(self.kv_lora_rank)
        self.kv_up_proj = nn.Linear(
            self.kv_lora_rank,
            self.num_heads * (
                    self.q_head_dim - self.qk_rope_head_dim + self.v_head_dim
            ),  # 其中 self.q_head_dim - self.qk_rope_head_dim 是 nope 部分
            bias=False,
        )
        # 对应公式 47 行
        self.o_proj = nn.Linear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            bias=config.attention_bias,
        )
        # 初始化 rope 的参数
        self.rotary_emb = DeepseekV2RotaryEmbedding(
            self.qk_rope_head_dim,
            self.max_postion_embeddings,
            self.rope_theta,
        )
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """
        MLA (Multi-head Linearized Attention) forward pass
        """

        bsz, q_len, _ = hidden_states.size()
        # 1. Query projection and split
        q = self.q_up_proj(
            self.q_down_layernorm(
                self.q_down_proj(hidden_states)
            )
        )
        q = q.view(bsz, q_len, self.num_heads, self.q_head_dim).transpose(1, 2)
        q_nope, q_pe = torch.split(
            q,
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1
        )
        # 2. Key/Value projection and split
        compressed_kv = self.kv_down_proj(hidden_states)
        compressed_kv, k_pe = torch.split(
            compressed_kv,
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1
        )
        k_pe = k_pe.view(bsz, q_len, 1, self.qk_rope_head_dim).transpose(1, 2)
        kv = (
            self.kv_up_proj(self.kv_down_layernorm(compressed_kv))
            .view(bsz, q_len, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
            .transpose(1, 2)
        )
        k_nope, value_states = torch.split(
            kv,
            [self.qk_nope_head_dim, self.v_head_dim],
            dim=-1
        )

        # 3. Apply RoPE to position-dependent parts
        kv_seq_len = value_states.shape[-2]

        cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
        q_pe, k_pe = apply_rotary_pos_emb(q_pe, k_pe, cos, sin, position_ids)

        # 最终 Q, k, V 的 Shape 都希望是 (batch_size, num_heads, seq_len, head_dim)
        # 其中 q / k 的 head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        # v 的 head_dim = self.v_head_dim

        # 4. Combine position-dependent and independent parts
        query_states = torch.empty(
            bsz, self.num_heads, q_len, self.q_head_dim,
            device=k_pe.device
        )
        query_states[:, :, :, :self.qk_nope_head_dim] = q_nope
        query_states[:, :, :, self.qk_nope_head_dim:] = q_pe

        key_states = torch.empty(
            bsz, self.num_heads, q_len, self.q_head_dim,
            device=k_pe.device
        )
        key_states[:, :, :, :self.qk_nope_head_dim] = k_nope
        key_states[:, :, :, self.qk_nope_head_dim:] = k_pe

        # 5. Compute attention scores
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = attn_weights / math.sqrt(self.q_head_dim)

        if attention_mask is not None:
            attn_weights = torch.masked_fill(
                attn_weights,
                attention_mask == 0,
                float("-inf"),
            )

        # 6. Softmax and dropout
        attn_weights = F.softmax(
            attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(
            attn_weights, p=self.attention_dropout, training=self.training)

        # 7. Compute attention output
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights
config = DeepseekConfig(
    hidden_size=768,  # Keep consistent with PatchEmbedding
    num_heads=1,
    max_position_embeddings=1024,
    rope_theta=1280,
    attention_dropout=0,
    q_lora_rank=1536,
    qk_rope_head_dim=64,
    kv_lora_rank=512,
    v_head_dim=128,
    qk_nope_head_dim=128,
    attention_bias=False,
)
# 定义整体分类模型
class LCHnet(nn.Module):
    def __init__(self,config, in_channels, out_channels, num_classes=10):
        super(LCHnet, self).__init__()
        # 全连接层，输入维度根据 PatchEmbedding 输出计算
        self.fc1 = nn.Linear((32 // 4) ** 2 * 768, 768)  # 10 是输出类别数
        self.embed = PatchEmbedding(in_channels, patch_size=4, embed_dim=768)
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()
        self.lch = MLA(config)


    def forward(self, x):
        x = self.embed(x)
        x, _ = self.lch(x)
        # 展平 x，去除除了 batch_size 之外的所有维度
        x = x.flatten(1)  # 仅展平非batch维度
        # 通过第一个全连接层
        x= self.fc1(x)
        # 应用激活函数
        x = self.relu(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x

