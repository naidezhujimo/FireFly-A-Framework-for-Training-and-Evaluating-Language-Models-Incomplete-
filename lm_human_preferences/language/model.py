import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List

import os
import sys
# 获取当前脚本的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取 FireFly 的路径
firefly_path = os.path.abspath(os.path.join(current_directory, '..', '..'))
sys.path.insert(0, firefly_path)  # 将 FireFly 路径添加到 sys.path
from lm_human_preferences.utils import core as utils

# 定义超参数类
class HParams:
    def __init__(self):
        self.n_vocab = 0  # 词汇表大小
        self.n_ctx = 512  # 上下文长度
        self.n_embd = 768  # 嵌入维度
        self.n_head = 12  # 多头注意力头数
        self.n_layer = 12  # Transformer 层数

        # Dropout 概率
        self.embd_pdrop = 0.1  # 嵌入层 dropout
        self.attn_pdrop = 0.1  # 注意力 dropout
        self.resid_pdrop = 0.1  # 残差连接 dropout
        self.head_pdrop = 0.1  # 输出头 dropout

    def override_from_dict(self, d):
        for key, value in d.items():
            setattr(self, key, value)

# GELU 激活函数
def gelu(x):
    return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))

# 归一化层
class LayerNorm(nn.Module):
    """自定义 LayerNorm,支持可学习的缩放和偏移"""
    def __init__(self, n_state, eps=1e-5):
        super().__init__()
        self.g = nn.Parameter(torch.ones(n_state)) # 缩放因子
        self.b = nn.Parameter(torch.zeros(n_state)) # 偏移因子
        self.eps = eps # 稳定性参数
    
    def forward(self, x):
        u = x.mean(-1, keepdim=True)  # 计算均值
        s = (x - u).pow(2).mean(-1, keepdim=True) # 计算方差
        x = (x - u) / torch.sqrt(s + self.eps) # 归一化
        return x * self.g + self.b  # 缩放和偏移

# 多头注意力机制
class MultiHeadAttention(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.n_state = hparams.n_embd
        self.n_head = hparams.n_head
        self.c_attn = nn.Linear(self.n_state, self.n_state * 3)  # 线性变换生成 Q, K, V
        self.c_proj = nn.Linear(self.n_state, self.n_state)  # 输出线性变换
        self.attn_dropout = nn.Dropout(hparams.attn_pdrop)
        self.resid_dropout = nn.Dropout(hparams.resid_pdrop)

    def forward(self, x, past=None, mask=None, do_dropout=True):
        batch, seq, _ = x.shape
        q, k, v = self.c_attn(x).split(self.n_state, dim=-1)  # 分割 Q, K, V
        q = q.view(batch, seq, self.n_head, -1).transpose(1, 2)
        k = k.view(batch, seq, self.n_head, -1).transpose(1, 2)
        v = v.view(batch, seq, self.n_head, -1).transpose(1, 2)

        present = None
        if past is not None:
            # 确保 past 是一个包含两个张量的元组
            if isinstance(past, tuple) and len(past) == 2:
                pk, pv = past
            else:
                raise ValueError("past should be a tuple of key and value tensors")
            k = torch.cat([pk, k], dim=-2)
            v = torch.cat([pv, v], dim=-2)
            present = torch.stack([k, v], dim=1)  # 保存当前 K, V 供未来使用
        else:
            present = torch.stack([k, v], dim=1)  # 保存当前 K, V 供未来使用

        # 计算注意力分数
        attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.n_state // self.n_head))
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn) if do_dropout else attn

        # 加权求和
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch, seq, -1)
        out = self.c_proj(out)
        out = self.resid_dropout(out) if do_dropout else out
        return out, present

# Transformer块
class TransformerBlock(nn.Module):
    def __init__(self, hparams):
        self.hparams = hparams
        self.ln_1 = LayerNorm(hparams.n_embd)
        self.attn = MultiHeadAttention(hparams)
        self.ln_2 = LayerNorm(hparams.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(hparams.n_embd, hparams.n_embd * 4),
            nn.GELU(),
            nn.Linear(hparams.n_embd * 4, hparams.n_embd),
            nn.Dropout(hparams.resid_pdrop)
        )

    def forward(self, x, past=None, mask=None, do_dropout=True):
        attn_out, present = self.attn(self.ln_1(x), past=past, mask=mask, do_dropout=do_dropout)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x, present

# 完整Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, hparams, scalar_heads=None):
        super().__init__()
        self.hparams = hparams
        self.scalar_heads = scalar_heads if scalar_heads is not None else []

        # 词嵌入和位置嵌入
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embd)
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embd)

        # Transformer层
        self.blocks = nn.ModuleList([TransformerBlock(hparams) for _ in range(hparams.n_layer)])

        # 输出层
        self.ln_f = LayerNorm(hparams.n_embd)
        self.heads = nn.ModuleDict({
            head: nn.Linear(hparams.n_embd, 1) for head in self.scalar_heads
        })

    def forward(self, X, past=None, mask=None, do_dropout=True):
        batch, seq = X.shape
        device = X.device

        # 位置编码
        positions = torch.arange(seq, dtype=torch.long, device=device).unsqueeze(0)
        h = self.wte(X) + self.wpe(positions)

        # Transformer层
        presents = []
        for block in self.blocks:
            h, present = block(h, past=past, mask=mask, do_dropout=do_dropout)
            presents.append(present)

        # 最终LayerNorm
        h = self.ln_f(h)

        # 输出头
        results = {'present': torch.stack(presents, dim=1), 'h': h}
        for head_name, head in self.heads.items():
            results[head_name] = head(h)
        return results

# positions_for 函数
def positions_for(batch: int, sequence: int, past_length: int, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Generate position embeddings for the input"""
    if mask is not None:
        positions = torch.cumsum(mask, dim=1).to(torch.long) - 1
    else:
        positions = torch.arange(past_length, past_length + sequence, dtype=torch.long)
        positions = positions.unsqueeze(0).expand(batch, sequence)
    return positions

# Block 类（作为 TransformerBlock 的翻版）
class Block(nn.Module):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.ln_1 = LayerNorm(hparams.n_embd)
        self.attn = MultiHeadAttention(hparams)
        self.ln_2 = LayerNorm(hparams.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(hparams.n_embd, hparams.n_embd * 4),
            nn.GELU(),
            nn.Linear(hparams.n_embd * 4, hparams.n_embd),
            nn.Dropout(hparams.resid_pdrop)
        )

    def forward(self, x, past=None, mask=None, do_dropout=True):
        attn_out, present = self.attn(self.ln_1(x), past=past, mask=mask, do_dropout=do_dropout)
        x = x + attn_out
        mlp_out = self.mlp(self.ln_2(x))
        x = x + mlp_out
        return x, present

def past_shape(*, hparams, batch_size=None, sequence=None):
    """计算 past 状态的形状

    Args:
        hparams (HParams): 超参数类，包含模型的配置如 n_layer, n_head, n_embd
        batch_size (int, optional): 批次大小。默认为 None。
        sequence (int, optional): 序列长度。默认为 None。

    Returns:
        list: 形状列表
    """
    n_head = hparams.n_head
    n_embd = hparams.n_embd
    head_size = utils.exact_div(n_embd, n_head)
    shape = [
        batch_size if batch_size is not None else -1,  # 批次大小（动态）
        hparams.n_layer,  # 层数
        2,  # 键和值
        n_head,  # 头数
        sequence if sequence is not None else -1,  # 序列长度（动态）
        head_size  # 每个头的大小
    ]
    return shape

class Model(nn.Module):
    def __init__(self, hparams, scalar_heads=[], scope=None):
        super().__init__()
        self.hparams = hparams
        self.scalar_heads = scalar_heads
        self.scope = scope

        # Embedding layers
        self.wpe = nn.Embedding(hparams.n_ctx, hparams.n_embd)
        self.wte = nn.Embedding(hparams.n_vocab, hparams.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            Block(hparams) for i in range(hparams.n_layer)
        ])

        # Final layer normalization
        self.ln_f = nn.LayerNorm(hparams.n_embd)

        # Scalar heads
        self.heads = nn.ModuleDict({
            head_name: nn.Linear(hparams.n_embd, 1) for head_name in scalar_heads
        })

    def forward(self, X, Y=None, past=None, past_tokens=None, mask=None, padding_token: Optional[int] = None, do_dropout=False):
        if mask is not None:
            mask = torch.tensor(mask, dtype=torch.bool)
            assert mask.dtype == torch.bool
        if padding_token is not None:
            assert mask is None, 'At most one of mask and padding_token should be set'
            mask = X != padding_token
            X = torch.where(mask, X, torch.zeros_like(X))
            if past is not None:
                assert past_tokens is not None, 'padding_token requires past_tokens'
                mask = torch.cat([past_tokens != padding_token, mask], dim=1)

        results = {}
        batch, sequence = X.shape

        # Positional embeddings
        if past is not None:
            past_length = past.shape[-2]
        else:
            past_length = 0
        positions = positions_for(batch=batch, sequence=sequence, past_length=past_length, mask=mask)
        h = self.wte(X) + self.wpe(positions)

        # Transformer blocks
        presents = []
        pasts = torch.unbind(past, dim=1) if past is not None else [None] * self.hparams.n_layer
        for layer, (past_layer, block) in enumerate(zip(pasts, self.blocks)):
            h, present = block(h, past=past_layer, mask=mask, do_dropout=do_dropout)
            presents.append(present)
        results['present'] = torch.stack(presents, dim=1)
        h = self.ln_f(h)

        # Handle mask and padding
        if mask is not None:
            # Generate indices
            indices = torch.arange(sequence, device=X.device).unsqueeze(0)
            indices = indices.expand(batch, sequence)

            # Create present_indices
            present_indices = torch.where(mask[:, past_length:], indices, -1)
            present_indices = present_indices.max(dim=1).values  # 取每个样本的最大索引

            # Ensure present_indices has the correct shape
            present_indices = present_indices.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h.shape[-1])
            h = torch.gather(h, 1, present_indices)
        else:
            # 如果没有 mask，直接取最后一个 token
            h = h[:, -1, :].unsqueeze(1)

        results['h'] = h

        # Language model loss
        h_flat = h.reshape(-1, self.hparams.n_embd)
        flat_lm_logits = F.linear(h_flat, self.wte.weight)

        labels = torch.cat([X[:, 1:], X[:, :1]], dim=1)
        flat_labels = labels.reshape(-1)

        flat_losses = F.cross_entropy(flat_lm_logits, flat_labels, reduction='none')
        lm_losses = flat_losses.reshape(batch, sequence)
        lm_logits = flat_lm_logits.reshape(batch, sequence, -1)

        relevant_losses = lm_losses[:, :sequence]
        results['lm_all_losses'] = relevant_losses
        results['lm_logits'] = lm_logits
        results['lm_losses'] = torch.mean(relevant_losses, dim=-1)

        # Scalar heads
        for head_name in self.scalar_heads:
            dropped_h = F.dropout(h, p=self.hparams.head_pdrop, training=do_dropout)
            res = self.heads[head_name](dropped_h)
            results[head_name] = res.float()
            results[f"{head_name}_regularizer"] = torch.tensor(0.0, dtype=torch.float32)  # Placeholder for regularization loss

        return results

    def get_params(self):
        return list(self.parameters())
    
# 测试脚本
import numpy as np
# 初始化超参数
hparams = HParams()
hparams.override_from_dict({
    'n_vocab': 100,
    'n_ctx': 512,
    'n_embd': 768,
    'n_head': 12,
    'n_layer': 2
})

# 定义模型
policy = Model(hparams=hparams, scalar_heads=["head1", "head2"])

# 输入数据
batch_size = 1
sequence_length = 5
X = torch.randint(0, hparams.n_vocab, (batch_size, sequence_length))
mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)

# 前向传播
logits = policy(X=X, mask=mask)['lm_logits']
print(logits.shape)  # 输出形状应为 (batch_size, sequence_length, n_vocab)