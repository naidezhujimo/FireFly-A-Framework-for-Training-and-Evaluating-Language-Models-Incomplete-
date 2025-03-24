import numpy as np
import torch
import torch.nn.functional as F

import os
import sys
# 获取当前脚本的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取 FireFly 的路径
firefly_path = os.path.abspath(os.path.join(current_directory, '..', '..'))
sys.path.insert(0, firefly_path)  # 将 FireFly 路径添加到 sys.path
from lm_human_preferences.utils import core as utils
from lm_human_preferences.language import model

def test_incremental():
    # 定义超参数
    hparams = model.HParams()
    hparams.override_from_dict(dict(
        n_vocab=10,  # 词汇表大小
        n_ctx=5,     # 上下文长度
        n_embd=9,    # 嵌入维度
        n_head=3,    # 注意力头数
        n_layer=2,   # 层数
    ))
    batch_size = 2  # 批量大小
    steps = 5       # 步数
    np.random.seed(7)  # 设置随机种子
    torch.manual_seed(7)

    # 定义 Transformer 模型
    m = model.Model(hparams=hparams)
    X = torch.randint(0, hparams.n_vocab, (batch_size, steps), dtype=torch.long) # 随机生成输入
    logits = m(X=X)['lm_logits'] # 获取模型的 logits 输出

    # 定义 past 张量
    past = torch.zeros(model.past_shape(hparams=hparams, batch_size=batch_size, sequence=0), dtype=torch.float32)

    # 测试增量推理
    for step in range(steps):
        # 获取当前步的 logits 和 past
        logits_v = m(X=X[:, :step + 1])['lm_logits']
        past_logits_v = m(X=X[:, step:step + 1], past=past)['lm_logits']
        past = torch.cat([past, m(X=X[:, step:step + 1], past=past)['present']], dim=-2)

        # 断言当前步的 logits 是否与增量推理的 logits 一致
        assert torch.allclose(logits_v[:, -1:], past_logits_v, atol=1e-3, rtol=1e-3)

def test_mask():
    np.random.seed(7) # 设置随机种子
    torch.manual_seed(7)

    # 定义超参数
    hparams = model.HParams()
    hparams.override_from_dict(dict(
        n_vocab=10,  # 词汇表大小
        n_ctx=8,     # 上下文长度
        n_embd=3,    # 嵌入维度
        n_head=3,    # 注意力头数
        n_layer=2,   # 层数
    ))
    batch_size = 4  # 批量大小
    policy = model.Model(hparams=hparams)  # 定义模型

    # 随机生成 past 和输入 tokens
    past_length = 4
    length = 3
    past = torch.randn(*model.past_shape(hparams=hparams, batch_size=batch_size, sequence=past_length))
    X = torch.randint(0, hparams.n_vocab, (batch_size, length), dtype=torch.long)

    # 运行模型(无间隔)
    logits = policy(past=past, X=X)['lm_logits']

    # 运行模型(有间隔)
    gap_past_length = 7
    gap_length = 5

    def random_subsequence(n, size):
        # 随机生成子序列
        sub = [np.concatenate(([0], np.random.choice(np.arange(1, n), size=size - 1, replace=False))) for _ in range(batch_size)]
        return np.sort(sub, axis=-1)
    
    past_sub = random_subsequence(n=gap_past_length, size=past_length)
    X_sub = random_subsequence(n=gap_length, size=length)
    past_gap = torch.randn(*model.past_shape(hparams=hparams, batch_size=batch_size, sequence=gap_past_length))
    X_gap = torch.randint(0, hparams.n_vocab, (batch_size, gap_length), dtype=torch.long)
    mask = torch.zeros((batch_size, gap_past_length + gap_length), dtype=torch.bool)

    for b in range(batch_size):
        for i in range(past_length):
            past_gap[b, :, :, :, past_sub[b, i]] = past[b, :, :, :, i]
        for i in range(length):
            X_gap[b, X_sub[b, i]] = X[b, i]
        mask[b, past_sub[b]] = mask[b, gap_past_length + X_sub[b]] = 1
    
    gap_logits = policy(past=past_gap, X=X_gap, mask=mask)['lm_logits']
    sub_logits = utils.index_each(gap_logits, X_sub)

    # 比较结果
    assert logits.shape == sub_logits.shape
    assert torch.allclose(logits, sub_logits, atol=1e-5)

def test_attention_mask():
    for nd in 1, 2, 3:
        for ns in range(nd, 4):
            ours = model.attention_mask(nd, ns, dtype=torch.int32)
            theirs = torch.tril(torch.ones(nd, ns), diagonal=ns - nd).to(torch.int32)
            assert torch.all(ours == theirs)

if __name__=='__main__':
    test_mask()
    test_attention_mask()
    test_incremental()