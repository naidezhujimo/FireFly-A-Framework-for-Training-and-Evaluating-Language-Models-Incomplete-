import torch
import numpy as np

import os
import sys

# 获取当前脚本的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取 FireFly 的路径
firefly_path = os.path.abspath(os.path.join(current_directory, '..', '..'))
sys.path.insert(0, firefly_path)  # 将 FireFly 路径添加到 sys.path

from lm_human_preferences.language import sample  # 修复后的导入路径


# 定义超参数
n_vocab = 10  # 词汇表大小
batch_size = 2  # 批量大小
hparams = {
    'n_layer': 0,
    'n_head': 1,
    'n_embd': 0,
    'n_attn': 0,
}

# 定义一个步进函数，该函数根据当前 tokens 生成下一个 token 的 logits
def step(hparams, tokens, past=None, past_tokens=None):
    # 生成 logits，选择前一个 token + 1 作为下一个 token
    next_tokens = tokens[:, -1] + 1  # 获取最后一个 token
    # 使用 clamp 确保 next_tokens 在有效范围内
    next_tokens = torch.clamp(next_tokens, min=0, max=n_vocab - 1)
    logits = torch.nn.functional.one_hot(next_tokens, n_vocab).float()
    logits[logits == 0] = -np.inf  # 将其他位置设置为 -inf
    return {
        'logits': logits,
        'presents': torch.zeros(2, 0, 2, 1, 0, 0)  # 返回一个空的 presents 张量
    }

# 定义采样函数
def sample_sequence(step, model_hparams, length, batch_size, context):
    tokens = context.clone() # 复制初始上下文
    for _ in range(length):
        # 调用步进函数生成下一个 token 的 logits
        outputs = step(model_hparams, tokens)
        logits = outputs['logits']
        # 使用 argmax 选择下一个 token
        next_tokens = torch.argmax(logits, dim=-1)
        # 将新生成的 token 添加到 tokens 中
        tokens = torch.cat([tokens, next_tokens.unsqueeze(-1)], dim=-1)
    return {'tokens': tokens}

# 测试函数
def test_sample_sequence():
    context = torch.tensor([[5, 0], [4, 3]]) # 初始上下文
    output = sample_sequence(step=step, model_hparams=hparams, length=4, batch_size=batch_size, context=context)
    expected = torch.tensor([[5, 0, 1, 2, 3, 4], [4, 3, 4, 5, 6, 7]]) # 期望的输出

    # 断言生成的 tokens 是否与期望的输出一致
    assert torch.allclose(output['tokens'], expected), f"Expected {expected}, but got {output['tokens']}"

if __name__ == '__main__':
    test_sample_sequence()
