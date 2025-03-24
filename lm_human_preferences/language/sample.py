"""这段代码实现了一个通用的序列采样器，用于从自回归模型中生成序列。它支持以下功能：
温度控制： 通过调整温度参数，控制生成的随机性。
top-k 和 top-p 采样： 通过限制采样范围，提高生成质量。
灵活的条件函数： 可以通过 cond 函数控制生成的终止条件。
额外输出： 支持返回模型的额外输出，例如隐藏状态或其他中间结果。"""


import torch
from lm_human_preferences.language import model
from lm_human_preferences.utils import core as utils 

def sample_sequence(*, step, model_hparams, length, batch_size=None, context=None, 
                    temperature=1, top_k=0, top_p=1.0, extra_outputs={}, cond=None):
    """
    从自回归序列模型中采样

    输入:
        step: 一个函数,接收模型超参数、tokens 张量和 past,返回包含 'logits' 和 'presents' 的字典,以及其他额外变量
        context: 包含起始 tokens
        extra_outputs: 额外输出的键值对映射
    返回:
        包含 'presents', 'logits' 和 extra_outputs 中键的字典
    """
    with torch.no_grad(): # 采样过程中，不需要计算梯度
        batch_size = context.size(0) # 获取 batch_size
        
        # 计算temperature的倒数
        # temperature的倒数用于调整 logits 的分布。temperature越低，生成结果越确定；temperature越高，生成结果越随机
        beta = 1 / max(float(temperature), 1e-10)

        # 初始上下文输出
        context_output = step(model_hparams, context)
        logits = context_output['logits'][:, -1].float() # 提取最后一个 token 的 logits

        # 计算第一个输出的 logits 和采样结果
        first_output_logits = beta * logits # 使用beta调整 logits
        first_outputs = utils.sample_from_logits(first_output_logits) # 从调整后的 logits 中采样第一个 token
        first_logprobs = utils.logprobs_from_logits(logits=first_output_logits, labels=first_outputs) # 计算采样 token 的对数概率

        # 定义循环体
        def body(past, prev, output, logprobs, *extras):
            # 计算下一个 token 的 logits 和 presents
            next_outputs = step(model_hparams, prev.unsqueeze(1), past=past, past_tokens=output[:, :-1])
            logits = next_outputs['logits'].float() * beta

            # 应用 top-k 和 top-p 采样
            if top_k != 0: # 如果 top_k 不为零，只保留 logits 中的前 k 个最高概率值
                logits = utils.take_top_k_logits(logits, top_k)
            if top_p != 1.0: #如果 top_p 小于 1.0，只保留累积概率大于 p 的 token。
                logits = utils.take_top_p_logits(logits, top_p)

            # 采样下一个 token
            next_sample = utils.sample_from_logits(logits, dtype=torch.int32)

            # 计算下一个token的对数概率
            next_logprob = utils.logprobs_from_logits(logits=logits, labels=next_sample)

            # 更新past, prev, output, logprobs 和 extras
            return [
                torch.cat([past, next_outputs['presents']], dim=-2),
                next_sample.squeeze(1),
                torch.cat([output, next_sample], dim=1),
                torch.cat([logprobs, next_logprob], dim=1),
                *[torch.cat([prev, next_outputs[k]], dim=1) for k, prev in zip(extra_outputs, extras)]
            ]
        
        # 初始化循环变量
        if cond is None:
            def always_true(*args):
                return True
            cond = always_true

        presents = context_output['presents']
        prev = first_outputs
        output = torch.cat([context, first_outputs.unsqueeze(1)], dim=1)
        logprobs = first_logprobs.unsqueeze(1)
        extras = [context_output[k][:, -1:] for k in extra_outputs]

        # 使用 while 循环进行采样
        # 使用 cond 函数判断是否继续生成。如果未指定 cond，则始终继续生成
        # 循环最多执行 length - 1 次，因为第一个 token 已经在初始化时生成
        i = 0
        while cond(presents, prev, output, logprobs, *extras) and i < length - 1:
            presents, prev, output, logprobs, *extras = body(presents, prev, output, logprobs, *extras)
            i += 1
        
        # 返回结果
        return {
            'tokens': output, # 生成的 token 序列
            'presents': presents, # 最终的中间状态
            'logprobs': logprobs, # 生成 token 的对数概率
            **dict(zip(extra_outputs, extras))
        }