import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from lm_human_preferences.language import model, sample
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils.core import Schema

class Policy(nn.Module):
    """
    参数讲解：
    - trained_model:预训练的语言模型
    - scope:模型的作用域
    - use_resource:是否使用资源变量
    - embed_queries:一个函数,用于对输入的查询进行嵌入处理(默认为恒等函数)
    - temperature:生成时的温度参数，控制生成的随机性
    - is_root:是否为根节点(用于分布式训练)
    - build_respond:是否构建响应生成函数
    """
    def __init__(self,
                 trained_model, *,
                 scope=None, use_resource=False,
                 embed_queries=lambda queries: queries,
                 temperature=1.0, is_root=True,
                 build_respond=True):
        super(Policy, self).__init__()
        self.trained_model = trained_model # 保存预训练模型
        self.model_hparams = trained_model.hparams() # 获取预训练模型的超参数
        self.is_root = is_root # 标记当前进程是否是主进程

        self.use_resourece = use_resource # 标记是否使用资源变量
        self.encoder = self.trained_model.encoding.get_encoder() # 获取编码器,用于将文本转换为 token 序列

        # 创建一个自定义的 Transformer 模型实例,指定超参数和输出头
        self.model = model.Model(
            hparams=self.model_hparams,
            scalar_heads=['value']
        )

        self.built = False # 标记模型尚未构建完成
        self.embed_queries = embed_queries # 保存嵌入查询的函数
        self.temperature = temperature #  保存 temperature 参数
        self.padding_token = self.encoder.padding_token #  获取编码器的填充 token

        if build_respond:
            self.respond = self.respond_op
        self.analyze_responses = self.analyze_responses_op

    def get_encoder(self):
        return self.encoder
    

    """
    执行单步 Transformer 模型的前向传播
    使用自定义的 Transformer 模型(self.model)计算语言模型的输出
    提取 logits 和 presents(中间状态)
    如果是第一次调用，初始化模型的参数(通过 _set_initializers)
    返回 logits、values 和 presents
    """
    def step_core(self, model_hparams, tokens, past=None, past_tokens=None, do_dropout=False, name=None):
        with torch.no_grad():
            lm_output = self.model(X=tokens, past=past, past_tokens=past_tokens, do_dropout=do_dropout, padding_token=self.padding_token)

            # 需要切片 logits，因为我们不想生成特殊 token
            logits = lm_output['lm_logits'][:, :, :self.model_hparams.n_vocab] # 语言模型的输出 logits ,限制在词汇表范围内
            presents = lm_output['present'] # 当前步的中间状态
            value = lm_output['value'] # 值函数输出
            if not self.built:
                self._set_initializers() # 加载预训练权重
            self.built = True
            return {
                'logits': logits,
                'values': value,
                'presents': presents
            }
        
    # 确保模型已经构建完成。
    def ensure_built(self):
        if not self.built:
            with torch.no_grad():
                self.step_core(self.model_hparams, tokens=torch.zeros([0, 0], dtype=torch.int32))
    
    # 获取模型的可训练参数
    def get_params(self):
        self.ensure_built()
        params = list(self.parameters())
        assert len(params) > 0
        return params
    
    # 从预训练模型加载权重
    def _set_initializers(self):
        if not self.is_root or self.trained_model.name == 'test':
            return
        
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in self.trained_model.state_dict():
                    param.data.copy_(self.trained_model.state_dict()[name])

    """
    根据输入的查询生成指定长度的文本响应
    将输入的查询嵌入到模型可接受的格式
    使用 sample.sample_sequence 生成文本序列
    返回生成的响应、对数概率和值函数输出
    """
    def respond_op(self, queries, length):
        contexts = self.embed_queries(queries) # 使用 embed_queries 函数将查询嵌入到模型可接受的格式
        context_length = contexts.size(1) # 获取嵌入后查询的长度
        result = sample.sample_sequence(
            step = self.step_core, # 指定单步前向传播函数
            context = contexts, # 输入的上下文（嵌入后的查询）
            length = length, # 生成的文本长度
            model_hparams = self.model_hparams, # 模型的超参数
            temperature = self.temperature, # temperature 参数，控制生成的随机性
            extra_outputs = {'values': torch.float32} #  指定额外输出的类型（值函数输出）
        )
        return dict(
            responses = result['tokens'][:, context_length], #  生成的响应(从上下文结束位置开始提取)
            logprobs = result['logprobs'], # 生成的对数概率
            values = result['values'] #  值函数输出
        )
    
    """
    分析生成的响应，计算对数概率、熵和值函数输出
    将查询和响应拼接为完整的 token 序列
    使用 step_core 计算 logits
    计算对数概率、熵和值函数输出
    """
    def analyze_responses_op(self, queries, responses):
        contexts = self.embed_queries(queries) # 使用 embed_queries 函数将查询嵌入到模型可接受的格式
        context_length = contexts.size(1) # 获取嵌入后查询的长度
        tokens = torch.cat([contexts, responses], dim=1) # 将查询和响应拼接为完整的 token 序列
        result = self.step_core(self.model_hparams, tokens) # 计算完整的 token 序列的 logits
        logits = result['logits'][:, context_length-1:-1] # 提取响应部分的 logits(从 context_length-1 开始到倒数第二个 token)
        
        logits /= self.temperature
        return dict(
            logprobs = utils.logprobs_from_logits(logits=logits, labels=responses), # 计算的对数概率
            entropies = utils.entropy_from_logits(logits), # 计算的熵
            values = result['values'][:, context_length-1:-1] # 值函数输出（提取响应部分）
        )
    
# 计算给定 logits 和标签的对数概率
def logprobs_from_logits(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)

# 计算给定 logits 的熵
def entropy_from_logits(logits):
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -(probs * log_probs).sum(dim=-1)

# 生成指定长度的文本序列
"""
初始化 tokens 为输入的上下文
循环生成指定长度的 token:
    调用 step 函数计算 logits
    使用温度调整 logits,并采样下一个 token
    将采样的 token 添加到 tokens 中
返回生成的 tokens、对数概率和值函数输出
"""
def sample_sequence(step, context, length, model_hparams, temperature, extra_outputs):
    tokens = context
    for _ in range(length):
        output = step(model_hparams, tokens)
        logits = output['logits'][:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return dict(
        tokens = tokens,
        logprobs = logprobs_from_logits(output['logits'], tokens),
        values = output['values']
    )