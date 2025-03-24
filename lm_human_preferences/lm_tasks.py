from dataclasses import dataclass, field
from typing import Optional
import torch
from torch.utils.data import DataLoader
from lm_human_preferences.language import datasets
from lm_human_preferences.utils import hyperparams

@dataclass
class PolicyHParams(hyperparams.HParams):
    temperature: float = 1.0  # 用于控制采样时的随机性
    initial_model: str = None  # 初始模型的路径或名称

@dataclass
class TaskHParams(hyperparams.HParams):
    # 查询参数
    query_length: int = None  # 查询的长度
    query_dataset: str = None  # 查询数据集的名称
    query_prefix: str = ''  # 查询的前缀
    query_suffix: str = ''  # 查询的后缀
    start_text: Optional[str] = '.'  # 查询的起始文本
    end_text: Optional[str] = None  # 查询的结束文本

    # 响应参数
    response_length: int = None # 响应的长度

    # 在采样时, 从 'truncate_after 开始, 遇到 'truncate_token' 后截断响应
    truncate_token: Optional[int] = None # 截断的token
    truncate_after: int = 0 # 从哪个位置开始检查截断 token
    penalty_reward_value: int = -1 # 未通过过滤的响应的惩罚值

    policy: PolicyHParams = field(default_factory=PolicyHParams) # 策略的超参数

def postprocess_fn_from_hparams(hparams: TaskHParams, padding_token: int):
    """根据超参数生成后处理函数,用于在评分前对响应进行处理"""
    def get_mask(responses, truncate_token, truncate_after):
        # 生成一个掩码，标记从 `truncate_after` 开始第一个 `truncate_token` 之后的位置
        mask = (responses == truncate_token).int()
        mask = torch.cat([torch.zeros_like(mask)[:, :truncate_after], mask[:, truncate_after:]], dim=1)
        return (torch.cumsum(mask, dim=1) - mask).bool()
    
    if hparams.truncate_token is not None:
        def truncate(responses):
            # 将 'truncate_token' 之后的部分替换为 'padding_token'
            mask = get_mask(responses, hparams.truncate_token, hparams.truncate_after)
            return torch.where(mask, padding_token * torch.ones_like(responses), responses)
        return truncate
    else:
        return lambda responses: responses # 如果没有截断 token, 直接返回原始响应
    
def filter_fn_from_hparams(hparams: TaskHParams):
    """根据超参数生成过滤函数,用于筛选响应"""
    def filter(responses):
        if hparams.truncate_token is not None:
            # 检查响应中是否包含 'turncate_token'
            matches_token = (responses[:, hparams.truncate_after:] == hparams.truncate_token)
            return matches_token.any(dim=-1)
        else:
            # 如果没有截断 token, 所有响应都通过
            return torch.ones(responses.size(0), dtype=torch.bool)
    return filter

def query_formatter(hparams: TaskHParams, encoder):
    """将查询转换为语言模型的输入上下文"""
    def query_formatter(queries):
        batch_size = queries.size(0)
        prefix_tokens = torch.tensor(encoder.encode(hparams.query_prefix), dtype=torch.int32)
        tiled_prefix = prefix_tokens.unsqueeze(0).expand(batch_size, -1) # 扩展前缀以匹配批次大小
        suffix_tokens = torch.tensor(encoder.encode(hparams.query_suffix), dtype=torch.int32)
        tiled_suffix = suffix_tokens.unsqueeze(0).expand(batch_size, -1) # 扩展后缀以匹配批次大小

        return torch.cat([tiled_prefix, queries, tiled_suffix], dim=-1) # 拼接前缀、查询和后缀
    return query_formatter

def make_query_sampler(*, hparams: TaskHParams, encoder, batch_size: int, mode='train', comm=None):
    """创建查询采样器,用于从数据集中生成查询"""
    if hparams.start_text:
        start_token = encoder.encode(hparams.start_text)[0] # 起始token
    else:
        start_token = None
    
    if hparams.end_text:
        end_token = encoder.encode(hparams.end_text)[0] # 结束token
    else:
        end_token = None
    
    # 获取数据集
    dataset = datasets.get_dataset(hparams.query_dataset).torch_dataset(
        sequence_length=hparams.query_length, mode=mode, comm=comm, encoder=encoder,
        start_token=start_token, end_token=end_token,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=True) # 创建DataLoader

    def sampler(scope=None):
            """采样函数，返回一个批次的查询 tokens"""
            # 这里改为使用循环来提供批次，避免只返回第一个批次
            iterator = iter(dataloader)
            while True:
                try:
                    batch = next(iterator)
                    context_tokens = batch['tokens'].to(torch.int32)
                    return dict(tokens=context_tokens)
                except StopIteration:
                    # 数据用尽，重新重置 DataLoader
                    iterator = iter(dataloader)

    return sampler