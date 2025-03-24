from abc import ABC, abstractmethod
from typing import Optional, Dict

import torch
import torch.nn.functional as F

from lm_human_preferences.utils.core import Schema, pearson_r

# 定义一个名为 LabelType 的抽象基类(Abstract Base Class,ABC)
class LabelType(ABC):
    @abstractmethod # 声明抽象方法
    def label_schemas(self) -> Dict[str, Schema]:
        """定义标签(labels)的结构(schema),即标签数据的格式"""
    
    @abstractmethod
    def target_scales(self, labels: Dict[str, torch.Tensor]) -> Optional[torch.Tensor]:
        """从标签中提取与奖励模型输出相关的标量值"""

    @abstractmethod
    def loss(self, reward_model, labels: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """计算奖励模型的损失函数"""

    @abstractmethod
    def question_schemas(self, *, query_length, response_length) -> Dict[str, Schema]:
        """定义与标签相关的“问题”数据的结构(schema)"""

# “选择最佳响应”的标签类型
class PickBest(LabelType):
    def __init__(self, num_responses):
        self.num_responses = num_responses # 响应的数量
    
    def label_schemas(self):
        return dict(best=Schema(torch.int32, ())) # 标签是选择的最佳响应的索引
    
    def target_scales(self, labels):
        return None # 没有与奖励模型输出直接相关的标量值
    
    def loss(self, reward_model, labels):
        # 计算每个样本的 logits
        logits = torch.stack([reward_model(labels['query'], labels[f'sample{i}'])
                              for i in range(self.num_responses)], dim=1)
        # 计算交叉熵损失
        error = F.cross_entropy(logits, labels['best'])
        return dict(loss=error, error=error) # 返回损失
    
    def question_schemas(self, *, query_length, response_length):
        return dict(
            query=Schema(torch.int32, (query_length,)),
            **{f"sample{i}": Schema(torch.int32, (response_length,)) for i in range(self.num_responses)}
        )
# 处理“标量评分”的标签类型处理
class ScalarRating(LabelType):
    def __init__(self):
        pass

    def label_schemas(self):
        return dict(
            score=Schema(torch.float32, ()) # 标签是标量评分
        )
    
    def target_scales(self, labels):
        return labels['score'] # 返回目标尺度
    
    def loss(self, reward_model, labels):
        # 预测分数
        predicted = reward_model(labels['query'], labels['sample'])
        labels = labels['score']
        # 计算均方误差
        error = F.mse_loss(predicted, labels)
        # 计算标签的均值和方差
        label_mean = torch.mean(labels)
        label_var = torch.var(labels)
        # 计算 Pearson 相关系数
        corr = pearson_r(labels, predicted)
        return dict(loss=error, error=error,
                    label_mean=label_mean, label_var=label_var, corr=corr) # 返回损失和统计信息

    def question_schemas(self, *, query_length, response_length):
        return dict(
            query=Schema(torch.int32, (query_length,)),
            sample=Schema(torch.int32, (response_length,))
        )
    
# 用于处理“标量比较”的标签类型
class ScalarComparison(LabelType):
    def label_schemas(self):
        return dict(difference=Schema(torch.float32, ())) # 标签是两个响应的评分差值
    
    def target_scales(self, labels):
        # 将差值除以 2，使其方差与奖励模型输出一致
        return labels['difference'] / 2
    
    def loss(self, reward_model, labels):
        # 计算两个样本的奖励分数
        outputs0 = reward_model(labels['query'], labels['sample0'])
        outputs1 = reward_model(labels['query'], labels['sample1'])
        # 计算预测差值
        differences = labels['difference']
        predicted_differences = outputs1 - outputs0
        # 计算均方误差
        error = F.mse_loss(predicted_differences, differences)
        return dict(loss=error, error=error) # 返回损失
    
    def question_schemas(self, *, query_length, response_length):
        return dict(
            query=Schema(torch.int32, (query_length,)),
            sample0=Schema(torch.int32, (response_length)),
            sample1=Schema(torch.int32, (response_length))
        )
    
def get(label_type: str) -> LabelType:
    if label_type == 'scalar_rating':
        return ScalarRating()
    if label_type == 'scalar_compare':
        return ScalarComparison()
    if label_type.startswith('best_of_'):
        n = int(label_type[len('best_of_'):])
        return PickBest(n)
    raise ValueError(f"Unexpected label type {label_type}")