import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import tempfile
import numpy as np

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_human_preferences import train_policy


class HParams:
    def __init__(self):
        self.ppo = self.PPO()
        self.task = self.Task()
        self.rewards = self.Rewards()
        self.run = self.Run()

    class PPO:
        def __init__(self):
            self.batch_size = 8
            self.total_episodes = 8
    
    class Task:
        def __init__(self):
            self.policy = self.Policy()
            self.query_length = 2
            self.response_length = 3
            self.query_dataset = 'test'
            self.truncate_token = None
            self.truncate_after = None
            self.query_prefix = None
            self.query_suffix = None

        class Policy:
            def __init__(self):
                self.initial_model = 'test'
    
    class Rewards:
        def __init__(self):
            self.trained_model = 'test'
            self.adaptive_kl = 'off'
            self.adaptive_kl_target = 3.0
            self.adaptive_kl_horizon = 100
            self.train_new_model = False  # 是否启用训练新模型
            self.train_new_model_params = self.TrainNewModelParams()  # 嵌套参数结构

        class TrainNewModelParams:
            def __init__(self):
                self.task = self.TaskParams()
                self.labels = self.LabelParams()
                self.batch_size = 8  # 添加 batch_size 属性

            class TaskParams:
                def __init__(self):
                    self.policy = self.PolicyParams()
                    self.query_length = None
                    self.response_length = None
                    self.query_dataset = None

                class PolicyParams:
                    def __init__(self):
                        self.initial_model = None

            class LabelParams:
                def __init__(self):
                    self.source = None
                    self.num_train = None
                    self.type = None
    
    class Run:
        def __init__(self):
            self.log_interval = 1
            self.save_dir = None
            self.save_interval = None

    def override_from_dict(self, override_dict):
        for key, value in override_dict.items():
            keys = key.split('.')
            obj = self
            last_key = keys[-1]
            for k in keys[:-1]:
                obj = getattr(obj, k)
                if isinstance(obj, str):
                    raise AttributeError(f"Unsupported path {key}, attribute {k} is a string.")
            # 设置最后一个键
            if isinstance(obj, str):
                raise AttributeError(f"Unsupported path {key}, attribute {last_key} is part of a string.")
            setattr(obj, last_key, value)

    def validate(self):
        pass

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
class RewardNetwork(nn.Module):
    def __init__(self, input_size):
        super(RewardNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
def train_policy_test(override_params):
    hparams = HParams()
    hparams.override_from_dict(override_params)
    hparams.validate()

    # 初始化策略网络
    policy_net = PolicyNetwork(input_size=hparams.task.query_length, output_size=hparams.task.response_length)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    # 训练循环
    for episode in range(hparams.ppo.total_episodes):
        # 生成模拟数据
        queries = torch.randn(hparams.ppo.batch_size, hparams.task.query_length)
        responses = policy_net(queries)

        # 计算奖励（这里使用随机奖励作为示例）
        rewards = torch.randn(hparams.ppo.batch_size, 1)

        # 计算策略梯度
        optimizer.zero_grad()
        loss = -torch.mean(rewards * torch.log(responses))
        loss.backward()
        optimizer.step()

        if episode % hparams.run.log_interval == 0:
            print(f"Episode {episode}, Loss: {loss.item()}")


def test_truncation():
    train_policy_test({
        'task.truncate_token': 13,
        'task.truncate_after': 2,
    })

def test_defaults():
    train_policy_test({})

def test_affixing():
    train_policy_test({
        'task.query_prefix': 'a',
        'task.query_suffix': 'b'
    })

def test_adaptive_kl():
    train_policy_test({
        'rewards.trained_model': 'test', # not sure why needed
        'rewards.adaptive_kl': 'on',
        'rewards.adaptive_kl_target': 3.0,
        'rewards.adaptive_kl_horizon': 100,
    })

def test_save():
    train_policy_test({
        'run.save_dir': tempfile.mkdtemp() ,
        'run.save_interval': 1
    })


# 测试用例
def test_reward_training():
    train_policy_test({
        'rewards.train_new_model': True,  # 启用训练新奖励模型
        'rewards.train_new_model_params.task.policy.initial_model': 'test',
        'rewards.train_new_model_params.task.query_length': 2,
        'rewards.train_new_model_params.task.response_length': 3,
        'rewards.train_new_model_params.task.query_dataset': 'test',
        'rewards.train_new_model_params.labels.source': 'test',
        'rewards.train_new_model_params.labels.num_train': 16,
        'rewards.train_new_model_params.batch_size': 8,
        'rewards.train_new_model_params.labels.type': 'best_of_4',
    })

if __name__ == "__main__":
    test_truncation()
    test_defaults()
    test_affixing()
    test_adaptive_kl()
    test_save()
    test_reward_training()