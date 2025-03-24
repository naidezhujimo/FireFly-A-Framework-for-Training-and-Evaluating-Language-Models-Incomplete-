import json
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Optional

import numpy as np
import torch
import torch.distributed as dist
from mpi4py import MPI
from torch.utils.tensorboard import SummaryWriter
from torch.optim import Adam

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_human_preferences import label_types, lm_tasks, rewards
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import gcs, hyperparams
from lm_human_preferences.utils.core import Schema

# 用于存储与标签相关的超参数
@dataclass
class LabelHParams(hyperparams.HParams):
    type: str = None # 标签的类型
    num_train: int = None # 训练数据的数量
    source :str = None # 训练数据的来源

# 用于存储与运行相关的超参数
@dataclass
class RunHParams(hyperparams.HParams):
    seed: Optional[int] = None # 随机种子,用于确保实验的可重复性
    log_interval: int = 10 # 日志记录的间隔
    save_interval: int = 50 # 模型保存的间隔
    save_dir: Optional[str] = None # 模型保存的目录

# 用于存储整体的超参数
@dataclass
class HParams(hyperparams.HParams):
    run: RunHParams = field(default_factory=RunHParams) # 包含运行相关的超参数
    # 包含任务相关的超参数
    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)
    labels: LabelHParams = field(default_factory=LabelHParams) # 包含标签相关的超参数

    batch_size: int = 40 # 每个批次的大小
    lr: float = 5e-5 # 学习率

    rollout_batch_size: int = 64 # 用于采样的批次大小
    normalize_samples: int = 0 # 归一化时使用的样本数量
    normalize_before: bool = False # 是否在训练前进行归一化
    normalize_after: bool = False # 是否在训练后进行归一化

    # 验证超参数的合法性.例如确保训练数据量能被批次大小整除
    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        utils.exact_div(self.labels.num_train, self.batch_size)

# 将一个数向下取整到某个除数的倍数
def round_down_to_multiple(n, divisor):
    return n - n % divisor

# 下载标签数据
def download_labels(source, label_type, question_schemas, total_labels, comm):
    """
    参数
    - source: 数据来源
    - label_type: 标签类型
    - question_schemas: 查询数据的结构
    - total_labels: 需要的标签数量
    - comm: MPI通信对象
    - schemas: 合并了查询数据结构和标签数据结构
    """
    schemas = {**question_schemas, **label_type.lable_schemas()}

    if source != 'test': # 如果数据来源不是测试集,则从指定路径加载数据
        with open(gcs.download_file_cached(source, comm=comm)) as f:
            results = json.load(f) # 从文件中加载JSON格式的数据
            print('Num labels found in source:', len(results))
    else: # 如果是测试集,则生成虚拟数据
        results = [
            {
                name: np.zeros(schema.shape, dtype=schema.dtype.as_numpy_dtype) # 创建形状和数据类型与schema匹配的零数组
                for name, schema in schemas.items()
            }
            for _ in range(50)
        ]
    
    assert len(results) >= total_labels # 确保数据量满足需求
    results = results[:total_labels] # 截取所需数量的数据
    return {k: [a[k] for a in results] for k in schemas.keys()} # 将数据按字段重新组织为字典格式

# 定义了一个奖励模型训练器类
class RewardModelTrainer():
    def __init__(self, *, reward_model, policy, query_sampler, hparams, comm):
        """
        参数
        - reward_model: 奖励模型
        - policy: 策略模型
        - query_sampler: 查询采样器
        - hparams: 超参数
        - comm: MPI通信对象
        """
        self.reward_model = reward_model

        self.policy = policy
        self.hparams = hparams
        self.num_ranks = comm.Get_size() # MPI通信中进程的数量
        self.rank = comm.Get_rank() # 当前进程的编号
        self.comm = comm

        """
        根据超参数获取标签类型和查询数据结构
        label_type: 标签类型
        question_schemas: 查询数据的结构,根据查询长度和响应长度生成
        """
        self.label_type = label_types.get(hparams.label.type)
        self.question_schemas = self.label_type.question_schemas(
            query_length=hparams.task.query_length,
            response_length=hparams.task.response_length
        )

        # 合并查询数据结构和标签数据结构
        data_schemas = {
            **self.question_schemas,
            **self.label_type.label_schemas()
        }

        """
        初始化一个样本缓冲区,用于存储训练数据
        - capacity: 缓冲区的容量,即训练数据的数量
        - schemas: 数据的结构
        """
        self.train_buffer = utils.SampleBuffer(capacity=hparams.labels.num_train, schemas=data_schemas)

        """
        初始化TensorBoard日志记录器
        - log_dir: 日志保存的路径
        """
        self.summary_writer = SummaryWriter(log_dir=os.path.join(hparams.run.save_dir, 'reward_model'))

        """
        初始化优化器，用于更新奖励模型的参数
        - lr: 学习率
        """
        self.optimizer = Adam(self.reward_model.parameters(), lr=hparams.lr)

        """
        如果需要在训练前或训练后进行归一化,则定义一个统计函数stats
        - query_responses: 查询和响应的组合
        - rewards: 计算奖励值
        - means和stds: 计算奖励值的均值和标准差
        - self.comm.allreduce: 使用MPI进行全局归约操作,计算所有进程的总和
        """
        if self.hparams.normalize_before or self.hparams.normalize_after:
            def stats(query_responses):
                rewards = np.concatenate([self.reward_model.get_rewards(qs, rs) for qs, rs in query_responses], axis=0)
                assert len(rewards.shape) == 1, f'{rewards.shape}'
                sums = np.asarray([rewards.sum(axis=0), np.square(rewards).sum(axis=0)])
                means, sqr_means = self.comm.allreduce(sums, op=dist.ReduceOp.SUM) / (self.num_ranks * rewards.shape[0])
                stds = np.sqrt(sqr_means - means ** 2)
                return means, stds
            self.stats = stats

            # 在归一化后记录统计信息
            def log_stats_after_normalize(stats):
                if comm.Get_rank() != 0: # 只有在根进程(rank=0)中打印日志
                    return
                means, stds = stats
                print(f'after normalize: {means} +- {stds}')
            self.log_stats_after_normalize = log_stats_after_normalize

            # 用于重置奖励模型的比例
            def reset_reward_scales():
                self.reward_model.reset_reward_scale()
            self.reset_reward_scales = reset_reward_scales

            """
            设置奖励模型的归一化参数
            - mean和std: 当前的均值和标准差
            - new_mean和new_std: 目标均值和标准差
            """
            def set_reward_norms(mean, std, new_mean, new_std):
                print(f'targets: {new_mean} +- {new_std}')
                print(f'before normalize: {mean} +- {std}')
                assert np.isfinite((mean, std, new_mean, new_std)).all()
                self.reward_model.set_reward_norm(old_mean=mean, old_std=std, new_mean=new_mean, new_std=new_std)
                self.set_reward_norms = set_reward_norms
            
            if self.hparams.normalize_before or self.hparams.normalize_after:
                # 从策略模型中采样查询和响应
                def sample_policy_batch():
                    queries = query_sampler('ref_queries')['tokens']
                    responses = policy.respond(
                        queries=queries, length=hparams.task.response_length)['responses']
                    return queries, responses
                
                # 采样指定数量的样本
                def sample_policy_responses(n_samples):
                    n_batches = utils.ceil_div(n_samples, hparams.rollout_batch_size)
                    return [sample_policy_batch() for _ in range(n_batches)]
                self.sample_policy_responses = sample_policy_responses

    """
    定义归一化方法
    - sample_fn: 采样函数
    - target_means和target_stds: 目标均值和标准差
    """
    def normalize(self, sample_fn, target_means, target_stds):
        if not self.hparams.normalize_samples:
            return
    
        self.reset_reward_scales() # 重置奖励模型的比例
        query_responses = sample_fn(self.hparams.normalize_samples) # 使用采样函数获取样本
        means, stds = self.stats(query_responses) # 计算样本的均值和标准差

        # 设置奖励模型的归一化参数
        self.set_reward_norms(means, stds, target_means, target_stds)
        if self.hparams.debug_normalize: # 如果开启了调试模式,则再次采样并记录归一化后的统计信息
            query_responses = sample_fn(self.hparams.normalize_samples)
            stats = self.stats(query_responses)
            self.log_stats_after_normalize(stats)
    
    # 开始训练过程
    def train(self):
        labels = download_labels(
            self.hparams.labels.source,
            label_type=self.label_type,
            question_schemas=self.question_schemas,
            total_labels=self.hparams.labels.num_train,
            comm=self.comm
        )

        self.train_buffer.add(labels) # 将下载的标签数据添加到样本缓冲区

        if self.hparams.normalize_before:
            target_mean, target_std = self.target_mean_std()
            self.normalize(self.sample_policy_responses, target_mean, target_std)

        # 计算每个进程的批次大小
        per_rank_batch_size = utils.exact_div(self.hparams.batch_size, self.num_ranks)

        # 使用MPI广播操作,将训练数据的索引随机打乱
        train_indices = self.comm.bcast(np.random.permutation(self.hparams.labels.num_train))

        print(self.rank, "training on", self.hparams.labels.num_train, "in batches of", per_rank_batch_size)
        """
        遍历训练数据，按批次进行训练
        - start_index和end_index: 当前批次的起始和结束索引
        - all_rank_indices: 当前批次的所有索引
        - our_indices: 当前进程负责的索引
        - lr: 动态调整的学习率
        """
        for start_index in range(0, self.hparams.labels.num_train, self.hparams.batch_size):
            end_index = start_index + self.hparams.batch_size
            all_rank_indices = train_indices[start_index, end_index]
            our_indices = all_rank_indices[self.rank::self.num_ranks]
            lr = (1 - start_index / self.hparams.labels.num_train) * self.hparams.lr

            batch_data = self.train_buffer.read(our_indices) # 从缓冲区读取当前批次的数据

            rewards = self.reward_model(batch_data) # 计算奖励值
            loss = self.label_type.loss(rewards, batch_data) # 计算损失

            self.optimizer.zero_grad() # 清空梯度
            loss.backward() # 反向传播计算梯度
            self.optimizer.step() # 更新模型参数

            # 如果是根进程,并且满足日志记录的间隔,则记录损失值
            if self.rank == 0 and start_index & self.hparams.run.log_interval == 0:
                self.summary_writer.add_scalar('loss', loss.item(), start_index)
        
        if self.hparams.normalize_after:
            target_mean, target_std = np.zeros([]), np.ones([])
            self.normalize(self.sample_policy_responses, target_mean, target_std)


def train(hparams: HParams):
    # 设置随机种子以确保实验的可重复性
    utils.set_mpi_seed(hparams.run.seed)

    # 加载预训练模型
    m = trained_models.TrainedModel(hparams.task.policy.initial_model)
    # 获取编码器
    encoder = m.encoding.get_encoder()
    # 保存模型超参数
    hyperparams.dump(m.hparams(), name='model_hparams')

    # 初始化 MPI 通信
    comm = MPI.COMM_WORLD
    # 创建参考策略模型
    ref_policy = Policy(
        m, scope='ref_policy',
        is_root=comm.Get_rank() == 0,  # 仅在根节点执行某些操作
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder), # 格式化查询
        temperature=hparams.task.policy.temperature,  # 设置 temperature 参数
        build_respond=False # 不构建响应函数
    )

    # 创建奖励模型训练器
    reward_model = rewards.RewardModelTrainer(m, is_root=comm.Get_rank() == 0)

    # 创建查询采样器
    query_sampler = lm_tasks.make_query_sampler(
        hparams=hparams.task, encoder=encoder, comm=comm,
        batch_size=utils.exact_div(hparams.rollout_batch_size, comm.Get_size()) # 计算每个 rank 的 batch size
    )

    # 创建奖励模型训练器实例
    reward_trainer = RewardModelTrainer(
        reward_model=reward_model,
        policy=ref_policy,
        query_sampler=query_sampler,
        hparams=hparams,
        comm=comm
    )

    # 设置保存目录
    save_dir = hparams.run.save_dir
    if comm.Get_rank() == 0 and save_dir: # 仅在根节点执行保存操作
        print(f'Wile save to {save_dir}')
        # 创建保存目录
        if not save_dir.startswith('gs://'): # 如果不是 Google Cloud Storage 路径
            os.makedirs(os.path.join(save_dir, 'reward_model'), exist_ok=True)
        # 保存训练超参数
        with open(os.path.join(save_dir, 'train_reward_hparams.json'), 'w') as f:
            json.dump(hparams.to_nested_dict(), f, indent=2)
        # 保存奖励模型超参数
        with open(os.path.join(save_dir, 'reward_model', 'hparams.json'), 'w') as f:
            json.dump(reward_model.hparams.to_nested_dict(), f, indent=2)
        # 保存编码器名称
        with open(os.path.join(save_dir, 'reward_model', 'encoding'), 'w') as f:
            json.dump(reward_model.trained_model.encoding.name, f, indent=2)

    # 初始化模型参数
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.weight)
    ref_policy.apply(init_weights)
    reward_model.apply(init_weights)

    # 同步所有 rank 的模型参数
    def sync_models():
        for param in ref_policy.parameters():
            dist.broadcast(param.data, src=0) # 将根进程的模型参数广播到其他进程
        for param in reward_model.parameters():
            dist.broadcast(param.data, src=0) # 将根进程的模型参数广播到其他进程
    
    # 初始化分布式训练环境
    dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo')
    sync_models() # 同步模型参数

    # 训练奖励模型
    reward_trainer.train()

    # 保存模型
    if comm.Get_rank() == 0 and save_dir:
        checkpoint_dir = os.path.join(save_dir, 'reward_model/checkpoints/model.ckpt')
        torch.save({
            'ref_policy_state_dict': ref_policy.state_dict(),
            'reward_model_state_dict': reward_model.state_dict(),
        }, checkpoint_dir)