import os
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from typing import Optional
from functools import partial
from mpi4py import MPI
import sys
import json

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from lm_human_preferences import lm_tasks, train_reward
from lm_human_preferences.language import trained_models
from lm_human_preferences.policy import Policy
from lm_human_preferences.rewards import TrainedRewardModel
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils import hyperparams
from lm_human_preferences.utils.core import Schema

@dataclass
class AdaptiveKLParams(hyperparams.HParams):
    target: float = None
    horizon: int = 10000  # 在多少 episode 内调整 KL 散度

@dataclass
class RewardHParams(hyperparams.HParams):
    kl_coef: float = 0.2  # KL 散度系数
    adaptive_kl: Optional[AdaptiveKLParams] = None  # 自适应 KL 散度参数
    trained_model: Optional[str] = None  # 预训练模型路径
    train_new_model: Optional[train_reward.HParams] = None  # 训练新模型的参数

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        assert self.trained_model is None or self.train_new_model is None, 'Cannot use trained_model and train new model'
        assert self.trained_model is not None or self.train_new_model is not None, 'Need either trained_model or to train a new model'

@dataclass
class PpoHParams(hyperparams.HParams):
    total_episodes: int = 2000000  # 总训练 episode 数
    batch_size: int = 64  # 每批次的样本数
    nminibatches: int = 1  # 每个批次分成多少个小批次
    noptepochs: int = 4  # 每个批次的优化 epoch 数
    lr: float = 5e-6  # 学习率
    vf_coef: float = .1  # 价值函数的系数
    cliprange: float = .2  # PPO 的 clip 范围
    cliprange_value: float = .2  # 价值函数的 clip 范围
    gamma: float = 1  # 折扣因子
    lam: float = 0.95  # GAE 的 lambda 参数
    whiten_rewards: bool = True  # 是否对奖励进行白化处理

@dataclass
class HParams(hyperparams.HParams):
    run: train_reward.RunHParams = field(default_factory=train_reward.RunHParams)
    task: lm_tasks.TaskHParams = field(default_factory=lm_tasks.TaskHParams)
    rewards: RewardHParams = field(default_factory=RewardHParams)
    ppo: PpoHParams = field(default_factory=PpoHParams)

    def validate(self, *, prefix=''):
        super().validate(prefix=prefix)
        minibatch_size = utils.exact_div(self.ppo.batch_size, self.ppo.nminibatches)
        if self.ppo.whiten_rewards:
            assert minibatch_size >= 8, f"Minibatch size {minibatch_size} is insufficient for whitening in PPOTrainer.loss"

def nupdates(hparams):
    return utils.ceil_div(hparams.ppo.total_episodes / hparams.ppo.batch_size)

# 定义 KL 控制器
class FixedKLController:
    def __init__(self, kl_coef):
        self.value = kl_coef # 固定 KL 系数
    
    def update(self, current, n_steps):
        pass # 固定 KL 系数不需要更新

# 定义 KL 控制器
class AdaptiveKLController:
    def __init__(self, init_kl_coef, hparams):
        self.value = init_kl_coef # 初始 KL 系数
        self.hparams = hparams # 超参数

    def update(self, current, n_steps):
        target = self.hparams.target # 目标 KL 值
        proportional_error = np.clip(current / target -1, -0.2, 0.2) # 计算比例误差
        mult = 1 + proportional_error * n_steps / self.hparams.horizon # 调整系数
        self.value *= mult # 更新 KL 系数

# 定义 PPO 训练器
class PPOTrainer:
    def __init__(self, policy, ref_policy, query_sampler, score_fn, hparams, comm):
        self.comm = comm  # MPI 通信器
        self.policy = policy  # 策略模型
        self.ref_policy = ref_policy  # 参考策略模型
        self.score_fn = score_fn  # 奖励计算函数
        self.hparams = hparams  # 超参数

        # 初始化 KL 控制器
        if hparams.rewards.adaptive_kl is None:
            self.kl_ctl = FixedKLController(hparams.rewards.kl_coef)
        else:
            self.kl_ctl = AdaptiveKLController(hparams.rewards.kl_coef, hparams=hparams.rewards.adaptive_kl)

        # 定义采样查询的函数
        self.sample_queries = query_sampler

        # 定义计算奖励的函数
        def compute_rewards(scores, logprobs, ref_logprobs):
            kl = logprobs - ref_logprobs # 计算 KL 散度
            non_score_reward = -self.kl_ctl.value * kl # 非分数奖励
            rewards = non_score_reward.clone() # 克隆奖励
            rewards[:, -1] += scores # 将分数奖励加到最后一个时间步
            return rewards, non_score_reward, self.kl_ctl.value
        self.compute_rewards = compute_rewards

        # 初始化优化器
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=hparams.ppo.lr)

    def step(self):
        step_started_at = time.time()

        # 采样查询
        queries = self.sample_queries()
        rollouts = self.policy.respond(queries, length=self.hparams.task.response_length)

        # 计算奖励
        responses = rollouts['responses']
        logprobs = rollouts['logprobs']
        ref_logprobs = self.ref_policy.analyze_responses(queries, responses)['logprobs']
        scores, postprocessed_responses, score_stats = self.score_fn(queries, responses)

        rewards, non_score_reward, kl_coef = self.compute_rewards(
            scores=scores,
            logprobs=logprobs,
            ref_logprobs=ref_logprobs)
        rollouts['rewards'] = rewards

        # 训练模型
        train_stats = self.train(rollouts=rollouts)

        # 记录统计信息
        self.record_step_stats(
            scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs,non_score_reward=non_score_reward,
            train_stats=train_stats, score_stats=score_stats, kl_coef=kl_coef
        )

        # 更新 KL 控制器
        self.kl_ctl.update(train_stats['objective/kl'], self.hparams.ppo.batch_size)

        # 打印样本
        self.print_samples(queries=queries, responses=postprocessed_responses,
                           scores=scores, logprobs=logprobs, ref_logprobs=ref_logprobs)
        
        # 记录时间
        step_time = time.time() - step_started_at
        eps_per_second = float(self.hparams.ppo.batch_size) / step_time
        if self.comm.Get_rank() == 0:
            print(f"[ppo_step {self.global_step}] step_time={step_time:.2f}s, eps/s={eps_per_second:.2f}")

    def train(self, rollouts):
        # 训练逻辑
        stat_list = []

        # 每个 rank 的批次大小
        per_rank_rollout_batch_size = utils.exact_div(self.hparams.ppo.batch_size, self.comm.Get_size())
        per_rank_minibatch_size = utils.exact_div(per_rank_rollout_batch_size, self.hparams.ppo.nminibatches)

        # 多轮 PPO 训练
        for ppo_epoch_idx in range(self.hparams.ppo.noptepochs):
            order = np.random.permutation(per_rank_rollout_batch_size) # 随机打乱顺序
            for mb_start in range(0, per_rank_rollout_batch_size, per_rank_minibatch_size):
                mb_data = {k: v[order[mb_start:mb_start+per_rank_minibatch_size]]
                           for k, v in rollouts.items()} # 获取小批次数据

                # 计算损失并更新模型
                loss, stats = self.loss(mb_data)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                stat_list.append(stats)

        # 收集统计信息
        return {k: [s[k] for s in stat_list] for k in stat_list[0].keys()}
    
    def loss(self, rollouts):
        values = rollouts['values']
        old_logprob = rollouts['logprobs']
        rewards = rollouts['rewards']

        # 计算优势函数
        advantages = self.compute_advantages(rewards, values)

        # 计算策略损失
        outputs = self.policy.analyze_responses(rollouts['queries'], rollouts['responses'])
        logprob = outputs['logprobs']
        ratio = torch.exp(logprob - old_logprob)
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(ratio, 1.0 - self.hparams.ppo.cliprange, 1.0 + self.hparams.ppo.cliprange)
        pg_loss = torch.mean(torch.max(pg_losses, pg_losses2))

        # 计算价值函数损失
        vpred = outputs['values']
        vpredclipped = torch.clamp(vpred, values - self.hparams.ppo.cliprange_value, values + self.hparams.ppo.cliprange_value)
        vf_losses1 = torch.square(vpred - rewards)
        vf_losses2 = torch.square(vpredclipped - rewards)
        vf_loss = 0.5 * torch.mean(torch.max(vf_losses1, vf_losses2))

        # 总损失
        loss = pg_loss + self.hparams.ppo.vf_coef * vf_loss

        # 计算统计信息
        stats = {
            'loss': {'policy': pg_loss.item(), 'value': vf_loss.item(), 'total': loss.item()},
            'policy': {'entropy': torch.mean(outputs['entropies']).item()},
            'returns': {'mean': torch.mean(rewards).item(), 'var': torch.var(rewards).item()},
            'val': {'vpred': torch.mean(vpred).item(), 'error': torch.mean((vpred - rewards) ** 2).item()}
        }
        return loss, stats
    
    def compute_advantages(self, rewards, values):
        # 计算 GAE 优势函数
        advantages = []
        lastgaelam = 0
        for t in reversed(range(self.hparams.task.response_length)):
            nextvalues = values[:, t + 1] if t < self.hparams.task.response_length - 1 else 0.0
            delta = rewards[:, t] + self.hparams.ppo.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.hparams.ppo.gamma * self.hparams.ppo.lam * lastgaelam
            advantages.append(lastgaelam)
        advantages = torch.stack(advantages[::-1], dim=1)
        return advantages
    
def make_score_fn(hparams, score_model):
    padding_token = score_model.padding_token # 获取填充标记

    # 定义后处理函数
    postprocess_fn = lm_tasks.postprocess_fn_from_hparams(hparams, padding_token)

    def postprocess(responses):
        return postprocess_fn(responses)
    
    # 定义过滤函数
    filter_fn = lm_tasks.filter_fn_from_hparams(hparams)

    def penalize(responses, rewards):
        valid = filter_fn(responses) # 判断响应是否有效
        return torch.where(valid, rewards, hparams.penalty_reward_value * torch.ones_like(rewards))
    
    def unpenalized_score_fn(queries, responses):
        return score_model.score_fn(queries, responses)
    
    def score_fn(queries, responses):
        responses = postprocess(responses) # 对响应进行后处理
        score = penalize(responses, unpenalized_score_fn(queries, responses)) # 计算惩罚后的分数
        stats = {'score': score.mean().item()}  # 统计分数的均值
        return score, responses, stats

    # 定义评分函数的统计模式
    score_fn.stat_schemas = {'score': Schema(torch.float32, (None,))}
    
    return score_fn


def train(hparams: HParams):
    save_dir = hparams.run.save_dir

    if hparams.rewards.train_new_model:
        assert hparams.task == hparams.rewards.train_new_model.task, \
            f'{hparams.task} != {hparams.rewards.train_new_model.task}'
        hparams.rewards.train_new_model.run.save_dir = save_dir
        train_reward.train(hparams.rewards.train_new_model)
        if 'pytest' in sys.modules:
            hparams.rewards.trained_model = 'test'  # 测试模式下使用虚拟模型
        elif save_dir:
            hparams.rewards.trained_model = os.path.join(save_dir, 'reward_model')  # 保存奖励模型路径

    comm = MPI.COMM_WORLD # 初始化 MPI 通信器

    # 设置随机种子
    utils.set_mpi_seed(hparams.run.seed)

    # 加载预训练模型和编码器
    m = trained_models.TrainedModel(hparams.task.policy.initial_model)
    encoder = m.encoding.get_encoder()

    # 保存超参数和模型配置
    if save_dir:
        os.makedirs(os.path.join(save_dir, 'policy'), exist_ok=True)  # 创建保存目录
        with open(os.path.join(save_dir, 'train_policy_hparams.json'), 'w') as f:
            json.dump(hparams.to_nested_dict(), f, indent=2)  # 保存超参数
        with open(os.path.join(save_dir, 'policy', 'hparams.json'), 'w') as f:
            json.dump(m.hparams().to_nested_dict(), f, indent=2)  # 保存模型配置
        with open(os.path.join(save_dir, 'policy', 'encoding'), 'w') as f:
            json.dump(m.encoding.name, f, indent=2)  # 保存编码器配置
    
    # 初始化奖励模型
    score_model = TrainedRewardModel(hparams.rewards.trained_model, m.encoding, comm=comm)

    # 初始化参考策略模型
    ref_policy = Policy(
        m, scope='ref_policy',
        is_root=comm.Get_rank() == 0,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=hparams.task.policy.temperature,
        build_respond=False)
    
    # 初始化策略模型
    policy = Policy(
        m, scope='policy',
        is_root=comm.Get_rank() == 0,
        embed_queries=lm_tasks.query_formatter(hparams.task, encoder),
        temperature=hparams.task.policy.temperature
    )
    
    # 初始化查询采样器
    query_sampler = lm_tasks.make_query_sampler(
        hparams=hparams.task, encoder=encoder, comm=comm,
        batch_size=utils.exact_div(hparams.ppo.batch_size, comm.Get_size())
    )

    # 检查小批次大小是否满足白化需求
    per_rank_minibatch_size = utils.exact_div(hparams.ppo.batch_size, hparams.ppo.nminibatches * comm.Get_size())
    if hparams.ppo.whiten_rewards:
        assert per_rank_minibatch_size >= 8, \
            f"Per-rank minibatch size {per_rank_minibatch_size} is insufficient for whitening"
        
    # 初始化 PPO 训练器
    ppo_trainer = PPOTrainer(
        policy=policy, ref_policy=ref_policy,  query_sampler=query_sampler,
        score_fn=make_score_fn(hparams.tesk, score_model=score_model),
        hparams=hparams, comm=comm
    )

    # 初始化模型保存器
    if comm.Get_rank() == 0 and save_dir:
        print(f"Will save to {save_dir}")
        checkpoint_dir = os.path.join(save_dir, 'policy/checkpoints')
        os.makedirs(checkpoint_dir, exist_ok=True) # 创建检查点目录
    else:
        checkpoint_dir = None

    # 同步模型参数
    def sync_models():
        score_model.ensure_built()
        params = list(score_model.parameters()) + list(ref_policy.parameters()) + list(policy.parameters())
        for param in params:
            dist.broadcast(param.data, src=0) # 使用 PyTorch 的分布式广播同步参数

    # 训练循环
    global_step = 0
    while global_step < nupdates(hparams):
        ppo_trainer.step() # 执行一步 PPO 训练
        global_step += 1

        # 定期保存模型
        if comm.Get_rank() == 0 and checkpoint_dir and global_step % hparams.run.save_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'model_{global_step}.pt')
            torch.save({
                'global_step': global_step,
                'policy_state_dict': policy.state_dict(),
                'ref_policy_state_dict': ref_policy.state_dict(),
                'score_model_state_dict': score_model.state_dict(),
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")

    # 训练结束后保存最终模型
    if comm.Get_rank == 0 and checkpoint_dir:
        final_checkpoint_path = os.path.join(checkpoint_dir, 'model_final.pt')
        torch.save({
            'global_step': global_step,
            'policy_state_dict': policy.state_dict(),
            'ref_policy_state_dict': ref_policy.state_dict(),
            'score_model_state_dict': score_model.state_dict(),
        }, final_checkpoint_path)
        print(f"Saved final checkpoint to {final_checkpoint_path}")

