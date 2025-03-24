import os
import torch
import torch.nn as nn
from mpi4py import MPI

from lm_human_preferences.language import trained_models, model
from lm_human_preferences.utils import core as utils
from lm_human_preferences.utils.core import Schema

class RewardModelTrainer(nn.Module):
    """
    trained_model:一个预训练的语言模型,提供了模型的结构和参数
    scope:模型的命名空间,用于区分不同模型的参数
    use_resource:是否使用额外的资源(在代码中未明确使用)
    is_root:是否是主进程(用于分布式训练场景)
    """
    def __init__(self,
                 trained_model, *,
                 scope='reward_model', use_resource=False,
                 is_root=True):
        super(RewardModelTrainer, self).__init__()
        self.trained_model = trained_model # 保存传入的预训练模型对象,该模型提供了初始的权重和结构
        self.hparams = trained_model.hparams() # 从预训练模型中获取超参数
        self.is_root = is_root # 标记当前进程是否是主进程,在分布式训练中,只有主进程会执行某些操作(如加载权重)

        self.use_resource = use_resource # 标记是否使用额外资源
        self.encoder = self.trained_model.encoding.get_encoder() # 从预训练模型中获取编码器
        
        self.scope = scope # 设置模型的命名空间
        # 创建奖励模型实例, scalar_heads=['reward'] 表示模型输出一个标量奖励值
        self.model = model.Model(hparams=self.hparams, scope=f'{scope}/model', scalar_heads=['reward'])
        self.built = False # 初始化一个标志,这用于控制模型的初始化流程
        self.padding_token = self.encoder.padding_token # 获取编码器的填充 token
        self.get_rewards = self.get_rewards_op # 将 get_rewards_op 方法绑定到 get_rewards

    def get_encoder(self):
        return self.encoder
    
    # 执行奖励模型的前向传播
    def _build(self, tokens, do_dropout=False, name=None):
        with torch.no_grad():
            # 调用奖励模型的前向传播
            lm_output = self.model(X=tokens, do_dropout=do_dropout, padding_token=self.padding_token)

            reward = lm_output['reward'][:, -1] # 从模型输出中提取最后一个时间步奖励值
            with torch.no_grad():
                if not self.built:
                    self.reward_gain = nn.Parameter(torch.tensor(1.0)) # 初始化奖励归一化参数 
                    self.reward_bias = nn.Parameter(torch.tensor(0.0)) # 初始化奖励归一化偏移参数
                    self._reward_gain_p = torch.tensor(1.0) # 初始化一个临时的归一化参数
                    self._reward_bias_p = torch.tensor(0.0) # 初始化一个临时的偏移参数
                    self._set_reward_norm = lambda: None # 初始化一个空的归一化调整函数
                if reward is not None:
                    reward = self.reward_gain * reward + self.reward_bias # 对奖励值进行归一化调整
                if not self.built:
                    self._set_initializers() # 加载预训练权重
                self.built = True # 标记模型已经构建完成
                return reward
    
    # 确保模型已经构建完成
    def ensure_built(self):
        if self.built:
            return
        with torch.no_grad():
            self._build(tokens=torch.zeros([0,0], dtype=torch.int32))

    # 获取模型的可训练参数
    def get_params(self):
        self.ensure_built()
        return list(self.model.parameters()) + [self.reward_gain, self.reward_bias]
    
    # 重置奖励信号的归一化参数
    def reset_reward_scale(self):
        self.reward_gain.data.copy_(torch.tensor(1.0)) # 将 reward_gain 的值重置为 1.0
        self.reward_bias.data.copy_(torch.tensor(0.0)) # 将 reward_bias 的值重置为 0.0

    # 设置奖励信号的归一化参数
    def set_reward_norm(self, *, old_mean, old_std, new_mean, new_std):
        old_gain, old_bias = self.reward_gain.item(), self.reward_bias.item() # 获取当前的归一化参数值
        assert old_gain == 1 and old_bias == 0,\
            f'set_reward_norm expects gain = 1 and bias = 0, not {old_gain}, {old_bias}'
        gain = new_std / old_std # 计算新的归一化增益
        bias = new_mean - gain * old_mean # 计算新的偏移量
        self.reward_gain.data.copy_(torch.tensor(gain)) # 更新 reward_gain 的值
        self.reward_bias.data.copy_(torch.tensor(bias)) # 更新 reward_bias 的值
    
    # 从预训练模型加载权重
    def _set_initializers(self):
        # 如果当前进程不是主进程或者模型是测试模型,不加载权重
        if not self.is_root or self.trained_model.name == 'test':
            return
        
        with torch.no_grad():
            for name, param in self.named_parameters(): # 遍历当前模型的所有参数
                if name in self.trained_model.state_dict(): # 如果当前参数的名称在预训练模型的权重字典中
                    param.data.copy_(self.trained_model.state_dict()[name]) # 将预训练模型的权重复制到当前模型的对应参数中
    
    # 计算奖励信号
    def get_rewards_op(self, queries, responses):
        tokens = torch.cat([queries, responses], dim=1)
        return self._build(tokens)
    
class TrainedRewardModel(nn.Module):
    def __init__(self, train_dir, encoding, *, scope='reward_model', comm=MPI.COMM_WORLD):
        super(TrainedRewardModel, self).__init__()
        self.train_dir = train_dir # 保存预训练模型的存储目录
        self.comm = comm # 保存分布式通信对象

        self.encoding = encoding # 保存编码器对象
        self.encoder = encoding.get_encoder() # 获取编码器实例
        if train_dir != 'test': # 如果不是测试模式
            # 从 train_dir 中加载超参数文件
            self.hparams = trained_models.load_hparams(os.path.join(train_dir, 'hparams.json'))
            assert self.hparams.n_vocab == encoding.n_vocab, f'{self.hparams.n_vocab} != {encoding.n_vocab}'
        else:
            self.hparams = trained_models.test_hparams # 使用默认的测试超参数
        
        self.padding_token = self.encoder.padding_token # 获取编码器的填充 token

        self.scope = scope # 设置模型的命名空间
        self.model = model.Model(hparams=self.hparams, scope=f'{scope}/model', scalar_heads=['reward'])
    
    def _build(self, X):
        results = self.model(X=X, padding_token=self.padding_token)
        reward = results['reward'][:, -1]
        with torch.no_grad():
            self.reward_gain = nn.Parameter(torch.tensor(1.0))
            self.reward_bias = nn.Parameter(torch.tensor(0.0))
        reward = self.reward_gain * reward + self.reward_bias
        self._set_initializers()
        return reward
    
    def ensure_built(self):
        if self.model.built:
            return
        with torch.no_grad():
            self._build(X=torch.zeros([0, 0], dtype=torch.int32))

    def _set_initializers(self):
        if self.comm.Get_rank() > 0 or self.train_dir == 'test':
            return
        assert self.model.built
        checkpoint_scope = 'reward_model'

        with torch.no_grad():
            checkpoint = torch.load(os.path.join(self.train_dir, 'checkpoints/'))
            for name, param in self.named_parameters():
                if name.startswith(checkpoint_scope + '/'):
                    param.data.copy_(checkpoint[name])

    def get_params(self):
        return list(self.model.parameters()) + [self.reward_gain, self.reward_bias]
    
    def score_fn(self, queries, responses):
        tokens = torch.cat([queries, responses], dim=1)
        return self._build(tokens)
