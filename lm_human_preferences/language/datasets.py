import random
from typing import Dict, Iterator, Optional
import torch
from torch.utils.data import IterableDataset
from lm_human_preferences.datasets.books import books_generator
from lm_human_preferences.datasets.cnndm import cnndm_generator
from lm_human_preferences.datasets.tldr import tldr_generator

# 注册表，保存不同数据集的生成器函数
_registry: Dict[str, Iterator] = {}

def register_dataset(name: str, generator: Iterator) -> None:
    """注册数据集生成器函数到全局注册表"""
    _registry[name] = generator

def get_dataset(name: str) -> Iterator:
    """从注册表中获取指定名称的数据集生成器"""
    return _registry[name]

class TextDataset(IterableDataset):
    def __init__(
         self,
         generator: Iterator,
         encoder: object,
         sequence_length: int,
         mode: str = 'train',
         start_token: Optional[int] = None,
         end_token: Optional[int] = None,
         padding_token: Optional[int] = None,
         seed: int = 0,
         shuffle: bool = True,
         comm: Optional[object] = None   
    ):
        """
        参数
        - generator: 文本生成器函数
        - encoder: 编码器对象，需实现 encode 方法
        - sequence_length: 固定序列长度
        - mode: 数据集模式 (train/valid/test)
        - start_token: 起始标记(编码后)
        - end_token: 结束标记(编码后)
        - padding_token: 填充标记(编码后)
        - seed: 随机种子
        - shuffle: 是否打乱数据
        - comm: 分布式通信对象(MPI等)
        """
        super().__init__()
        self.generator = generator
        self.encoder = encoder
        self.sequence_length = sequence_length
        self.mode = mode
        self.start_token = start_token
        self.end_token = end_token
        self.padding_token = padding_token or getattr(encoder, 'padding_token', 0)
        self.seed = seed
        self.shuffle = shuffle
        self.comm = comm
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """生成器：产生处理后的样本"""
        random.seed(self.seed)  # 设置随机种子

        # 获取分布式参数（如果存在）
        if self.comm is not None:
            num_shards = self.comm.Get_size()
            shard_idx = self.comm.Get_rank()
        else:
            num_shards = 1
            shard_idx = 0

        # 获取原始文本生成器
        texts = self.generator(
            mode=self.mode, seed=self.seed, shuffle=self.shuffle, comm=self.comm
        )

        for i,text in enumerate(texts):
            # 分布式分片处理：每个进程只取自己负责的数据
            if i % num_shards != shard_idx:
                continue

            # 文本编码为token序列
            tokens = self.encoder.encode(text)

            # 处理起始标记（截断前面内容）
            if self.start_token is not None:
                try:
                    start_pos = tokens.index(self.start_token) + 1
                    tokens = tokens[start_pos:] if start_pos < len(tokens) else []
                except ValueError:
                    continue  # 如果找不到起始标记则跳过
            
            # 截断到最大长度
            tokens = tokens[: self.sequence_length]

            # 处理结束标记（截断后面内容）
            if self.end_token is not None:
                try:
                    end_pos = len(tokens) -1 - tokens[::-1].index(self.end_token)
                    tokens = tokens[:end_pos]
                except ValueError:
                    continue  # 如果找不到结束标记则跳过
            
            # 填充到固定长度
            if len(tokens) < self.sequence_length:
                padding = [self.padding_token] * (self.sequence_length - len(tokens))
                tokens += padding
            else:
                tokens = tokens[:self.sequence_length]

            # 转换成张量
            yield {
                'tokens': torch.tensor(tokens, dtype=torch.long)
            }


# ================= 注册各数据集生成器 =================
# 注意：这里需要实际实现各生成器函数
def cnndm_generator(mode: str, **kwargs) -> Iterator[str]:
    # 实际应实现CNN/DM数据集的加载逻辑
    while True: yield "sample text"

def tldr_generator(mode: str, **kwargs) -> Iterator[str]:
    # 实际应实现TLDR数据集的加载逻辑
    while True: yield "sample text"

def books_generator(mode: str, **kwargs) -> Iterator[str]:
    # 实际应实现Books数据集的加载逻辑
    while True: yield "sample text"

def test_generator(mode: str, **kwargs) -> Iterator[str]:
    # 测试用随机文本生成器
    random.seed(kwargs.get('seed', 0))
    while True:
        yield ''.join(random.choices('abcdefghijklmnopqrstuvwxyz .', k=40))

# 注册各数据集
register_dataset("cnndm", cnndm_generator)
register_dataset("tldr", tldr_generator)
register_dataset("books", books_generator)
register_dataset("test", test_generator)


if __name__ == "__main__":
    # 假设已实现编码器类
    class DummyEncoder:
        padding_token = 0
        start_token = 1
        end_token = 2
        
        def encode(self, text: str) -> list:
            return [random.randint(3, 100) for _ in text.split()]

    encoder = DummyEncoder()
    
    # 创建测试数据集
    dataset = TextDataset(
        generator=get_dataset("test"),
        encoder=encoder,
        sequence_length=16,
        mode="test",
        start_token=encoder.start_token,
        end_token=encoder.end_token,
        padding_token=encoder.padding_token,
        seed=42,
    )

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=4,
        num_workers=2,
    )

    # 迭代获取数据
    for batch in dataloader:
        print("Batch shape:", batch["tokens"].shape)
        print("Sample tokens:", batch["tokens"][0])
        input("Press Enter to continue...")