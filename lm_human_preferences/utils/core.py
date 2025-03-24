import os
import platform
import subprocess
import collections
import contextlib
import inspect
from functools import partial, wraps
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Optional, List

import numpy as np
import torch 
from mpi4py import MPI
import shutil
from functools import lru_cache
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist

def nvidia_gpu_count():
    """ 计算当前机器上的GPU数量 """
    if shutil.which('nvidia-smi') is None:
        return 0
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=gpu_name', '--format=csv'])
    except subprocess.CalledProcessError:
        # 如果没有 GPU 或驱动程序未运行，返回 0
        return 0
    return max(0, len(output.split(b'\n')) - 2)

def get_local_rank_size(comm):
    """ 返回当前进程在本地机器上的排名和本地机器上的进程总数 """
    this_node = platform.node()
    ranks_nodes = comm.allgather((comm.Get_rank(), this_node))
    node2rankssofar = collections.defaultdict(int)
    local_rank = None
    for (rank, node) in ranks_nodes:
        if rank == comm.Get_rank():
            local_rank = node2rankssofar[node]
        node2rankssofar[node] += 1
    assert local_rank is not None
    return local_rank, node2rankssofar[this_node]

@lru_cache()
# 获取可用的 GPU 设备列表
def gpu_devices():
    if 'CUDA_VISIBLE_DEVICES' in os.environ:
        raise ValueError('CUDA_VISIBLE_DEVICES should not be set (it will cause nccl slowdowns). Use VISIBLE_DEVICES instead!')
    devices_str = os.environ.get('VISIBLE_DEVICES')
    if devices_str is not None:
        return list(map(int, filter(len, devices_str.split(','))))
    else:
        return list(range(nvidia_gpu_count()))

# 获取可用的 GPU 数量
@lru_cache()
def gpu_count():
    return len(gpu_devices()) or None


@lru_cache()
def _our_gpu():
    """ 确定当前 MPI 进程应该使用的 GPU """
    gpus = gpu_devices()
    if not gpus:
        return None
    rank = MPI.COMM_WORLD.Get_rank()
    local_rank, local_size = get_local_rank_size(MPI.COMM_WORLD)
    if gpu_count() not in (0, local_size):
        raise ValueError('Expected one GPU per rank, got gpus %s, local size %d' % (gpus, local_size))
    gpu = gpus[local_rank]
    print('rank %d: gpus = %s, our gpu = %d' % (rank, gpus, gpu))
    return gpu

def mpi_session_config():
    """配置 PyTorch 会话，仅使用分配给当前 MPI 进程的 GPU。"""
    gpu = _our_gpu()
    if gpu is not None:
        torch.cuda.set_device(gpu)
    return {}

def mpi_session():
    """创建 PyTorch 会话，仅使用分配给当前 MPI 进程的 GPU。"""
    return torch.device('cuda' if _our_gpu() is not None else 'cpu')


def set_mpi_seed(seed: Optional[int]):
    """设置随机种子，确保 MPI 进程之间的随机性一致"""
    if seed is not None:
        rank = MPI.COMM_WORLD.Get_rank()
        seed = seed + rank * 100003 # 保持向后兼容性
    torch.manual_seed(seed)
    np.random.seed(seed)

def exact_div(a, b): # 实现精确整除
    q = a // b
    if a != q * b:
        raise ValueError('Inexact division: %s / %s = %s' % (a, b, a / b))
    return q

def ceil_div(a, b): # 实现向上取整的除法
    return (a - 1) // b + 1


def expand_tile(value, size, *, axis, name=None):
    """在指定轴上扩展并平铺张量"""
    value = torch.tensor(value)
    expanded = value.unsqueeze(axis) # 在指定轴上扩展维度
    tile_dims = [1] * axis + [size] + [1] * (value.dim() - axis) # 构造平铺的维度
    return expanded.repeat(tile_dims) # 平铺张量

def index_each(a, ix):
    """对张量进行批量索引操作"""
    a = torch.tensor(a)
    ix = torch.tensor(ix, dtype=torch.int32)
    i0 = torch.arange(a.size(0), dtype=ix.dtype) # 创建索引范围,表示 a 的第 0 维的索引
    if ix.dim() > 1: # 如果 ix 是多维的
        i0 = i0.view(-1, *([1] * (ix.dim() - 1)).expand_as(ix)) # 将 i0 广播到与 ix 相同的形状
    return torch.gather(a, 1, ix) # 根据 ix 的索引从 a 的第一维中选择元素


def cumulative_max(x):
    """计算张量的累积最大值"""
    x = torch.tensor(x)
    repeated = x.unsqueeze(-1).repeat(*([1] * x.dim()), x.size(-1)) # 创建重复张量
    # 创建上三角矩阵
    upper_triangle = torch.triu(torch.ones_like(repeated, dtype=torch.bool), diagonal=0)
    # 创建一个与 repeated 形状相同的张量,所有元素为负无穷大
    neg_inf = torch.ones_like(repeated) * -float('inf')
    # 使用 torch.where,将 upper_triangle 为 True 的位置保留 repeated 的值,其余位置替换为 neg_inf
    prefixes = torch.where(upper_triangle, repeated, neg_inf)
    # 在prefixes 倒数第二个维度上计算累积最大值
    return torch.max(prefixes, dim=-2)[0]

def flatten_dict(nested, sep=''):
    """
    将嵌套字典展平

    参数：
    - nested: 输入的嵌套字典
    - sep: 用于连接嵌套键的分隔符，默认为空字符串''
    """
    def rec(nest, prefix, into):
        for k, v in nest.items():
            if sep in k: # 检查当前键 k 是否包含分隔符 sep
                raise ValueError(f"separator '{sep}' not allowed to be in key '{k}'")
            if isinstance(v, collections.Mapping): # 如果当前值 v 是一个字典,则递归调用 rec,并将当前键 k 添加到前缀中
                rec(v, prefix + k + sep, into)
            else:
                into[prefix + k] = v # 不是字典,则将键值对存储到目标字典 into 中
    flat = {}
    rec(nested, '', flat)
    return flat

@dataclass
class Schema:
    """定义张量的数据类型和形状"""
    dtype: Any
    shape: Tuple[Optional[int], ...]

class SampleBuffer:
    """用于存储和采样数据的循环缓冲区"""
    def __init__(self, *, capacity: int, schemas: Dict[str, Schema], name=None) -> None:
        self._capacity = capacity
        self._total = 0
        self._vars = {n: torch.zeros((capacity,) + s.shape, dtype=s.dtype) for n, s in schemas.items()}

    def add(self, **data):
        """将新数据添加到缓冲区"""
        n = next(iter(data.values())).size(0)
        i0 = self._total % self._capacity
        i1 = min(i0 + n, self._capacity)
        i2 = (i0 + n) % self._capacity
        slices = slice(i0, i1), slice(i2, i2 + n -(i1 - i0))
        for k, d in data.items():
            self._vars[k][slices[0]] = d[:i1-i0]
            if i2 > 0:
                self._vars[k][slices[1]] = d[i1-i0:]
        self._total += n

    def size(self):
        """返回当前缓冲区中的数据量。"""
        return min(self._total, self.capacity)
    
    def total(self):
        return self._total.read_value()

    def sample(self, n, seed=None):
        """从缓冲区中随机采样 n 个数据"""
        size = min(self._total, self._capacity)
        indices = torch.randint(0, size, (n,))
        return {k: v[indices] for k, v in self._vars.items()}
    
    def read(self, indices: torch.Tensor):
        """
        根据索引读取数据。

        参数:
        - indices (torch.Tensor): 索引张量。

        返回:
            Dict[str, torch.Tensor]: 读取的数据。
        """
        return {name: buffer[indices] for name, buffer in self._vars.items()}

    def data(self):
        """返回缓冲区中的所有数据。"""
        return {name: buffer[:self.size()] for name, buffer in self._vars.items()}
    
    def write(self, indices: torch.Tensor, updates: Dict[str, torch.Tensor]):
        """
        根据索引更新缓冲区中的数据。

        参数:
            indices (torch.Tensor): 要更新的索引。
            updates (Dict[str, torch.Tensor]): 要更新的数据，键为字段名，值为张量。
        """
        for k, v in updates.items():
            self._vars[k][indices] = v

    def write_add(self, indices: torch.Tensor, deltas: Dict[str, torch.Tensor]):
        """
        根据索引向缓冲区中的数据添加增量。

        参数:
            indices (torch.Tensor): 要更新的索引。
            deltas (Dict[str, torch.Tensor]): 要添加的增量，键为字段名，值为张量。
        """
        for k, d in deltas.items():
            self.buffers[k][indices] += d

def where(cond: torch.Tensor, true: torch.Tensor, false: torch.Tensor):
    """
    类似于 tf.where,但支持标量广播

    参数:
    - cond (torch.Tensor): 条件张量
    - true (torch.Tensor): 条件为真时的值
    - false (torch.Tensor): 条件为假时的值

    返回:
        torch.Tensor: 结果张量
    """
    # 广播标量值
    if true.dim() == 0:
        true = true.expand_as(cond)
    if false.dim() == 0:
        false = false.expand_as(cond)
    # 如果 cond[i] 为 True，则输出张量的第 i 个元素为 true[i]
    # 如果 cond[i] 为 False，则输出张量的第 i 个元素为 false[i]
    return torch.where(cond, true, false)


def map_flat(f, values: List[torch.Tensor]):
    """
    将函数 f 应用于展平后的张量，然后将结果拆分并重塑回原始形状

    参数:
    - f (Callable): 要应用的函数
    - values (List[torch.Tensor]): 输入张量列表

    返回:
        List[torch.Tensor]: 处理后的张量列表
    """
    flat = torch.cat([v.view(-1) for v in values], dim=0) # 展平并拼接所有张量
    flat = f(flat)
    parts = torch.split(flat, [v.numel() for v in values]) # 拆分处理后的张量
    return [p.view(v.shape) for p, v in zip(parts, values)]  # 重塑回原始形状


def mpi_bcast(comm, value: torch.Tensor, root=0):
    """
    使用 MPI 广播张量

    参数:
    - comm: MPI 通信器
    - value (torch.Tensor): 要广播的张量
    - root (int): 广播的根进程

    返回:
        torch.Tensor: 广播后的张量
    """
    if comm.Get_size() == 1:
        return value
    if comm.Get_rank() == root:
        comm.bcast(value.numpy(), root=root) # 根进程广播张量
    else:
        value_np = np.empty_like(value.numpy()) # 非根进程创建一个空数组用于接收
        comm.bcast(value_np, root=root)  # 接收广播的张量
        value = torch.from_numpy(value_np).to(value.device) # 将接收的 NumPy 数组转换回 PyTorch 张量,并确保其设备与原始张量一致
    return value


def mpi_allreduce_sum(values: torch.Tensor, comm):
    """
    使用 MPI 对所有进程的张量求和

    参数:
    - values (torch.Tensor): 输入张量
    - comm: MPI 通信器

    返回:
        torch.Tensor: 求和后的张量
    """
    if comm.Get_size() == 1:
        return values
    output = torch.zeros_like(values)
    # 对所有进程的 values 进行全局归约求和
    """
    values.numpy() 和 output.numpy() 将 PyTorch 张量转换为 NumPy 数组,因为 MPI 的 Allreduce 操作需要操作 NumPy 数组
    op=dist.ReduceOp.SUM 指定归约操作为“求和”
    """
    comm.Allreduce(values.numpy(), output.numpy(), op=dist.ReduceOp.SUM)
    return output


def mpi_allreduce_mean(values: torch.Tensor, comm):
    """
    使用 MPI 对所有进程的张量求平均

    参数:
    - values (torch.Tensor): 输入张量
    - comm: MPI 通信器

    返回:
        torch.Tensor: 平均后的张量
    """
    values = mpi_allreduce_sum(values, comm) # 对所有进程的输入张量进行全局归约求和
    return values / comm.Get_size() # 计算全局平均值

def mpi_bcast_tensor_dict(d: Dict[str, torch.Tensor], comm):
    """
    使用 MPI 广播字典中的张量

    参数:
    - d (Dict[str, torch.Tensor]): 要广播的字典
    - comm: MPI 通信器

    返回:
        Dict[str, torch.Tensor]: 广播后的字典
    """
    sorted_keys = sorted(d.keys()) # 按键排序，确保所有进程的顺序一致
    values = map_flat_bits(partial(mpi_bcast, comm), [d[k] for k in sorted_keys]) # 广播每个张量
    return {k: v for k, v in zip(sorted_keys, values)} # 返回广播后的字典

def variable_synchronizer(comm, vars: List[torch.Tensor], limit=1 << 28):
    """
    使用 MPI 同步变量

    参数:
    - comm: MPI 通信器
    - vars (List[torch.Tensor]): 要同步的变量列表
    - limit (int): 每块的最大字节数

    返回:
        None
    """
    if comm.Get_size() == 1: # 如果通信器中只有一个进程，无需同步
        return

    # 将变量分块,按变量名排序,确保所有进程中的变量顺序一致
    batches = chunk_tensors(sorted(vars, key=lambda v: v.name), limit=limit)

    # 同步每个块
    for batch in batches:
        # 对每个分块调用 map_flat_bits,将变量展平、广播后再恢复原始形状
        values = map_flat_bits(partial(mpi_bcast, comm), batch)
        for var, value in zip(batch, values):
            var.copy_(value) # 将广播后的值更新到变量中


def mpi_read_file(comm, path: str):
    """
    使用 MPI 读取文件并广播内容

    参数:
    - comm: MPI 通信器
    - path (str): 文件路径

    返回:
        bytes: 文件内容
    """
    if comm.Get_rank() == 0: # 根进程读取文件内容
        with open(path, 'rb') as fh:
            data = fh.read()
        comm.bcast(data) # 将文件内容广播到所有其他进程
    else:
        data = comm.bcast(None) # 如果不是根进程,接受广播内容
    return data

def chunk_tensors(tensors: List[torch.Tensor], limit=1 << 28):
    """
    将张量列表分块，每块的大小不超过 limit 字节

    参数:
    - tensors (List[torch.Tensor]): 输入张量列表
    - limit (int): 每块的最大字节数

    返回:
        List[List[torch.Tensor]]: 分块后的张量列表
    """
    total = 0 # 当前分块的总字节数
    batches = [] # 用于存储分块后的张量列表
    for v in tensors:
        size = v.element_size() * v.numel() # 计算当前张量的字节数
        if not batches or total + size > limit: # 如果当前分块已满或为空
            total = 0 # 重置当前分块的总字节 
            batches.append([]) # 创建新的分块
        total += size  # 更新当前分块的总字节数
        batches[-1].append(v) # 将当前张量加入到最新的分块中
    return batches

def map_flat_chunked(f, values: List[torch.Tensor], limit=1 << 29):
    """
    将函数 f 应用于分块、展平、连接后的张量，然后将结果拆分并重塑回原始形状

    参数:
    - f (Callable): 要应用的函数
    - values (List[torch.Tensor]): 输入张量列表
    - limit (int): 每个分块的最大字节数

    返回:
        List[torch.Tensor]: 处理后的张量列表
    """
    # 使用 chunk_tensors 函数将输入张量列表 values 分成多个分块,每个分块的总字节数不超过 limit
    chunks = chunk_tensors(values, limit=limit)
    #对每个分块调用 map_flat 函数,将函数 f 应用于分块后的张量
    mapped_values = [v for chunk in chunks for v in map_flat(f, chunk)]
    return mapped_values


def map_flat_bits(f, values: List[torch.Tensor]):
    """
    将函数 f 应用于位连接后的张量，然后转换回原始形状和数据类型

    参数:
    - f (Callable): 要应用的函数
    - values (List[torch.Tensor]): 输入张量列表

    返回:
        List[torch.Tensor]: 处理后的张量列表
    """
    bits = [v.to(torch.uint8) for v in values] # 将每个输入张量转换为 torch.uint8 类型,以减少内存占用
    flat = torch.cat([b.view(-1) for b in bits], dim=0) # 将每个张量展平为一维张量,并将它们连接成一个长张量
    flat = f(flat)
    # 将处理后的长张量拆分成多个部分,每个部分的大小与原始输入张量一致
    parts = torch.split(flat, [b.numel() for b in bits])
    # 将每个部分重塑回原始输入张量的形状
    # 将数据类型转换回原始输入张量的数据类型
    return [p.view(b.shape).to(v.dtype) for p, v, b in zip(parts, values, bits)]


@dataclass
class FlatStats:
    def __init__(self, keys: Tuple[str, ...], flat: torch.Tensor):
        self.keys = keys # 包含统计信息的键
        self.flat = flat # 包含所有统计信息的值

    @staticmethod
    def from_dict(stats: Dict[str, torch.Tensor]):
        keys = tuple(sorted(stats.keys())) # 将字典的键排序并转换为元组,确保键的顺序一致
        flat = torch.stack([stats[k] for k in keys]) # 按排序后的键顺序,将字典中的值堆叠成一个一维张量 flat
        return FlatStats(keys, flat)

    def concat(self, more: 'FlatStats'):
        dups = set(self.keys) & set(more.keys)
        if dups: # 检查两个实例是否有重复的键
            raise ValueError(f'Duplicate statistics: {", ".join(dups)}')
        # 将两个实例的键和值分别合并
        return FlatStats(self.keys + more.keys, torch.cat([self.flat, more.flat], dim=0))

    def as_dict(self):
        flat = torch.unbind(self.flat, dim=0) # 将 flat 张量解绑为一个张量列表
        return dict(zip(self.keys, flat))  # 使用 zip 将键和值配对，创建一个字典

    def with_values(self, flat: torch.Tensor):
        return FlatStats(self.keys, flat) # 使用新的值张量 flat 替换当前实例的值

    def map_flat(self, f):
        return FlatStats(self.keys, f(self.flat)) # 对值张量 flat 应用一个函数 f

def find_trainable_variables(key: str, model: torch.nn.Module):
    """
    查找模型中名称以 key 开头的可训练参数

    参数:
    - key (str): 参数名前缀
    - model (torch.nn.Module): 模型

    返回:
        List[torch.Tensor]: 可训练参数列表
    """
    return [v for k, v in model.named_parameters() if k.startswith(key + '/')] # 这里的 key + '/' 是一个约定,表示参数名的层级结构(类似于路径)


def variables_on_gpu(model: torch.nn.Module):
    """
    将模型参数移动到 GPU

    参数:
    - model (torch.nn.Module): 模型

    返回:
        None
    """
    if torch.cuda.is_available():
        model.cuda()


def graph_function(**schemas: Schema):
    """
    将函数包装为图函数

    参数:
    - schemas (Dict[str, Schema]): 输入模式

    返回:
        Callable: 包装后的函数
    """
    def decorate(make_op):
        def make_ph(path, schema):
            return torch.zeros(schema.shape, dtype=schema.dtype) # 根据输入模式创建占位符

        # 根据输入模式(schemas)为每个输入参数创建占位符
        phs = {k: make_ph(k, v) for k, v in schemas.items()}
        op = make_op(**phs) # 使用这些占位符调用原始函数 make_op,生成一个计算图(op)
        sig = inspect.signature(make_op) # 获取原始函数的签名,以便在运行时绑定参数

        @wraps(make_op)
        def run(*args, **kwargs):
            bound = sig.bind(*args, **kwargs) # 将输入参数绑定到函数签名,
            bound.apply_defaults() # 应用默认值
            feed = {k: torch.tensor(v) for k, v in bound.arguments.items()} # 将输入参数转换为张量
            return op(**feed) # 将转换后的张量传递给计算图 op

        return run
    return decorate


def shape_list(x: torch.Tensor):
    """
    获取张量的形状列表，支持动态形状

    参数:
    - x (torch.Tensor): 输入张量

    返回:
        List[int]: 形状列表
    """
    return list(x.shape)


def safe_zip(*args):
    """
    将多个序列压缩为一个元组列表，要求所有序列长度相同

    参数:
    - *args: 输入序列

    返回:
        List[Tuple]: 压缩后的元组列表
    """
    if len(args) == 0:
        return []
    for a in args[1:]:
        if len(args[0]) != len(a): # 从第二个序列开始,检查每个序列的长度是否与第一个序列的长度一致
            raise ValueError(f'Lengths do not match: {[len(a) for a in args]}')
    return list(zip(*args)) # 将多个序列压缩为一个元组列表然后转换为列表

def entropy_from_logits(logits):
    """计算熵"""
    pd = torch.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)

def logprobs_from_logits(*, logits, labels):
    """计算对数概率"""
    return -torch.nn.functional.cross_entropy(logits, labels, reduction='none')

def sample_from_logits(logits, dtype=torch.int32):
    """从 logits 中采样,根据概率分布随机选择一个类别"""
    return torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).to(dtype)

def take_top_k_logits(logits, k):
    """从 logits 中取 top-k"""
    values, _ = torch.topk(logits, k=k) # 提取每个位置的 top-k 最高值并忽略索引
    min_values = values[:, :, -1, None] # 每个位置的第 k 大值,None是将其扩展为与 logits 相同的维度,方便后续比较
    # 替换小于 top-k 值的 logits 为 -1e10
    return torch.where(logits < min_values, torch.ones_like(logits) * -1e10, logits)
    
def take_top_p_logits(logits, p):
    """从 logits 中取 top-p"""
    sorted_logits, _ = torch.sort(logits, descending=True, dim=-1) # 按降序对每个位置的 logits 进行排序
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1) # 计算累积概率
    indices = torch.sum((cumulative_probs <= p).int(), dim=-1) - 1 # 找到累积概率小于等于 p 的最大索引
    min_values = torch.gather(sorted_logits, -1, indices.unsqueeze(-1)) # 提取每个位置的最小截断值
    # 替换小于 top-p 值的 logits 为 -1e10
    return torch.where(logits < min_values, torch.ones_like(logits) * -1e10, logits)

def whiten(values, shift_mean=True):
    """对张量进行白化处理"""
    mean = torch.mean(values) # 计算均值
    var = torch.var(values) # 计算方差
    whitened = (values - mean) * torch.rsqrt(var + 1e-8) # 白化处理
    if not shift_mean: # 是否重新偏移
        whitened += mean 
    return whitened

def pearson_r(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    计算两个一维张量之间的 Pearson 相关系数。

    参数:
    - x (torch.Tensor): 第一个一维张量。
    - y (torch.Tensor): 第二个一维张量。

    返回:
        torch.Tensor: Pearson 相关系数。
    """
    # 确保输入张量是一维的
    assert x.dim() == 1, "输入张量 x 必须是一维的"
    assert y.dim() == 1, "输入张量 y 必须是一维的"

    # 计算 x 和 y 的均值
    x_mean = x.mean()
    y_mean = y.mean()

    # 计算 x 和 y 的方差
    x_var = x.var(unbiased=False)  # 使用有偏估计（与 TensorFlow 行为一致）
    y_var = y.var(unbiased=False)

    # 计算协方差
    cov = ((x - x_mean) * (y - y_mean)).mean()

    # 计算 Pearson 相关系数
    return cov / torch.sqrt(x_var * y_var)



def get_summary_writer(save_dir, subdir='', rank=0):
    if rank != 0: # 只有主进程（rank=0）负责保存日志，其他进程不需要记录日志
        return None
    if save_dir is None: # 确认配置日志路径
        return None
    """save_dir 是主目录
    'tb' 是一个固定的子目录名称，通常用于存放 TensorBoard 的日志文件
    subdir 是用户指定的额外子目录，用于进一步区分不同的日志(例如不同实验或不同配置)"""
    return SummaryWriter(os.path.join(save_dir, 'tb', subdir)) # 用于拼接完整的日志路径



def record_stats(stats, summary_writer, step, log_interval, name=None, rank=0):
    """
    一个字典,包含需要记录的统计信息。键(k)是统计信息的名称，值(v)是对应的数值
    summary_writer: 一个 SummaryWriter 对象,用于将统计信息写入 TensorBoard 日志文件。如果为 None,则不会写入日志文件
    step: 当前的步数(例如训练的迭代次数或 epoch)。用于在 TensorBoard 中记录统计信息的时间轴
    log_interval: 日志记录的间隔。只有当 step 是 log_interval 的整数倍时,才会记录日志
    name: 一个可选参数,未在代码中使用,可能是为后续扩展预留的
    rank: 当前进程的编号。在分布式训练中,只有主进程(rank=0)会执行日志记录操作
    """
    if rank != 0 or step % log_interval != 0:
        return

    for k, v in stats.items():
        print(f'{k} = {v}')
        if summary_writer:
            # 将统计信息写入 TensorBoard 日志文件
            summary_writer.add_scalar(k, v, step)


def minimize(loss, params, lr, name=None, comm=None):
    # 初始化优化器
    optimizer = torch.optim.Adam(params, lr=lr, eps=1e-5)
    optimizer.zero_grad() # 清除之前的梯度信息，避免梯度累加
    loss.backward() # 计算损失函数关于每个参数的梯度
    if comm is not None and comm.Get_size() > 1: # 检查是否提供了有效的通信器对象以及是否有多于一个进程参加训练
        for param in params:
            # 将所有进程中的梯度进行全局归约,归约操作会将所有进程的梯度相加，并将结果广播到每个进程
            dist.all_reduce(param.grad, op=dist.ReduceOp.SUM) 
            param.grad /= comm.Get_size() # 将归约后的梯度除以进程总数,以计算平均梯度
    optimizer.step() # 根据计算出的梯度更新模型参数

