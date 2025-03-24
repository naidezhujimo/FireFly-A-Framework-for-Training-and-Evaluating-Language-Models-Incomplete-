import numpy as np
import torch


def exact_div(a, b):
    # 精确除法
    if a % b != 0:
        raise ValueError(f"{a} is not divisible by {b}")
    return a // b

def ceil_div(a, b):
    # 向上取整的除法
    return -(-a // b)

def expand_tile(x, size, axis):
    # 在指定轴上扩展并重复张量
    x = x.unsqueeze(axis) # 在指定轴上增加一个维度
    shape = list(x.shape)
    shape[axis] = size # 将指定轴的大小设置为 size
    return x.expand(shape) # 扩展张量


class Schema:
    # 用于存储数据类型和形状信息的类
    def __init__(self, dtype, shape):
        self.dtype = dtype
        self.shape = shape


# 用于存储和采样数据,它有一个固定容量的缓冲区,可以添加数据并从中随机采样
class SampleBuffer:
    # 用于存储和采样数据的缓冲区
    def __init__(self, capacity, schemas):
        self.capacity = capacity
        self.buffer = {key: torch.zeros((capacity, *schema.shape), dtype=schema.dtype)
                       for key, schema in schemas.items()}
        self.size = 0
        self.index = 0

    def add(self, **kwargs):
         #向缓冲区添加数据
        for key, value in kwargs.items():
            # 如果 value 是一维张量,而 buffer 是二维张量,则将 value 扩展为二维
            if value.dim() == 1 and self.buffer[key].dim() == 2:
                value = value.unsqueeze(0)
            self.buffer[key][self.index] = value
        self.index = (self.index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)


    def sample(self, batch_size):
        # 从缓冲区中随机采样数据
        indices = torch.randint(0, self.size, (batch_size,))
        return {key: value[indices] for key, value in self.buffer.items()}
    
    def data(self):
        # 返回缓冲区中的所有数据
        return {key: value[:self.size] for key, value in self.buffer.items()}
    

def where(condition, x, y):
    # 根据条件选择元素
    return torch.where(condition, x, y)

def map_flat(func, inputs):
    # 对输入的每个元素应用函数
    return [func(x) for x in inputs]

def map_flat_bits(func, inputs):
    # 对输入的每个元素的每个位应用函数
    return [func(x) for x in inputs]

def cumulative_max(x):
    # 计算累积最大值
    return torch.cummax(x, dim=-1).values

def index_each(x, i):
    # 根据索引选择每个样本的元素
    return x[torch.arange(x.shape[0]), i]

def index_each_many(x, i):
    # 根据多个索引选择每个样本的元素
    return x[torch.arange(x.shape[0]).unsqueeze(-1), i]

def graph_function(**schemas):
    # 装饰器,用于将函数转换为 PyTorch 计算图
    def decorator(func):
        def warpper(*args, **kwargs):
            return func(*args, **kwargs)
        return warpper
    return decorator

def take_top_k_logits(logits, k):
    # 保留前 logits 中前 k 个最大值,其余置为负无穷
    values, _ = torch.topk(logits, k, dim=-1)  # 获取前 k 个最大值
    min_values = values[..., -1].unsqueeze(-1)  # 获取第 k 个最大值，并扩展维度
    return torch.where(logits >= min_values, logits, torch.tensor(float('-inf'), dtype=logits.dtype, device=logits.device))

def take_top_p_logits(logits, p):
    # 保留 logits 中概率和不超过 p 的最大值,其余置为负无穷
    sorted_logits, sorted_indices = torch.sort(logits, descending=True) # 对 logits 按降序排序
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > p
    # 将布尔张量向右移动一位，确保累积概率刚好超过 p 的第一个位置被保留
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0
    # 将布尔张量映射回原始 logits 的索引
    indices_to_remove = sorted_indices_to_remove.scatter(-1, sorted_indices, sorted_indices_to_remove)
    # 将布尔张量中标记为 True 的位置的 logits 值置为负无穷
    return logits.masked_fill(indices_to_remove, float('-inf'))

def safe_zip(*args):
    # 安全的zip操作,确保所有输入长度相同
    if len(set(len(arg) for arg in args)) != 1:
        raise ValueError("All inputs must have the same length")
    return zip(*args)

if __name__ == '__main__':
    # 测试 exact_div
    assert exact_div(12, 4) == 3
    assert exact_div(12, 3) == 4
    try:
        exact_div(7, 3)
        assert False
    except ValueError:
        pass

    # 测试 ceil_div
    for b in range(1, 10 + 1):
        for a in range(-10, 10 + 1):
            assert ceil_div(a, b) == int(np.ceil(a / b))

    # 测试 expand_tile
    np.random.seed(7)
    size = 11
    for shape in (), (7,), (3, 5):
        data = np.asarray(np.random.randn(*shape), dtype=np.float32)
        x = torch.tensor(data)
        for axis in range(-len(shape) - 1, len(shape) + 1):
            y = expand_tile(x, size, axis=axis)
            assert torch.all(torch.tensor(np.expand_dims(data, axis=axis)) == y)

    # 测试 SampleBuffer
    capacity = 100
    batch = 17
    lots = 100
    buffer = SampleBuffer(capacity=capacity, schemas=dict(x=Schema(torch.int32, (batch,))))
    for i in range(20):
        buffer.add(x=torch.arange(batch) + batch * i)
        samples = buffer.sample(lots)['x']
        hi = batch * (i + 1)
        lo = max(0, hi - capacity)

        # 调整断言范围
        assert 0 <= samples.min() <= hi
        assert lo <= samples.max() < hi

    # 测试 where
    # 情况 1：condition 是一维的，x 和 y 是标量
    assert torch.all(where(torch.tensor([False, True]), 7, 8) == torch.tensor([8, 7]))

    # 情况 2：condition 是一维的，x 是张量，y 是标量
    assert torch.all(where(torch.tensor([False, True, True]), torch.tensor([1, 2, 3]), 8) == torch.tensor([8, 2, 3]))

    # 情况 3：condition 是一维的，x 是标量，y 是张量
    assert torch.all(where(torch.tensor([False, False, True]), 8, torch.tensor([1, 2, 3])) == torch.tensor([1, 2, 8]))

    # 情况 4：condition 是二维的，x 和 y 是二维的
    condition = torch.tensor([[False, True], [True, False]])
    x = torch.tensor([[1, 2], [3, 4]])
    y = torch.tensor([[-1, -1], [-1, -1]])
    assert torch.all(where(condition, x, y) == torch.tensor([[-1, 2], [3, -1]]))

    # 情况 5：condition 是二维的，x 是二维的，y 是标量
    assert torch.all(where(torch.tensor([[False, True], [True, False]]), torch.tensor([[1, 2], [3, 4]]), -1) == torch.tensor([[-1, 2], [3, -1]]))
    
    # 测试 map_flat
    inputs = [2], [3, 5], [[7, 11], [13, 17]]
    inputs = [torch.tensor(i) for i in inputs]
    outputs = map_flat(lambda x: x ** 2, inputs)
    for i, o in zip(inputs, outputs):
        assert torch.all(i ** 2 == o)

    # 测试 map_flat_bits
    inputs = [2], [3, 5], [[7, 11], [13, 17]], [True, False, True]
    dtypes = torch.uint8, torch.int32, torch.int32, torch.int64, torch.bool
    inputs = [torch.tensor(i, dtype=d) for i, d in zip(inputs, dtypes)]
    outputs = map_flat_bits(lambda x: x + 1, inputs)
    for i, o in zip(inputs, outputs):
        assert torch.all(i + 1 == o)

    # 测试 cumulative_max
    np.random.seed(7)
    for x in [
            np.random.randn(10),
            np.random.randn(11, 7),
            np.random.randint(-10, 10, size=10),
            np.random.randint(-10, 10, size=(12, 8)),
            np.random.randint(-10, 10, size=(3, 3, 4)),
    ]:
        x = torch.tensor(x)
        assert torch.all(cumulative_max(x) == torch.cummax(x, dim=-1).values)

    # 测试 index_each
    np.random.seed(7)
    x = np.random.randn(7, 11)
    i = np.random.randint(x.shape[1], size=x.shape[0])
    y = index_each(torch.tensor(x), torch.tensor(i))
    assert torch.all(y == torch.tensor(x[np.arange(7), i]))

    # 测试 index_each_many
    x = np.random.randn(7, 11)
    i = np.random.randint(x.shape[1], size=[x.shape[0], 3])
    y = index_each_many(torch.tensor(x), torch.tensor(i))
    assert torch.all(y == torch.tensor(x[np.arange(7)[:, None], i]))

    # 测试 graph_function
    @graph_function(x=torch.int32, y=torch.int32)
    def tf_sub(x, y=1):
        return x - y

    assert tf_sub(3) == 2
    assert tf_sub(x=3) == 2
    assert tf_sub(5, 2) == 3
    assert tf_sub(y=2, x=5) == 3

    # 测试 top_k
    logits = torch.tensor([[[1, 1.01, 1.001, 0, 0, 0, 2]]], dtype=torch.float32)
    expected_output = torch.tensor([[[-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, 2]]], dtype=torch.float32)
    torch.allclose(take_top_k_logits(logits, 1), expected_output, atol=1e-5)

    expected_output = torch.tensor([[[-1e10, 1.01, -1e10, -1e10, -1e10, -1e10, 2]]], dtype=torch.float32)
    torch.allclose(take_top_k_logits(logits, 2), expected_output, atol=1e-5)

    expected_output = torch.tensor([[[-1e10, 1.01, 1.001, -1e10, -1e10, -1e10, 2]]], dtype=torch.float32)
    torch.allclose(take_top_k_logits(logits, 3), expected_output, atol=1e-5)

    expected_output = torch.tensor([[[1, 1.01, 1.001, -1e10, -1e10, -1e10, 2]]], dtype=torch.float32)
    torch.allclose(take_top_k_logits(logits, 4), expected_output, atol=1e-5)

    expected_output = torch.tensor([[[1, 1.01, 1.001, 0, 0, 0, 2]]], dtype=torch.float32)
    torch.allclose(take_top_k_logits(logits, 5), expected_output, atol=1e-5)
    
    # 测试 top_p
    logits = torch.tensor([[[1, 1.01, 1.001, 0, 0, 0, 2]]], dtype=torch.float32)
    assert torch.allclose(take_top_p_logits(logits, 1), logits)
    torch.allclose(take_top_p_logits(logits, 0), torch.tensor([[[-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, 2]]]))
    torch.allclose(take_top_p_logits(logits, 0.7), torch.tensor([[[-1e10, 1.01, 1.001, -1e10, -1e10, -1e10, 2]]]))
    torch.allclose(take_top_p_logits(logits, 0.6), torch.tensor([[[-1e10, 1.01, -1e10, -1e10, -1e10, -1e10, 2]]]))
    torch.allclose(take_top_p_logits(logits, 0.5), torch.tensor([[[-1e10, -1e10, -1e10, -1e10, -1e10, -1e10, 2]]]))

    # 测试 safe_zip
    assert list(safe_zip([1, 2], [3, 4])) == [(1, 3), (2, 4)]
    try:
        safe_zip([1, 2], [3, 4, 5])
        assert False
    except ValueError:
        pass