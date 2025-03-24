import json
import os
from functools import lru_cache
import re
import torch

@lru_cache()
# 创建一个从字节到 Unicode 字符的映射表，以便在 BPE 过程中使用
def bytes_to_unicode():
    """
    返回一个 UTF-8 字节和对应的 Unicode 字符串的列表
    BPE 编码工作在 Unicode 字符串上，因此需要大量的 Unicode 字符来避免未知字符(UNK)
    """
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2 ** 8):
        if b not in bs:
            bs.append(b)
            cs.append(2 ** 8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))

# 用于获取一个单词中的所有符号对，这些符号对将用于 BPE 合并
def get_pairs(word):
    """返回单词中的符号对集合"""
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs

# 实现了 BPE 编码和解码的核心逻辑
class ReversibleEncoder:
    def __init__(self, encoder, bpe_merges, errors='replace', eot_token=None):
        self.encoder = encoder # 编码器，将字符映射到索引
        self.decoder = {v: k for k, v in self.encoder.items()} # 解码器，将索引映射回字符
        self.errors = errors # 解码时的错误处理方式
        self.byte_encoder = bytes_to_unicode() # 字节到 Unicode 的映射
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()} # Unicode 到字节的映射
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges)))) # BPE 合并的优先级
        self.eot_token = eot_token # 结束符
        self.cache = {} # 缓存，用于存储已经处理过的 token
        self.padding_token = len(encoder) + 2 # 填充符
        self.decoder[self.padding_token] = '' # 解码时填充符映射为空字符串

        # 正则表达式，用于将文本分割成 token
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        """对单个 token 进行 BPE 编码"""
        if token in self.cache:
            return self.cache[token]
        word = tuple(token)
        pairs = get_pairs(word)

        if not pairs:
            return token
        

        """
        如果 token 已经在缓存中，则直接返回
        提取 token 中的所有字符对
        按照 BPE 合并规则逐步合并字符对，直到无法继续合并
        将合并后的 token 存入缓存并返回
        """
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                    new_word.extend(word[i:j])
                    i = j
                except:
                    new_word.extend(word[i:])
                    break

                if word[i] == first and i < len(word) - 1 and word[i+1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = ' '.join(word)
        self.cache[token] = word
        return word
    
    """
    使用正则表达式将文本分割成 token
    将每个 token 转换为 Unicode 字符序列
    对每个 token 应用 BPE 编码
    将编码后的 token 转换为索引
    """
    def encode(self, text):
        """将文本编码为 BPE token"""
        bpe_tokens = []
        for token in re.findall(self.pat, text):
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            bpe_tokens.extend(self.encoder[bpe_tokens] for bpe_token in self.bpe(token).split(' '))
            return bpe_tokens
        

    """
    使用正则表达式将文本分割成 token
    将每个 token 转换为 Unicode 字符序列
    对每个 token 应用 BPE 编码
    将编码后的 token 转换为索引
    """
    def decode(self, tokens, pretty=False):
        """将 BPE token 解码为文本"""
        del pretty
        text = ''.join([self.decoder[token] for token in tokens])
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text
    
def read_file(path):
    """读取文件内容。"""
    with open(path, "rb") as fh:
        return fh.read()
    
class Encoding:
    """ 
    name:编码器的名称
    n_vocab:词汇表大小
    eot_token:结束符的索引
    encoder_path 和 bpe_path:编码器和 BPE 合并规则的文件路径
    base_path:基础路径
    """
    def __init__(self,
            name,
            *,
            n_vocab=0,
            eot_token=None,
            encoder_path='encoder.json',
            bpe_path='vocab.bpe',
            base_path=None
    ):
        self.name = name
        self.eot_token = eot_token
        self.n_vocab = n_vocab

        if base_path is None:
            base_path = os.path.join("gpt-2/encodings", name)
        
        self.base_path = base_path
        if name != 'test':
            self.encoder_path = os.path.join(self.base_path, encoder_path)
            self.bpe_path = os.path.join(self.base_path, bpe_path)

    """
    如果是测试模式(name == "test")，返回一个简单的测试编码器
    否则，从文件加载编码器和 BPE 合并规则
    创建 ReversibleEncoder 实例并返回
    """
    def get_encoder(self):
        """获取编码器"""
        if self.name == "test":
            vocab = "abcdefghijklmnopqrstuvwxyz."
            assert len(vocab) == self.n_vocab

            class TestEncoder(ReversibleEncoder):
                def __init__(self):
                    super().__init__(encoder={w: i for i, w in enumerate(vocab)}, bpe_merges=list())
                    self.padding_token = len(vocab)
                def encode(self, text):
                    return [self.encoder.get(x, len(vocab) - 1) for x in text]
                def decode(self, tokens, pretty=False):
                    return ''.join([self.decoder.get(t, '<unk>') for t in tokens])
                
            return TestEncoder()
        
        encoder_dict = json.loads(read_file(self.encoder_path).decode())
        bpe_data = read_file(self.bpe_path).decode()
        bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
        assert len(encoder_dict) == self.n_vocab
        encoder = ReversibleEncoder(encoder=encoder_dict, bpe_merges=bpe_merges, eot_token=self.eot_token)
        assert encoder.padding_token >= self.n_vocab
        return encoder
    

Main = Encoding('main', n_vocab=50257, eot_token=50256)
Test = Encoding('test', n_vocab=27, eot_token=26)