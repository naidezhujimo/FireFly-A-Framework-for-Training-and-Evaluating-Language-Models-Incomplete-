import json
import random
import re
import ftfy # 用于修复文本中的编码问题
 
from lm_human_preferences.utils import gcs

def tldr_generator(mode, seed=0, shuffle=False, comm=None):
    """
    生成器函数,用于生成TLDR(Too Long; Didn't Read)数据
    
    参数：
    - mode: 字符串,指定要加载的数据集模式(如'train', 'valid'等)
    - seed: 整数,随机种子,用于控制随机打乱的数据
    - shuffle: 是否打乱数据顺序
    - comm: 可选参数,用于分布式计算时的通信对象

    返回:
    - 生成器，每次迭代返回一个经过处理的文本内容
    """
    random.seed(seed)

    # 如果mode为'test'，则将其替换为'valid'，因为测试集不可用
    if mode == 'test':
        mode = 'valid'
    
    # 检查mode是否为'train'或'valid'，否则抛出异常
    assert mode in ['train', 'valid']

    # 从Google Cloud Storage下载指定模式的数据集文件，并缓存到本地
    with open(gcs.download_file_cached(f'https://openaipublic.blob.core.windows.net/lm-human-preferences/tldr/{mode}-subset.json', comm=comm)) as f:
        datas = json.load(f)  # 加载JSON文件内容并解析为Python对象（通常是列表或字典）

    if shuffle:
        random.seed(seed) # 重新设置随机数种子，确保打乱顺序的一致性
        random.shuffle(datas) # 打乱数据列表的顺序
    
    for data in datas:
        text = data['content'] # 获取原始文本内容
        text = ftfy.fix_text(text) # 使用ftfy修复文本中的编码问题（如乱码）
        text = re.sub(r"\n{3,}", "\n\n", text) # 使用正则表达式将连续3个及以上的换行符替换为2个换行符
        text = text.strip() # 去除文本首尾的空白字符
        yield text
