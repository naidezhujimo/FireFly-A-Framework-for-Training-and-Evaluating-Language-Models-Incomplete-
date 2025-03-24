import json
import random

from lm_human_preferences.utils import gcs

def books_generator(mode, seed=0, shuffle=False, comm=None):
    """
    生成器函数,用于随机生成书籍段落数据
    
    参数：
    - mode: 字符串,指定要加载的数据集模式(如'train', 'valid'等)
    - seed: 整数,随机种子,用于控制随机打乱的数据
    - shuffle: 是否打乱数据顺序
    - comm: 可选参数,用于分布式计算时的通信对象

    返回:
    - 生成器，每次迭代返回一个书籍段落的字典
    """

    # 从Google Cloud Storage下载指定模式的数据集文件,并缓存到本地
    # gcs.download_file_cached函数返回文件的路径,然后使用open函数打开文件
    datas = [
        json.loads(line) for line in open(gcs.download_file_cached(f'https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/book_passages/{mode}.jsonl', comm=comm))
    ]

    if shuffle: # 数据打乱
        random.seed(seed)
        random.shuffle(datas)

    for x in datas:
        yield x # 将每个数据项逐个返回
    