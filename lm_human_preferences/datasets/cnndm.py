"""处理和生成CNN/DailyMail数据集的内容
CNN/DailyMail是一个广泛用于文本摘要任务的新闻数据集
代码的功能包括读取文本文件、修复缺失的句号、提取文章和摘要、
生成哈希值、清理文本内容以及生成数据集的内容"""


import hashlib # 用于生成哈希值
import os
import random
import re
import ftfy

from ..utils import gcs

dm_single_close_quote = u'\u2019' # Unicode中的单引号
dm_double_close_quote = u'\u201d' # Unicode中的双引号
END_TOKENS = ['.', '!', '?', '...', "'", "`", '"', dm_single_close_quote, dm_double_close_quote, ")"]  # 句子结束的符号列表

def read_text_file(text_file):
    lines = []
    with open(text_file, 'r')as f:
        for line in f:
            lines.append(line.strip())
    return lines

def fix_missing_period(line):
    # 如果句子没有以结束符号结尾，则添加句号
    if '@highlight' in line:  # 如果是摘要行，直接返回
        return line
    if line == '':  # 如果是空行，直接返回
        return line
    if line[-1] in END_TOKENS:  # 如果句子已经以结束符号结尾，直接返回
        return line
    return line + '.'  # 否则添加句号

def get_art_abs(story_file):
    """从故事文件中提取文章内容和摘要"""
    lines = read_text_file(story_file)
    article_lines = []  # 存储文章内容
    highlights = []  # 存储摘要内容
    next_is_highlight = False  # 标记下一行是否是摘要
    for line in lines:
        if line == '':
            continue
        elif line.startswith('@highlight'):  # 如果是摘要标记行
            next_is_highlight = True  # 标记下一行是摘要
        elif next_is_highlight:  # 如果下一行是摘要
            highlights.append(line)  # 添加到摘要列表
        else:
            article_lines.append(line)  # 添加到文章列表
    article = '\n\n'.join(article_lines)  # 将文章内容拼接成一个字符串
    highlights = [fix_missing_period(sent) for sent in highlights]  # 修复摘要中的缺失句号
    return article, highlights  # 返回文章和摘要

def hashhex(s):
    """返回输入字符串的SHA1哈希值的十六进制表示"""
    h = hashlib.sha1()  # 创建SHA1哈希对象
    h.update(s)  # 更新哈希值
    return h.hexdigest()  # 返回十六进制表示的哈希值

def get_path_of_url(url):
    """根据URL生成文件路径"""
    if 'dailymai.co.uk' in url or 'mailonsunday.ie' in url or 'lib.store.yahoo.net' in url: # 判断URL来源
        site = 'dailymail'
    else:
        assert 'cnn.com' in url or 'cnn.hk' in url, url # 确保URL是CNN的
        site = 'cnn' 
    url_hash = hashhex(url.encode('utf-8')) # 对URL进行哈希
    return f'{site}/stories/{url_hash}.story'  # 返回文件路径

def clean_up_start(text):
    """清理文本开头的不必要内容"""
    if text[:2] == 'By':
        text = '\n'.join(text.split('\n')[2:])
    text = re.split(r'\(CNN\) +--', text)[-1]  # 去掉CNN的标记
    text = re.split(r"\(CNN\)", text[:100])[-1] + text[100:]  # 去掉CNN的标记
    text = re.sub(r"^and \w+\n", "", text)  # 去掉以“and”开头的行
    text = re.split(r".*UPDATED:\s+[0-9]{2}:[0-9]{2}.*(2011|2012|2013|2014|2015)", text)[-1] # 去掉更新时间
    text = text.replace('’', "'")  # 替换Unicode单引号
    text = text.replace('‘', "'")  # 替换Unicode单引号
    return text.strip()  # 返回清理后的文本

def cnndm_generator(mode, seed=0, shuffle=False, comm=None):
    """生成CNN/DailyMail数据集的内容"""
    if mode == 'valid':  # 如果模式是验证集
        mode = 'val'  # 改为val
    # 下载并读取URL列表文件
    with open(gcs.download_file_cached(f'https://openaipublic.blob.core.windows.net/lm-human-preferences/datasets/cnndm/url_lists/all_{mode}.txt', comm=comm)) as f:
        urls = [line.strip() for line in f]  # 读取每一行的URL
        f.close()
    if shuffle:  # 如果需要打乱顺序
        random.seed(seed)  # 设置随机种子
        random.shuffle(urls)  # 打乱URL列表

    # 下载并缓存数据集目录
    urls_dir = gcs.download_directory_cached(f'gs://lm-human-preferences/datasets/cnndm/cache_{mode}', comm=comm)

    for i, url in enumerate(urls):  # 遍历每个URL
        path = os.path.join(urls_dir, get_path_of_url(url))  # 获取文件路径
        text = open(path).read()  # 读取文件内容
        text = clean_up_start(text)  # 清理文本开头
        text = ftfy.fix_text(text)  # 修复Unicode错误
        text = re.sub(r"\n{3,}", "\n\n", text)  # 将多个空行替换为两个空行
        text = text.split('@highlight')[0].strip()  # 去掉摘要部分
        yield text  # 返回清理后的文本