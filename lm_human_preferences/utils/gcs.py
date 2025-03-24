"""这段代码的主要功能是与Google Cloud Storage (GCS) 进行交互
包括文件的上传、下载、缓存等操作。通过使用指数退避重试机制
代码能够在网络不稳定或服务不可用的情况下自动重试，提高了系统的鲁棒性
此外，代码还支持分布式环境下的缓存同步，确保多个进程能够访问到相同的缓存数据"""


import os  # 用于处理文件路径和目录操作
import random  # 用于生成随机数，用于重试机制中的抖动（jitter）
import subprocess # 用于调用外部命令，例如gsutil
import time  # 用于时间相关的操作
import traceback  # 用于打印异常堆栈信息
import warnings  # 用于忽略特定的警告信息
from functools import wraps # 用于装饰器，保留原函数的元数据
from urllib.parse import urlparse, unquote # 用于解析URL和处理URL编码
import requests #  用于处理HTTP请求
# 用于处理Google API的异常
from google.api_core.exceptions import InternalServerError, ServiceUnavailable
# 用于与Google Cloud Storage进行交互
from google.cloud import storage


# 忽略特定的警告信息
warnings.filterwarnings('ignore', 'Your application has authenticated using end user credentials')


def exponential_backoff(
        retry_on=lambda e: True, *, init_delay_s=1, max_delay_s=600, max_tries=30,
        factor=2.0, jitter=0.2, log_errors=True):
    """
    返回一个装饰器,当函数抛出异常且retry_on返回True时,会进行尝试

    参数：
    - init_delay_s: 第一次重试的等待时间(秒)
    - max_delay_s: 重试间隔时间的上限(秒)
    - max_tries: 最大重试次数
    - factor: 每次重试后延迟时间的增长因子
    - jitter: 抖动因子(0到1之间),每次延迟时间会乘以一个随机值(1-jitter到1+jitter之间)
    - log_errors: 是否在每次重试时打印错误消息
    - retry_on: 一个函数,接收异常并返回是否应该尝试
    """
    def decorate(f): # 定义装饰器
        @wraps(f) # 保留原函数的元数据
        def f_retry(*args, **kwargs): # 定义重试逻辑
            delay_s = float(init_delay_s)  # 初始化延迟时间
            for i in range(max_tries):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    if not retry_on(e) or i == max_tries-1: # 如果不应该重试或达到最大重试次数
                        raise
                    if log_errors:
                        print(f"Retrying after try {i+1}/{max_tries} failed:")  # 打印重试信息
                        traceback.print_exc()  # 打印异常堆栈
                    jittered_delay = random.uniform(delay_s*(1-jitter), delay_s*(1+jitter)) # 计算带抖动的延迟时间
                    time.sleep(jittered_delay)
                    delay_s = min(delay_s * factor, max_delay_s) # 更新延迟时间，不超过最大值
        return f_retry
    return decorate
    
def _gcs_should_retry_on(e):
    # 判断是否应该对某个异常进行重试
    # 这里主要针对GCS的503（服务不可用）和500（内部服务器错误）错误，以及网络连接错误
    return isinstance(e, (InternalServerError, ServiceUnavailable, requests.exceptions.ConnectionError))

def parse_url(url):
    '给定一个gs://路径,返回bucket名称和blob路径'
    result = urlparse(url) # 解析URL
    if result.scheme == 'gs': # 如果URL是gs://格式
        return result.netloc, unquote(result.path.lstrip('/')) # 返回bucket名称和blob路径
    elif result.scheme == 'https': # 如果URL是https://格式
        assert result.netloc == 'storage.googleapis.com' # 确保URL是GCS的URL
        bucket, rest = result.path.lstrip('/').split('/', 1) # 分割路径，获取bucket名称和blob路径
        return bucket, unquote(rest)
    else:
        raise Exception(f"Could not parse {url} as gcs url") # 抛出异常，无法解析URL
    

# 指数退避重试装饰器
@exponential_backoff(_gcs_should_retry_on)
def get_blob(url, client=None):
    if client is None: # 如果没有提供client
        client = storage.Client() # 创建GCS客户端
    bucket_name, path = parse_url(url) # 解析URL，获取bucket名称和blob路径
    bucket = client.get_bucket(bucket_name) # 获取bucket对象
    return bucket.get_blob(path) # 返回blob对象


@exponential_backoff(_gcs_should_retry_on)
def download_contents(url, client=None):
    # 给定一个gs://路径，返回对应blob的内容
    blob = get_blob(url, client) # 获取blob对象
    if not blob: return None # 如果blob不存在，返回None
    return blob.download_as_string() # 下载blob内容并返回


@exponential_backoff(_gcs_should_retry_on)
def upload_contents(url, contents, client=None):
    # 给定一个gs://路径，上传内容到对应的blob
    if client is None:
        client = storage.Client()
    bucket_name, path = parse_url(url)
    bucket = client.get_bucket(bucket_name)
    blob = storage.Blob(path, bucket)
    blob.upload_from_string(contents) # 上传内容


def download_directory_cached(url, comm=None):
    """给定一个GCS路径url,将内容缓存到本地
    警告：仅当路径下的内容不会更改时使用此函数！"""
    cache_dir = '/tmp/gcs_cache' # 缓存目录
    bucket_name, path = parse_url(url) # 解析URL，获取bucket名称blob路径
    is_master = not comm or comm.Get_rank() == 0 # 判断当前进程是否为主进程
    local_path = os.path.join(cache_dir, bucket_name, path) # 本地缓存路径
    sentinel = os.path.join(local_path, 'SYNCED') # 同步标记文件
    if is_master:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True) # 创建目录
            cmd = 'gsutil', '-m', 'cp', '-r', url, os.path.dirname(local_path) + '/' # 构建gsutil命令
            print(' '.join(cmd))
            subprocess.check_call(cmd) # 执行命令
            open(sentinel, 'a').close() # 创建同步标记文件
    else:
        while not os.path.exists(sentinel): # 等待同步标记文件出现
            time.sleep(1)
    return local_path # 返回本地缓存路径


def download_file_cached(url, comm=None):
    """给定一个GCS路径url,将内容缓存到本地
    警告：仅当路径下的内容不会更改时使用此函数！"""
    cache_dir = '/tmp/gcs_cache' 
    bucket_name, path = parse_url(url)
    is_master = not comm or comm.Get_rank() == 0
    local_path = os.path.join(cache_dir, bucket_name, path)
    sentinel = local_path + '.SYNCED' # 同步标记文件
    if is_master:
        if not os.path.exists(local_path):
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            cmd = 'gsutil', '-m', 'cp', url, local_path
            print(' '.join(cmd))
            subprocess.check_call(cmd)
            open(sentinel, 'a').close()
    else:
        while not os.path.exists(sentinel):
            time.sleep(1)
    return local_path