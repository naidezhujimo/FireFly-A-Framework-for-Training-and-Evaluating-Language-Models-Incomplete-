import os
import subprocess
from functools import partial
import cloudpickle
import fire
import torch.multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor

def launch(name, f, *, namespace='safety', mode='local', mpi=1) -> None:
    """
    启动任务，支持本地模式和 MPI 并行模式
    :param name: 任务名称
    :param f: 要执行的函数
    :param namespace: 命名空间(未使用)
    :param mode: 执行模式('local' 或 'mpi')
    :param mpi: MPI 进程数
    """
    if mode == 'local':
        # 将函数序列化并保存到临时文件
        with open('/tmp/pickle_fn', 'wb') as file:
            cloudpickle.dump(f, file)
        
        # 使用 MPI 启动任务
        subprocess.check_call(['mpiexec', '-n', str(mpi), 'python', '-c', 
                              'import sys; import pickle; pickle.loads(open("/tmp/pickle_fn", "rb").read())()'])
        return
    elif mode == 'torch_mp': # 使用 Pytorch 的多进程模式
        processes = []
        for _ in range(mpi):
            p = mp.Process(target=f)
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
    else:
        raise Exception('Other modes unimplemented!')
    
def parallel(jobs, mode):
    """
    并行执行任务
    :param jobs: 任务列表
    :param mode: 执行模式('local' 或 'torch_mp')
    """
    if mode == 'local':
        assert len(jobs) == 1, "Cannot run jobs in parallel locally"
        for job in jobs:
            job()
    elif mode == 'torch_mp':  # 使用 PyTorch 的多进程模式
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(job) for job in jobs]
            for f in futures:
                f.result()
    else:
        raise Exception(f'Unsupported mode: {mode}')
    
def launch_trials(name, fn, trials, hparam_class, extra_hparams=None, dry_run=False, mpi=1, mode='local', save_dir=None):
    """
    启动多个试验任务
    :param name: 任务名称
    :param fn: 任务函数
    :param trials: 试验参数列表
    :param hparam_class: 超参数类
    :param extra_hparams: 额外的超参数
    :param dry_run: 是否仅打印任务信息而不执行
    :param mpi: MPI 进程数
    :param mode: 执行模式('local' 或 'torch_mp')
    :param save_dir: 保存结果的目录
    """
    jobs = []
    for trial in trials:
        descriptors = []
        kwargs = {}
        for k, v, s in trial:
            if k is not None:
                if k in kwargs:
                    print(f'WARNING: overriding key {k} from {kwargs[k]} to {v}')
                kwargs[k] = v
            if s.get('descriptor'):
                descriptors.append(str(s['descriptor']))
        hparams = hparam_class()
        hparams.override_from_dict(kwargs)
        if extra_hparams:
            hparams.override_from_str_dict(extra_hparams)
        job_name = (name + '/' + '-'.join(descriptors)).rstrip('/')
        hparams.validate()
        if dry_run:
            print(f"{job_name}: {kwargs}")
        else:
            if save_dir:
                hparams.run.save_dir = os.path.join(save_dir, job_name)
            trial_fn = partial(fn, hparams)
            jobs.append(partial(launch, job_name, trial_fn, mpi=mpi, mode=mode))
    parallel(jobs, mode=mode)

def main(commands_dict):
    class _Commands:
        def __init__(self):
            for name, cmd in commands_dict.items():
                setattr(self, name, cmd)
    fire.Fire(_Commands)
        