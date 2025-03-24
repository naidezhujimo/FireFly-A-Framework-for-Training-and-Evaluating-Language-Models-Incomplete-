import copy
import os
import torch
import torch.nn as nn

from lm_human_preferences.language import encodings, model

class TrainedModel:
    """
    name:模型的名称(例如 'test' 或 'main')
    savedir:模型权重和超参数文件的存储路径,如果未指定,则默认为云存储路径
    scope:模型的作用域(可选)
    """
    def __init__(self, name, *, savedir=None, scope=None):
        self.name = name
        self.scope = scope
        self.savedir = savedir if savedir else os.path.join('gs://gpt-2/models/', name)
        if name == 'test':
            self.encoding = encodings.Test
        else:
            self.encoding = encodings.Main
        self._hparams = None

    # 加载模型的检查点文件
    def checkpoint(self):
        if self.name == 'test':
            return None
        path = os.path.join(self.savedir, 'checkpoints', 'latest_checkpoint.pt')
        try:
            return torch.load(path)
        except FileNotFoundError:
            print(f"Checkpoint file {path} not found, returning None.")
            return None
        
    # 加载模型的超参数
    def hparams(self):
        if self._hparams is None:
            if self.name == 'test':
                hparams = test_hparams()
            else:
                hparams = load_hparams(
                    os.path.join(self.savedir, 'hparams.json')
                )
            self._hparams = hparams
        return copy.deepcopy(self._hparams)
    
    # 使用加载的检查点文件初始化模型参数
    def init_op(self, params, new_scope):
        assert params
        params = dict(**params)
        checkpoint = self.checkpoint()
        available = torch.load(checkpoint)
        unchanged = {}

        for name, shape in available.items():
            our_name = name
            if self.scope:
                if name.startswith(self.scope):
                    our_name = name[len(self.scope):].lstrip('/')
                else:
                    continue
            
            our_name = '%s/%s' % (new_scope, our_name)
            if our_name not in params:
                continue
            var = params[our_name]
            del params[our_name]
            assert var.shape == shape, 'Shape mismatch: %s.shape = %s != %s' % (var.op.name, var.shape, shape)
            unchanged[name] = var
            # 手动复制权重
            for name, var in unchanged.items():
                if name in available:
                    var.copy_(available[name])

#  从 JSON 文件加载超参数
def load_hparams(file):
    hparams = model.HParams()
    hparams.override_from_json_file(file)
    return hparams

# 定义测试模型的超参数
def test_hparams():
    hparams = model.HParams()
    hparams.override_from_dict(dict(
        n_vocab=27,
        n_ctx=8,
        n_layer=2,
        n_embd=7,
        n_head=1
    ))
    return hparams