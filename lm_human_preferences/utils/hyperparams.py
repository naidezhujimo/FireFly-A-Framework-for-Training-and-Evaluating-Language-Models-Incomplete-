import json
import sys
import typing
from dataclasses import fields, is_dataclass
from functools import lru_cache
from typeguard import check_type
from dataclasses import dataclass

import os
import sys
# 获取当前脚本的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取 FireFly 的路径
firefly_path = os.path.abspath(os.path.join(current_directory, '..', '..'))
sys.path.insert(0, firefly_path)  # 将 FireFly 路径添加到 sys.path
from lm_human_preferences.utils import gcs


@dataclass
class HParams:
    """用于管理超参数的基类,需要与 @dataclass 一起使用"""
    def override_from_json_file(self, filename):
        """从 JSON 文件中加载并覆盖超参数"""
        if filename.startswith('gs://'):
            # 如果是 Google Cloud Storage 路径，下载内容
            hparams_str = gcs.download_contents(filename)
        else:
            # 否则从本地文件读取
            with open(filename, 'r') as f:
                hparams_str = f.read()
        self.parse_json(hparams_str)

    def override_from_str(self, hparams_str):
        """从字符串中加载并覆盖超参数，字符串格式为 'x.y=1,name=foobar'"""
        kvp_strs = hparams_str.split(',')
        flat_dict = {}
        for kvp_str in kvp_strs:
            k, sep, v = kvp_str.partition('=')
            if not sep:
                raise ValueError(f"格式错误的超参数值: '{kvp_str}'")
            flat_dict[k] = v
        self.override_from_str_dict(flat_dict)
    
    def override_from_str_dict(self, flat_dict, separator='.'):
        """从字典中加载并覆盖超参数，字典格式为 {'x.y': "1", 'name': "foobar"}"""
        typemap = _type_map(type(self), separator=separator)
        flat_dict_parsed = {}
        for flat_k, v in flat_dict.items():
            cls = _type_to_class(typemap[flat_k])
            if is_hparam_type(cls) and v == 'on':
                parsed_v = cls()
            if is_hparam_type(cls) and v == 'off':
                parsed_v = None
            else:
                parsed_v = v
            flat_dict_parsed[flat_k] = parsed_v

    def parse_json(self, s:str):
        """从 JSON 字符串中加载并覆盖超参数"""
        self.override_from_nested_dict(json.loads(s))
    
    def override_from_dict(self, flat_dict, separator='.'):
        """从字典中加载并覆盖超参数，字典格式为 {'x.y': 1, 'name': "foobar"}"""
        typemap = _type_map(type(self), separator=separator)
        flat_dict_parsed = {}
        for flat_k, v in flat_dict.items():
            cls = _type_to_class(typemap[flat_k])
            if is_hparam_type(cls) and v == 'on':
                parsed_v = cls()
            elif is_hparam_type(cls) and v == 'off':
                parsed_v = None
            else:
                parsed_v = v
            flat_dict_parsed[flat_k] = parsed_v

        # 扩展隐式嵌套的 'on' 值
        flat_dict_expanded = {}
        for flat_k, v in flat_dict_parsed.items():
            flat_dict_expanded[flat_k] = v
            cls = _type_to_class(typemap[flat_k])
            if is_hparam_type(cls) and v is not None:
                parts = flat_k.split(separator)
                prefix = parts[0]
                for i in range(1, len(parts)):
                    if prefix not in flat_dict_expanded:
                        flat_dict_expanded[prefix] = _type_to_class(typemap[prefix])()
                    prefix += separator + parts[i]
        
        # 设置所有值
        for flat_k in sorted(flat_dict_expanded.keys()):
            v = flat_dict_expanded(flat_k)
            *ks, f = flat_k.split(separator)
            hp = self
            for i, k in enumerate(ks):
                try:
                    hp = getattr(hp, k)
                except AttributeError:
                    raise AttributeError(f"{hp} ({separator.join(ks)}) 没有字段 '{f}'")
            
    def override_from_nested_dict(self, nested_dict):
        """从嵌套字典中加载并覆盖超参数"""
        for k, v in nested_dict.items():
            if isinstance(v, dict):
                if getattr(self, k) is None:
                    cls = _type_to_class(_get_field(self, k).type)
                    setattr(self, k, cls())
                getattr(self, k).override_from_nested_dict(v)
            else:
                getattr(self, k, v)
    
    def to_nested_dict(self):
        """将超参数转换为嵌套字典"""
        d = {}
        for f in fields(self):
            fieldval = getattr(self, f.name)
            if isinstance(fieldval, HParams):
                fieldval = fieldval.to_nested_dict()
            d[f.name] = fieldval
        return d

    def validate(self, *, prefix=''):
        """验证超参数的类型是否正确"""
        assert is_dataclass(self), f"忘记用 @dataclass 注解 {type(self)}"
        for f in fields(self):
            fieldval = getattr(self, f.name)
            check_type(prefix + f.name, fieldval, f.type)
            if isinstance(fieldval, HParams):
                fieldval.validate(prefix=prefix + f.name + '.')

def is_hparam_type(ty):
    """检查类型是否为 HParams 类型"""
    if isinstance(ty, type) and issubclass(ty, HParams):
        assert is_dataclass(ty)
        return True
    else:
        return False
    
def _is_union_type(ty):
    """检查类型是否为union类型"""
    return getattr(ty, '__origin__', None) is typing.Union

def dump(hparams, *, name='hparams', out=sys.stdout):
    """打印超参数"""
    out.write(f'{name}:\n')
    def dump_nested(hp, indent):
        for f in sorted(fields(hp), key=lambda f: f.name):
            v = getattr(hp, f.name)
            if isinstance(v, HParams):
                out.write(f'{indent}{f.name}:\n')
                dump_nested(v, indent=indent+'  ')
            else:
                out.write(f'{indent}{f.name}: {v}\n')
    dump_nested(hparams, indent='  ')

def _can_distinguish_unambiguously(type_set):
    """检查是否可以明确区分类型集中的类型"""
    if len(type_set) == 1:
        return True
    if type(None) in type_set:
        return True
    if str in type_set:
        return False
    if int in type_set and float in type_set:
        return False
    if any(_is_union_type(ty) for ty in type_set):
        return False
    return True

def _parse_typed_value(ty, s):
    """根据类型解析字符串值"""
    if ty is str:
        return s
    elif ty in (int, float):
        return ty(s)
    elif ty is bool:
        if s in ('t', 'true', 'True'):
            return True
        elif s in ('f', 'false', 'False'):
            return False
        else:
            raise ValueError(f"无效的布尔值 '{s}'")
    elif ty is type(None):
        if s in ('None', 'none', ''):
            return None
        else:
            raise ValueError(f"无效的 None 值 '{s}'")
    elif is_hparam_type(ty):
        if s in ('on', 'off'):
            return s
        else:
            raise ValueError(f"无效的 hparam 类值 '{s}'")
    elif _is_union_type(ty):
        if not _can_distinguish_unambiguously(ty.__args__):
            raise TypeError(f"无法明确解析联合类型 '{ty}' 的值")
        for ty_option in ty.__args__:
            try:
                return _parse_typed_value(ty_option, s)
            except ValueError:
                continue
        raise ValueError(f"无法将 '{s}' 解析为 '{ty}' 中的任何类型")
    else:
        raise ValueError(f"不支持的 hparam 类型 '{ty}'")

def _get_field(data, fieldname):
    """获取字段对象"""
    matching_fields = [f for f in fields(data) if f.name == fieldname]
    if len(matching_fields) != 1:
        raise AttributeError(f"找不到字段 '{fieldname}'")
    return matching_fields[0]

def _update_disjoint(dst: dict, src: dict):
    """将 src 中的键值对更新到 dst 中，确保键不重复"""
    for k, v in src.items():
        assert k not in dst
        dst[k] = v

@lru_cache()
def _type_map(ty, separator):
    """生成类型映射表"""
    typemap = {}
    for f in fields(ty):
        typemap[f.name] = f.type
        if is_hparam_type(f.type):
            nested = _type_map(f.type, separator=separator)
        elif _is_union_type(f.type):
            nested = {}
            for ty_option in f.type.__args__:
                if is_hparam_type(ty_option):
                    _update_disjoint(nested, _type_map(ty_option, separator=separator))
        else:
            nested = {}
        _update_disjoint(typemap, {f'{f.name}{separator}{k}': t for k, t in nested.items()})
    return typemap

def _type_to_class(ty):
    """从类型中提取可构造的类"""
    if _is_union_type(ty):
        assert len(ty.__args__) == 2
        assert ty.__args__[1] is type(None)
        return ty.__args__[0]
    else:
        return ty