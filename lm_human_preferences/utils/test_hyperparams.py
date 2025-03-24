from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional
import pytest

import os
import sys
# 获取当前脚本的目录
current_directory = os.path.dirname(os.path.abspath(__file__))
# 获取 FireFly 的路径
firefly_path = os.path.abspath(os.path.join(current_directory, '..', '..'))
sys.path.insert(0, firefly_path)  # 将 FireFly 路径添加到 sys.path
from lm_human_preferences.utils import hyperparams


@dataclass
class Hparams:
    # 超参数基类，提供从字符串、字典等覆盖超参数的功能
    def override_from_str(self, s: str):
        """
        从字符串覆盖超参数

        参数:
        - s (str): 格式为 "key1=value1,key2=value2" 的字符串
        """
        for kv in s.split(','):
            if not kv:
                continue
            key, value = kv.split('=')
            self._set_value(key, value)

    def override_from_dict(self, d: dict, separator='.'):
        """
        从字典覆盖超参数

        参数:
        - d (dict): 包含超参数键值对的字典
        - separator (str): 嵌套键的分隔符，默认为 '.'
        """
        for k, v in d.items():
            keys = k.split(separator)
            self._set_nested_value(keys, v)
    
    def override_from_nested_dict(self, d: dict):
        """
        从嵌套字典覆盖超参数

        参数:
        - d (dict): 嵌套字典
        """
        for k, v in d.items():
            if isinstance(v, dict):
                getattr(self, k).override_from_nested_dict(v)
            else:
                setattr(self, k, v)
                                
    def to_nested_dict(self):
        """
        将超参数转换为嵌套字典

        返回:
        - dict: 嵌套字典
        """
        result = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Hparams):
                result[k] = v.to_nested_dict()
            else:
                result[k] = v
        return result
    
    def validate(self):
        # 验证超参数是否有效
        for k, v in self.__dict__.items():
            if v is None and not k.startswith('optional_'):
                raise TypeError(f"Field '{k}' is mandatory but not set.")
            if isinstance(v, Hparams):
                v.validate()

    def _set_value(self, key: str, value: str):
        """
        设置单个超参数的值

        参数:
        - key (str): 超参数名称
        - value (str): 超参数值
        """
        if not hasattr(self, key):
            raise AttributeError(f"Field '{key}' does not exist.")
        current_value = getattr(self, key)
        if isinstance(current_value, bool):
            setattr(self, key, value.lower() == 'true')
        elif isinstance(current_value, (int, float)):
            setattr(self, key, type(current_value)(value))
        elif isinstance(current_value, str):
            setattr(self, key, value)
        elif isinstance(current_value, Hparams):
            if value.lower() in ('on', 'off'):
                setattr(self, key, Simple() if value.lower() == 'on' else None)
            else:
                raise ValueError(f"Invalid value '{value}' for nested field '{key}'.")
        else:
            raise ValueError(f"Cannot set value for field '{key}' of type '{type(current_value)}'.")
        
    def _set_nested_value(self, keys: list, value: str):
        """
        设置嵌套超参数的值

        参数:
        - keys (list): 嵌套键的列表
        - value (str): 超参数值
        """
        current = self
        for key in keys[:-1]:
            current = getattr(current, key)
        self._set_value(keys[-1], value)


@dataclass
class Simple(Hparams):
    # 简单的超参数类
    mandatory_nodefault: int = None
    mandatory_withdefault: str = "foo"
    optional_nodefault: Optional[int] = None
    fun: bool = True

def test_simple_works():
    # 测试 Simple 类的功能
    hp = Simple()
    hp.override_from_str("mandatory_nodefault=3,optional_nodefault=None,fun=false")
    hp.validate()
    assert hp.mandatory_nodefault == 3
    assert hp.mandatory_withdefault == "foo"
    assert hp.optional_nodefault is None
    assert not hp.fun


def test_simple_failures():
    # 测试 Simple 类的异常情况
    hp = Simple()
    with pytest.raises(TypeError):
        hp.validate()  # mandatory_nodefault 未设置
    with pytest.raises(ValueError):
        hp.override_from_str("mandatory_nodefault=abc")
    with pytest.raises(AttributeError):
        hp.override_from_str("nonexistent_field=7.0")
    with pytest.raises(ValueError):
        hp.override_from_str("fun=?")


@dataclass
class Nested(Hparams):
    # 嵌套的超参数类，包含 Simple 类的实例
    first: bool = False
    simpie_1: Simple = field(default_factory=Simple)
    simpie_2: Optional[Simple] = None


def test_nested():
    # 测试 Nested 类的功能
    hp = Nested()
    hp.override_from_str("simple_1.mandatory_nodefault=8,simple_2=on,simple_2.mandatory_withdefault=HELLO")
    with pytest.raises(TypeError):
        hp.validate()  # simple_2.mandatory_nodefault 未设置
    hp.override_from_dict({'simple_2/mandatory_nodefault': 7, 'simple_1/optional_nodefault': 55}, separator='/')
    hp.validate()
    assert hp.simple_1.mandatory_nodefault == 8
    assert hp.simple_1.mandatory_withdefault == "foo"
    assert hp.simple_1.optional_nodefault == 55
    assert hp.simple_2.mandatory_nodefault == 7
    assert hp.simple_2.mandatory_withdefault == "HELLO"
    assert hp.simple_2.optional_nodefault is None

    hp.override_from_str("simple_2=off")
    hp.validate()
    assert hp.simple_2 is None

    with pytest.raises((TypeError, AttributeError)):
        hp.override_from_str("simple_2.fun=True")
    with pytest.raises(ValueError):
        hp.override_from_str("simple_2=BADVAL")

def test_nested_dict():
    """
    测试从嵌套字典覆盖超参数的功能
    """
    hp = Nested()
    hp.override_from_nested_dict(
        {'simple_1': {'mandatory_nodefault': 8}, 'simple_2': {'mandatory_withdefault': "HELLO"}})
    with pytest.raises(TypeError):
        hp.validate()  # simple_2.mandatory_nodefault 未设置
    hp.override_from_nested_dict(
        {'simple_2': {'mandatory_nodefault': 7}, 'simple_1': {'optional_nodefault': 55}, 'first': True})
    hp.validate()
    assert hp.to_nested_dict() == {
        'first': True,
        'simple_1': {
            'mandatory_nodefault': 8,
            'mandatory_withdefault': "foo",
            'optional_nodefault': 55,
            'fun': True,
        },
        'simple_2': {
            'mandatory_nodefault': 7,
            'mandatory_withdefault': "HELLO",
            'optional_nodefault': None,
            'fun': True,
        },
    }


def test_nested_order():
    """
    测试嵌套超参数的设置顺序
    """
    hp = Nested()
    # 两种顺序都应该有效
    hp.override_from_str_dict(OrderedDict([('simple_2.fun', 'True'), ('simple_2', 'on')]))
    hp.override_from_str_dict(OrderedDict([('simple_2', 'on'), ('simple_2.fun', 'True')]))


@dataclass
class Deeply(Hparams):
    # 深度嵌套的超参数类，包含 Nested 类的实例
    nested: Nested = None


def test_deeply_nested():
    """
    测试 Deeply 类的功能。
    """
    hp = Deeply()
    hp.override_from_str("nested.simple_2=on")
    assert hp.nested is not None
    assert hp.nested.simple_2 is not None

    hp = Deeply()
    hp.override_from_dict({'nested.simple_2': 'on'})
    assert hp.nested is not None
    assert hp.nested.simple_2 is not None


def test_set_order():
    """
    测试超参数的设置顺序。
    """
    hp = Deeply()
    hp.override_from_dict(OrderedDict([('nested.first', True), ('nested.simple_1', 'on')]))
    assert hp.nested.first is True

    hp = Deeply()
    hp.override_from_dict(OrderedDict([('nested.simple_1', 'on'), ('nested.first', True)]))
    assert hp.nested.first is True