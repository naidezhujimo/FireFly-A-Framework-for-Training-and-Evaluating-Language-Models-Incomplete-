import torch

def combos(*xs):
    """
    生成所有可能的组合
    :param xs: 输入的列表，每个元素是一个列表
    :return: 所有可能的组合
    """
    if xs:
        return [x + combo for x in xs[0] for combo in combos(*xs[1:])]
    else:
        return [()]
    
def each(*xs):
    """
    将多个列表展平为一个列表
    :param xs: 输入的列表
    :return: 展平后的列表"""
    return [y for x in xs for y in x]

def bind(var, val, descriptor=''):
    """
    绑定变量和值,并可以附加描述符
    :param var: 变量名
    :param val: 变量值
    :param descriptor: 描述符
    :return: 包含变量、值和描述符的元组列表"""
    extra = {}
    if descriptor:
        extra['descriptor'] = descriptor
    return [((var, val ,extra), )]

def label(descriptor):
    """
    生成一个标签
    :param descriptor: 描述符
    :return: 包含标签的元组列表"""
    return bind(None, None, descriptor)

def labels(*descriptors):
    """生成多个标签
    :param descriptor: 描述符列表
    :return: 包含多个标签的元组列表"""
    return each(*[label[d] for d in descriptors])

def options(var, opts_with_descs):
    """生成变量和选项的组合
    :param var: 变量名
    :param opts_with_descs: 选项和描述符的列表
    :return: 包含变量、选项和描述符的元组列表"""
    return each(*[bind(var, val, descriptor) for val, descriptor in opts_with_descs])

def options_shortdesc(var, desc, opts):
    """生成变量和选项的组合,并使用变量名作为描述符前缀
    :param var: 变量名
    :param desc: 描述符前缀
    :param opts: 选项列表
    :return: 包含变量、选项和简短描述符的元组列表"""
    return each(*[bind(var, val, desc + _shortstr(val)) for val in opts])

def options_vardesc(var, opts):
    """
    生成变量和选项的组合，并使用变量名作为描述符前缀
    :param var: 变量名
    :param opts: 选项列表
    :return: 包含变量、选项和描述符的元组列表"""
    return options_shortdesc(var, var, opts)

def repeat(n):
    """
    生成重复的标签
    :param n: 重复次数
    :return: 包含重复标签的元组列表
    """
    return each(*[label(i) for i in range(n)])

def foreach(inputs, body):
    """
    遍历输入并应用函数
    :param inputs: 输入列表
    :param body: 应用函数
    :return: 应用函数后的结果列表
    """
    return [inp + y for inp in inputs for y in body(*[extra['descriptor'] for var, val, extra in inp])]

def bind_nested(prefix, binds):
    """
    嵌套绑定变量
    :param prefix: 前缀
    :param binds: 绑定列表
    :return: 嵌套绑定后的元组列表
    """
    return [
        tuple([ (var if var is None else prefix + '.' + var, val, extra) for (var, val, extra) in x ])
        for x in binds
    ]

def _shortstr(v):
    """
    生成简短的字符串表示
    :param v: 值
    :return: 简短的字符串表示
    """
    if isinstance(v, float):
        s = f"{v:.03}"
        if '.' in s:
            s = s.lstrip('0').replace('.','x')
    else:
        s = str(v)
    return s