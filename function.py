import torch

# 计算输入的均值和方差
def calculation_mean_and_std(tensor,epsion=1e-6):
    # epsion是避免除以0添加的极小量
    # tensor的形状是[batch_size,channel,height,width]
    size = tensor.size()
    assert (len(size) == 4)
    N, C = size[:2]
    # 求出每个样本的均值和方差
    # view函数的作用是将一个tensor按照指定的形状进行重构
    tensor_var = tensor.view(N, C, -1).var(dim=2) + epsion
    tensor_std = tensor_var.sqrt().view(N, C, 1, 1)
    tensor_mean = tensor.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return tensor_mean, tensor_std


# Adaptive Instance Normalization
def AdaIN(content_tensor,style_tensor):
    # 断言两个tensor的形状相同
    assert (content_tensor.size()[:2] == style_tensor.size()[:2])
    # 获取content_tensor的形状
    size = content_tensor.size()
    # 计算content_tensor的均值和方差
    content_mean, content_std = calculation_mean_and_std(content_tensor)
    # 计算style_tensor的均值和方差
    style_mean, style_std = calculation_mean_and_std(style_tensor)
    # 标准化content_tensor
    # expand函数的作用是将一个tensor进行扩展
    # nromalized_tensor的形状是[batch_size,channel,height,width]，它是通过标准化content_tensor得到的，即减去均值后除以方差
    normalized_tensor = (content_tensor - content_mean.expand(size)) / content_std.expand(size)
    # 将标准化后的tensor乘以style_tensor的标准差并加上均值
    # 返回结果为经过AdaIN的tensor，包括了content_tensor的内容和style_tensor的风格
    return normalized_tensor * style_std.expand(size) + style_mean.expand(size)