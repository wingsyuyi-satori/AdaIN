import torch.nn as nn
from function import AdaIN
from function import calculation_mean_and_std

# 定义decoder，使用reflection pad和upsample将图像进行还原
# 相较于转置卷积，该方法可以有效避免棋盘效应
decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode='nearest'),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)
# 提出vgg16的第四层，第九层，第十六层，第二十三层，第三十层的中间输出，其中前四个中间输出均作为风格迁移的风格特征，最后一个中间输出作为内容特征
class Net(nn.Module):
    def __init__(self,encoder,decoder):
        super(Net, self).__init__()
        # encoder: vgg16
        # decoder: 上面定义的decoder
        encode_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*encode_layers[:4])  # vgg前四层
        self.enc_2 = nn.Sequential(*encode_layers[4:9])  # vgg4-9
        self.enc_3 = nn.Sequential(*encode_layers[9:16])  # vgg9-16
        self.enc_4 = nn.Sequential(*encode_layers[16:23])  # vgg16-23
        self.decoder = decoder
        self.mse_loss = nn.MSELoss()
        # 将vgg相关1梯度置为False
        self.enc_1.requires_grad = False
        self.enc_2.requires_grad = False
        self.enc_3.requires_grad = False
        self.enc_4.requires_grad = False

    def encode_with_intermediate(self, input):
        # 对风格图像提取风格层的输出
        x=input
        results = []
        # 计算所有中间输出
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            x=func(x)
            results.append(x)
        return results

    def encode(self, input):
        # 对内容图像提取内容层的输出
        x=input
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i+1))
            x=func(x)
        return x


    def calc_content_loss(self, input, target):
        # 计算内容损失
        assert (input.size() == target.size())
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        # 计算风格损失
        assert (input.size() == target.size())
        input_mean, input_std = calculation_mean_and_std(input)
        target_mean, target_std = calculation_mean_and_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(input_std, target_std)

    def forward(self, content, style):
        # style为风格图像，content为内容图像，rate为内容图像的保留率
        # style_tensor的四个输出作为风格特征，最后一个中间输出作为内容特征
        style_tensor = self.encode_with_intermediate(style)
        content_tensor = self.encode(content)
        # 对内容图像和风格图像进行风格迁移，利用两个图像的内容特征得到风格迁移的结果
        # target为风格迁移后的图像
        target = AdaIN(content_tensor, style_tensor[-1]).detach()
        # 对输出进行decoder，得到最终的风格迁移图像
        generated_img = self.decoder(target)
        generated_img_tensor = self.encode_with_intermediate(generated_img)
        # 计算内容损失和风格损失
        loss_content = self.calc_content_loss(generated_img_tensor[-1], target)
        loss_style = self.calc_style_loss(generated_img_tensor[0], style_tensor[0])
        for i in range(1, 4):
            loss_style += self.calc_style_loss(generated_img_tensor[i], style_tensor[i])
        return loss_content, loss_style, generated_img




