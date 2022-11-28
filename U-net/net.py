import torch
from torch import nn
from torch.nn import functional
import yaml
import warnings
# 此处搭建模型


# 计算填充值，采用same
def autopad(kernel_size):
    return (kernel_size-1)//2

# 声明Conv2D
class Conv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, bias=True, act=None):
        super(Conv2D, self).__init__()
        self.name = 'Conv2D'
        padding = padding if padding else autopad(kernel_size)
        act = act if act else nn.ReLU()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            act
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# 构建所需的双重Conv2D结构
class Double_Conv2D(nn.Module):
    def __init__(self, in_channels=3, out_channels=None, kernel_size=3, stride=1, padding=1, activation=None):
        super(Double_Conv2D, self).__init__()
        self.name = 'Double_Conv2D'
        # 计算填充层数
        padding = padding if padding else autopad(kernel_size)
        # 默认激活函数Relu
        default_act = nn.ReLU()
        # 两个卷积层，未指定激活函数
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        # print(self.name, x.size())
        return x


# 池化层，用作下采样
class MaxPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2):
        super(MaxPooling, self).__init__()
        self.name = 'MaxPooling'
        self.maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        x = self.maxpool(x)
        # print(self.name, x.size())
        return x


# 上采样及特征融合，Concat层
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=None, scale_factor=2.0, bilinear=True):
        super(Up, self).__init__()
        self.name = 'Up'
        self.kernel_size = kernel_size

        # 使用线性插值，特征图尺寸放大2倍
        if bilinear:
            self.up = nn.Sequential(nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=True),
                                    Conv2D(in_channels,
                                           out_channels,
                                           kernel_size=kernel_size,
                                           stride=stride
                                          )
                                    )

        else:
            # 使用反卷积上采样
            self.up = nn.ConvTranspose2d(in_channels,
                                         out_channels//2,
                                         kernel_size=2,
                                         stride=2
                                        )

    # x是上一层输入，y是之前层的输出
    def forward(self, x, y):
        x = self.up(x)

        # 计算上采样后的x与之前层的尺寸差距
        # diffh - 高度差距   diffw - 宽度差距
        diffh = torch.tensor(y.size()[2] - x.size()[2])
        diffw = torch.tensor(y.size()[3] - x.size()[3])
        # 忽略警告
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 使用pad将x补全到之前层尺寸
            # nn.functional.pad(torch, (left_weight, right_weight, up_height, down_height))
            # 即上下左右补多少层0
            x = nn.functional.pad(x, (diffw // 2, diffw - diffw // 2, diffh // 2, diffh - diffh // 2))

        # 使用cat在通道维度上融合
        out = torch.cat([x, y], dim=1)
        # print(self.name, x.size(), y.size(), '==>', out.size())
        return out


# Dropout层
class Dropout(nn.Module):
    def __init__(self, rate):
        super(Dropout, self).__init__()
        self.name = 'Dropout'
        self.drop = nn.Dropout(rate)

    def forward(self, x):
        x = self.drop(x)
        # print(self.name, x.size())
        return x


# 整体网络构建
class Unet(nn.Module):
    def __init__(self, cfg=None, in_channels=3, out_channels=1):
        super(Unet, self).__init__()

        # 加载网络结构
        if isinstance(cfg, dict):
            pass
        else:
            # 默认搭建Unet网络
            # 各层输出通道
            n = 64
            filters = [n, n * 2, n * 4, n * 8, n * 16]
            # 要获取初始双重卷积的特征图，需要分开写
            self.conv1 = Double_Conv2D(in_channels, filters[0])
            self.down1 = MaxPooling()

            self.conv2 = Double_Conv2D(filters[0], filters[1])
            self.down2 = MaxPooling()

            self.dropout1 = nn.Dropout(0.2)

            self.conv3 = Double_Conv2D(filters[1], filters[2])
            self.down3 = MaxPooling()

            self.conv4 = Double_Conv2D(filters[2], filters[3])
            self.down4 = MaxPooling()

            self.dropout2 = nn.Dropout(0.2)

            self.conv5 = Double_Conv2D(filters[3], filters[4])

            self.Up1 = Up(filters[4], filters[3])
            self.Up_conv1 = Double_Conv2D(filters[4], filters[3])

            self.Up2 = Up(filters[3], filters[2])
            self.Up_conv2 = Double_Conv2D(filters[3], filters[2])

            self.Up3 = Up(filters[2], filters[1])
            self.Up_conv3 = Double_Conv2D(filters[2], filters[1])

            self.Up4 = Up(filters[1], filters[0])
            self.Up_conv4 = Double_Conv2D(filters[1], filters[0])

            self.conv6 = Conv2D(filters[0], out_channels, kernel_size=1, stride=1)

            self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encoding部分
        conv1 = self.conv1(x)
        down1 = self.down1(conv1)

        conv2 = self.conv2(down1)
        down2 = self.down2(conv2)

        drop1 = self.dropout1(down2)

        conv3 = self.conv3(drop1)
        down3 = self.down3(conv3)

        conv4 = self.conv4(down3)
        down4 = self.down4(conv4)

        drop2 = self.dropout2(down4)

        medium = self.conv5(drop2)
        # Decoding部分
        up1 = self.Up1(medium, conv4)
        up_conv1 = self.Up_conv1(up1)

        up2 = self.Up2(up_conv1, conv3)
        up_conv2 = self.Up_conv2(up2)

        up3 = self.Up3(up_conv2, conv2)
        up_conv3 = self.Up_conv3(up3)

        up4 = self.Up4(up_conv3, conv1)
        up_conv4 = self.Up_conv4(up4)

        # 转到输出层做1*1卷积输出单通道
        out = self.conv6(up_conv4)

        return self.sigmoid(out)


if __name__ == '__main__':
    cfg_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data\models\Unet.yaml'
    model = Unet()
    print(model)


