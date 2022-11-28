# 训练代码

# 文件操作库
import os.path
# 网络相关库
import torch
from net import *
# 图像处理库
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
# 数据加载库
from torch.utils.data import DataLoader
from utils.data_loading import MyDatasets
# 损失函数库
# from utils.losses import dice_loss
# 进度条
from tqdm import *

data_root = r'C:\Users\13632\Desktop\DRIVE'
info_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data\retinal.xlsx'

models_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data\models\Unet.yaml'
weight_and_bias = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data\models\MyUnet.pth'

save_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data'
train_vis_filename = 'train_vis'

save_option = False
train_vis_opt = False

leraning_rate = 1e-4
weight_decay = 1e-8

MyDataset = MyDatasets(data_root, info_path, train=1, save_name='retinal')

if __name__ == '__main__':

    # 创建保存目录
    train_vis_path = os.path.join(save_path, train_vis_filename)
    if not os.path.exists(train_vis_path):
        os.makedirs(train_vis_path)

    # 实例化网络模型
    net = Unet()

    # 读取参数文件
    if os.path.exists(weight_and_bias):
        state_dict = torch.load(weight_and_bias)
        net.load_state_dict(state_dict['model'])
        print('Successfully loaded parameters!')
    else:
        pass

    # 设定设备
    device = torch.device('gpu' if torch.cuda.is_available() else 'cpu')
    net = net.to(device)
    # net.summary()
    # exit(0)

    # 加载模型
    # 使用DataLoader必须使用python 3.9及以下版本，3.10会报错。。。。。
    train_datagenerator = DataLoader(MyDataset, batch_size=1, shuffle=False)

    # 优化器，采用Adam优化方法
    opt = torch.optim.Adam(net.parameters(), lr=leraning_rate, weight_decay=weight_decay)
    # 损失函数，交叉熵
    criterion = nn.BCELoss()
    # 训练次数
    epochs = 5

    for epoch in range(epochs):
        net.train()
        loss = 0
        with tqdm(total=len(train_datagenerator)) as pbar:
            for i, (img, mask, name) in enumerate(train_datagenerator):
                # 进度条左边信息
                pbar.set_description('epoch-{}'.format(epoch+1))

                # 将数据放入设备
                image = img.to(device=device, dtype=torch.float)
                real_mask = mask.to(device=device, dtype=torch.float)

                # 前向传播
                output = net(img).float()

                # 计算损失，dice_loss
                # loss = criterion(output, real_mask) + dice_loss(output, real_mask.float())
                loss = criterion(output, real_mask)

                # 配置进度条右边信息
                pbar.set_postfix(loss=loss.item())

                # 打印信息
                # if (i+5) % 5 == 0:
                #    print('epoch:{}--{}/{}--loss:{}'.format(epoch + 1, i+1, len(train_datagenerator), loss.item()))

                # 梯度归零
                opt.zero_grad()

                # 计算梯度
                loss.backward()
                opt.step()

                # 保存图片
                if train_vis_opt:
                    output[output > 0.5] = 255
                    output[output <= 0.5] = 0
                    # 放大回原图大小
                    # reduction = transforms.Resize(eval(*ori_size))
                    # output = reduction(output)
                    # 保存图片
                    save_image(output, os.path.join(train_vis_path, '{}.png'.format(*name)))

                pbar.update(1)

    if save_option:
        torch.save({'model': net.state_dict()}, os.path.join(save_path, 'models', 'MyUnet.pth'))

