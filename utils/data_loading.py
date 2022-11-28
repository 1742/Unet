'''

读取数据，重点
在图像分割中，以原图为x，以mask为标签

'''
import torch
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
import numpy as np
import pandas as pd


# 数据增强器
transformer = transforms.Compose([
        # 缩放
        # transforms.Resize((224, 224)),
        # 随机翻转
        # transforms.RandomHorizontalFlip(),
        # 随机裁剪
        # transforms.RandomCrop(96),
        # 亮度、对比度、色彩变换
        # transforms.ColorJitter(brightness=0.5, contrast=0.5, hue=0.5),
        # 归一化
        transforms.ToTensor(),
        # 进行均值与方差的变化
        # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])


# 构建自己的数据集
class MyDatasets(Dataset):
    def __init__(self, data_root='', data_info='', train=bool, save_name=''):
        # data_root - 数据集路径
        # data_info - 数据集信息表格路径
        super(MyDatasets, self).__init__()
        # 程序data目录
        self.root = os.path.join(os.getcwd()[:-5], 'data')

        # 数据集目录
        if data_root:
            self.data_root = data_root
        else:
             raise FileNotFoundError

        if data_info != '' and os.path.exists(data_info):
            assert 'Can not find {}!'.format(data_info)

        # 生成的信息文件名
        self.save_name = save_name

        # 是否为训练集数据
        if train:
            self.train = 1
            self.train_path = os.path.join(self.data_root, 'training')
            # 数据类别
            self.labels = os.listdir(self.train_path)
        else:
            # 若非已生成的.xlsx文件，则再打开文件test
            self.train = 0
            self.test_path = os.path.join(self.data_root, 'test')
            # 数据类别
            self.labels = os.listdir(self.test_path)

        # 获取数据数量
        self.length = 0

        # 若传入的是数据路径，则生成对应信息文件
        if os.path.isdir(data_root) and data_info == '':
            # 获取路径下的所有图片
            # imgs的路径
            if self.train:
                imgs_path = os.path.join(self.train_path, 'images')
            else:
                imgs_path = os.path.join(self.test_path, 'images')

            # 获取各文件名
            imgs_names = []
            # 后缀
            imgs_format = []
            # 获取所有图片大小
            imgs_size = []
            # 获取相应数据类别下的所有图片
            for i, p in enumerate(os.listdir(imgs_path)):
                info = os.path.splitext(p)
                imgs_names.append(info[0].split('_')[0])
                imgs_format.append(info[1])
                img = Image.open(os.path.join(imgs_path, p))
                imgs_size.append(img.size)

            # 记录该类文件的索引范围
            self.length = len(imgs_names)

            # 创建该标签下的文件信息字典
            metadata = dict()
            metadata['Filename'] = imgs_names
            metadata['Format'] = imgs_format
            metadata['Size'] = imgs_size

            # 创建文件信息xlsx文件，方便读取，包含图片名、图片格式、图片大小
            dataframe = pd.DataFrame(metadata)
            dataframe.to_excel(os.path.join(self.root, self.save_name+'.xlsx'), sheet_name='Sheet1', index=False)

            print('Successfully generated {}.xlsx file from {}'.format(self.save_name, self.train_path))

            self.data = dict(pd.DataFrame(pd.read_excel(os.path.join(self.root, self.save_name+'.xlsx'), sheet_name='Sheet1')))

        # 若传入的是信息文件，则直接读取
        else:
            self.data = pd.DataFrame(pd.read_excel(data_info, sheet_name='Sheet1'))
            self.length = self.data.shape[0]
            self.data = dict(self.data)
            print('Successfully read data from {}'.format(data_info))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # 从信息文件中读取图片路径
        # 信息文件路径
        img_name = self.data['Filename'][index]
        img_format = self.data['Format'][index]

        # 获得单张图片的绝对路径
        if self.train:
            image_path = os.path.join(self.train_path, self.labels[1], str(img_name)+'_training'+img_format)
            label_path = os.path.join(self.train_path, self.labels[0], str(img_name)+'_manual1.gif')
            mask_path = os.path.join(self.train_path, self.labels[2], str(img_name)+'_training_mask.gif')
        else:
            image_path = os.path.join(self.test_path, self.labels[1], str(img_name) + '_test' + img_format)
            label_path = os.path.join(self.test_path, self.labels[0], str(img_name) + '_manual1.gif')
            mask_path = os.path.join(self.test_path, self.labels[2], str(img_name) + '_test_mask.gif')

        # 原图
        image = Image.open(image_path).convert('RGB')
        # 将蒙版的黑白部分颠倒以覆盖在标签图上
        mask = 255 - np.array(Image.open(mask_path).convert('L'))
        label = np.array(Image.open(label_path).convert('L'))

        # 制作蒙版
        mask = Image.fromarray(np.clip(mask+label, a_min=0, a_max=255))

        # 返回的是tensor类
        return transformer(image), transformer(mask), img_name


if __name__ == '__main__':
    data_path = r'C:\Users\13632\Desktop\DRIVE'
    info_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data\retinal.xlsx'
    save_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data'
    MyDataset = MyDatasets(data_path, info_path, train=1)
    image, mask, name = MyDataset.__getitem__(1)
    print(name)
    print(image.size(), mask.size())

