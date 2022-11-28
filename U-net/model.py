# 模型架构
# 模型结构
# 由backbone特征提取和head特征增强部分组成
# from, module, args[out_channels, kernel_size, stride]

from ruamel import yaml
import os

cfg = {'boneback': [[-1, 'Double_Conv2D', [64, 3, 1]], # 0
     [-1, 'MaxPooling', []],
     [-1, 'Double_Conv2D', [128, 3, 1]], # 2
     [-1, 'MaxPooling', []],
     [-1, 'Dropout', [0.2]],
     [-1, 'Double_Conv2D', [256, 3, 1]], # 5
     [-1, 'MaxPooling', []],
     [-1, 'Double_Conv2D', [512, 3, 1]], # 7
     [-1, 'MaxPooling', []],
     [-1, 'Dropout', [0.2]],
     [-1, 'Double_Conv2D', [1024, 3, 1]]],


    'head': [[[-1, 7], 'Concat', [-1, 1]],
     [-1, 'Double_Conv2D', [256, 3, 1]],
     [[-1, 5], 'Concat', [-1, 1]],
     [-1, 'Double_Conv2D', [128, 3, 1]],
     [[-1, 2], 'Concat', [-1, 1]],
     [-1, 'Double_Conv2D', [64, 3, 1]],
     [[-1, 0], 'Concat', [-1, 1]],
     [-1, 'Double_Conv2D', [32, 3, 1]],
     [-1, 'Conv2D', [1, 1, 1]]]
       }

if __name__ == '__main__':
    root_path = r'C:\Users\13632\Documents\Python Scripts\U-net\MyUnet\data\models'
    name = 'Unet'
    suffix = '.yaml'
    path = os.path.join(root_path, name + suffix)
    f = open(path,'w')
    # 使用yaml.dump生成yaml文件
    yaml.dump(cfg, f, Dumper=yaml.RoundTripDumper)
    f.close()

