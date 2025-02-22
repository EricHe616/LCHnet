import torchvision
import torch
from torch.utils.data import DataLoader
# 加载训练集
train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]), download=True)

# 加载测试集
test_data_set = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]), download=True)

# 创建训练集数据加载器
train_data_load = DataLoader(train_data_set, batch_size=1000, shuffle=True, drop_last=True)
# 创建测试集数据加载器
test_data_load = DataLoader(test_data_set, batch_size=1000, shuffle=False, drop_last=True)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')