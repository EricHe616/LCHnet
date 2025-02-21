import math
from dataclasses import dataclass
from typing import Tuple, Optional, Literal
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt, font_manager
import torchvision

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

# 定义 PatchEmbedding 类
class PatchEmbedding(nn.Module):
    def __init__(self, in_channels=3, patch_size=4, embed_dim=768, img_size=32,):
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        # 计算图像中 patch 的数量
        self.num_patches = (img_size // patch_size) ** 2
        # 卷积层来切分图像并将每个 patch 映射为一个嵌入向量
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # 使用卷积操作来提取每个 patch 并映射为嵌入向量
        x = self.conv(x)  # 输出形状为 (batch_size, embed_dim, num_patches, num_patches)
        # 将空间维度展平为序列
        x = x.flatten(2)  # 展平后形状为 (batch_size, embed_dim, num_patches^2)
        # 转置成 (batch_size, num_patches^2, embed_dim)
        x = x.transpose(1, 2)
        return x  # 最终的形状为 (batch_size, num_patches^2, embed_dim)
class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score

class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor
# 定义整体分类模型
class mynet(nn.Module):
    def __init__(self ,in_channels, out_channels, num_classes=10):
        super(mynet, self).__init__()
        self.embed = PatchEmbedding(in_channels, patch_size=4, embed_dim=768)
        self.fc1 = nn.Linear(768, 768)  # 输入为 768 而不是 76800
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()
        self.mla = MultiHeadAttention(d_model=768, n_head=4)

    def forward(self, x):
        x = self.embed(x)
        q = k = v = x  # 这里假设是自注意力（Self-Attention），q, k, v 都是一样的
        x = self.mla(q, k, v)[0]  # 只获取第一个返回值 x


        # 展平 x，去除除了 batch_size 之外的所有维度
        x = x.flatten(1)  # 仅展平非batch维度
        x = self.fc1(x)  # 这里输入的特征维度应为 768
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 初始化模型
model = mynet(in_channels=3, out_channels=10)
# 将模型移动到指定设备
model.to(device)

# 计算数据集大小
train_data_size = len(train_data_set)
test_data_size = len(test_data_set)

# 定义损失函数
criterion = nn.CrossEntropyLoss()
# 定义优化器
optim = torch.optim.Adam(model.parameters(), lr=0.001)

# 用于记录每个 epoch 的训练和验证准确率
train_accuracies = []
test_accuracies = []

# 训练和验证循环
for epoch in range(100):
    model.train()
    correct_train = 0
    total_train = 0
    train_loss = 0
    # 遍历训练集的批次
    for j, (imgs, targets) in enumerate(train_data_load):
        imgs = imgs.to(device)
        targets = targets.to(device)
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        _, predicted = torch.max(outputs.data, 1)
        total_train += targets.size(0)
        correct_train += (predicted == targets).sum().item()
        train_loss += loss.item()
    # 计算当前 epoch 的训练准确率
    train_accuracy = 100 * correct_train / total_train
    train_accuracies.append(train_accuracy)

    model.eval()
    correct_test = 0
    total_test = 0
    test_loss = 0
    with torch.no_grad():
        # 遍历验证集的批次
        for inputs, targets in test_data_load:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            _, predicted = torch.max(outputs.data, 1)
            total_test += targets.size(0)
            correct_test += (predicted == targets).sum().item()
            test_loss += loss.item()
    # 计算当前 epoch 的验证准确率
    valid_accuracy = 100 * correct_test / total_test
    test_accuracies.append(valid_accuracy)

    print(f'Epoch: {epoch + 1}, Train Loss: {train_loss / len(train_data_load):.4f}, '
          f'Train Accuracy: {train_accuracy:.2f}%, '
          f'Valid Loss: {test_loss / len(test_data_load):.4f}, '
          f'Valid Accuracy: {valid_accuracy:.2f}%')

# 绘制准确率曲线
epochs = list(range(1, 101))  # 100个epoch
plt.plot(epochs, train_accuracies, 'b', label='Train Accuracy')
plt.plot(epochs, test_accuracies, 'r', label='Valid Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Valid Accuracy')
plt.legend()
plt.show()

# 计算模型的总参数数量
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")