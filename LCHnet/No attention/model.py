import torch
import torch.nn as nn
class model(nn.Module):
    def __init__(self,config, in_channels, out_channels, num_classes=10):
        super(model, self).__init__()
        # 全连接层，输入维度根据 PatchEmbedding 输出计算
        self.fc1 = nn.Linear((32 // 4) ** 2 * 768, 768)  # 10 是输出类别数
        self.embed = PatchEmbedding(in_channels, patch_size=4, embed_dim=768)
        self.fc2 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()


    def forward(self, x):
        x = self.embed(x)
        # 展平 x，去除除了 batch_size 之外的所有维度
        x = x.flatten(1)  # 仅展平非batch维度
        # 通过第一个全连接层
        x= self.fc1(x)
        # 应用激活函数
        x = self.relu(x)
        # 通过第二个全连接层
        x = self.fc2(x)
        return x
