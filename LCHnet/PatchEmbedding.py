from torch import nn
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