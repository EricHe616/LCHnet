import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from matplotlib import pyplot as plt


# Vision Transformer (ViT) Model
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
train_data_load = DataLoader(train_data_set, batch_size=100, shuffle=True, drop_last=True)
# 创建测试集数据加载器
test_data_load = DataLoader(test_data_set, batch_size=100, shuffle=False, drop_last=True)

# 选择设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class ViT(nn.Module):
    def __init__(self, image_size=32, patch_size=8, num_classes=10, embed_dim=768, num_heads=1, num_layers=1):
        super(ViT, self).__init__()

        # Patch embedding layer
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.patch_dim = 3 * patch_size * patch_size  # RGB channel

        self.linear_embedding = nn.Linear(self.patch_dim, embed_dim)

        # Positional encoding
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer encoder
        self.encoder_layers = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layers, num_layers=num_layers
        )

        # Classifier head
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # Divide image into patches
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, self.num_patches, self.patch_dim)

        # Embedding
        x = self.linear_embedding(patches)

        # Adding positional encoding
        x = x + self.position_embedding

        # Transformer encoder
        x = x.permute(1, 0, 2)  # (seq_len, batch_size, embed_dim)
        x = self.transformer_encoder(x)

        # Global average pooling
        x = x.mean(dim=0)

        # Classifier
        x = self.fc(x)

        return x


# Initialize the model
model = ViT(image_size=32, patch_size=8, num_classes=10, embed_dim=256, num_heads=1, num_layers=1)
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