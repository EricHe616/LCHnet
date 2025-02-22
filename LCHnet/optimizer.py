from Data_preparation  import train_data_set, test_data_set,train_data_load, test_data_load,device
import torch
import torch.nn as nn
from LCHnet import LCHnet
from LCHnet import config
from matplotlib import pyplot as plt, font_manager
model = LCHnet(config,in_channels=3, out_channels=10)
# 将模型移动到指定设备
model.to(device)
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