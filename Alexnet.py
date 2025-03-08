import torchvision
from torch import nn
import torch
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_data_set = torchvision.datasets.CIFAR10(root='dataset', train=True, transform=torchvision.transforms.Compose([
    torchvision.transforms.RandomCrop(size=(32, 32), padding=4),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]), download=True)

test_data_set = torchvision.datasets.CIFAR10(root='dataset', train=False, transform=torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
]), download=True)

train_data_load = DataLoader(train_data_set, batch_size=64, shuffle=True, drop_last=True)
test_data_load = DataLoader(test_data_set, batch_size=64, shuffle=False, drop_last=True)

class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 4 * 4, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

alexnet = AlexNet().to(device)
print(f'模型参数总数: {sum(p.numel() for p in alexnet.parameters())}')

loss_fn = nn.CrossEntropyLoss().to(device)
optim = torch.optim.Adam(alexnet.parameters(), lr=0.001)

epoch = 100

if __name__ == '__main__':
    for i in range(epoch):
        alexnet.train()
        for imgs, targets in train_data_load:
            imgs, targets = imgs.to(device), targets.to(device)

            outputs = alexnet(imgs)
            loss = loss_fn(outputs, targets)
            optim.zero_grad()
            loss.backward()
            optim.step()

        alexnet.eval()
        accuracy_total = 0
        with torch.no_grad():
            for imgs, targets in test_data_load:
                imgs, targets = imgs.to(device), targets.to(device)
                outputs = alexnet(imgs)
                accuracy_total += (outputs.argmax(dim=1) == targets).sum().item()

        accuracy = accuracy_total / len(test_data_set)
        print(f'第 {i+1} 轮, 测试准确率: {accuracy:.4f}')
