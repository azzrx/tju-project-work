import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import time

# 定义 LeNet-5 网络结构
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        # 卷积层 1: 输入 1 通道 (灰度图), 输出 6 通道, 卷积核 5x5
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 池化层: 最大池化, 窗口大小 (2, 2)
        self.pool = nn.MaxPool2d(2, 2)
        # 卷积层 2: 输入 6 通道, 输出 16 通道, 卷积核 5x5
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层 1: 输入 16*5*5, 输出 120
        # 注意: MNIST 原始尺寸为 28x28。经过 transform 填充后变为 32x32。
        # 32x32 -> conv1(5x5) -> 28x28 -> pool(2x2) -> 14x14
        # 14x14 -> conv2(5x5) -> 10x10 -> pool(2x2) -> 5x5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # 全连接层 2: 输入 120, 输出 84
        self.fc2 = nn.Linear(120, 84)
        # 全连接层 3: 输入 84, 输出 10 (类别 0-9)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # C1 -> S2
        x = self.pool(torch.relu(self.conv1(x)))
        # C3 -> S4
        x = self.pool(torch.relu(self.conv2(x)))
        # 展平
        x = x.view(-1, 16 * 5 * 5)
        # C5
        x = torch.relu(self.fc1(x))
        # F6
        x = torch.relu(self.fc2(x))
        # 输出层
        x = self.fc3(x)
        return x

def main():
    # 1. 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # 3. 数据加载与预处理
    # 转换: 转为 Tensor 并归一化。
    # 调整大小/填充至 32x32 以匹配原始 LeNet-5 架构
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)) # MNIST 的均值和标准差
    ])

    print("正在加载 MNIST 数据集...")
    try:
        train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                                download=True, transform=transform)
        test_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                               download=True, transform=transform)

        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True)
        test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)
    except Exception as e:
        print(f"加载数据集出错: {e}")
        return

    # 4. 初始化模型、损失函数和优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 5. 训练循环
    print(f"开始训练，共 {num_epochs} 个轮次 (epochs)...")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)

            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, labels)

            # 反向传播与优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if (i+1) % 100 == 0:
                print(f'轮次 [{epoch+1}/{num_epochs}], 步数 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}')

    print(f"训练完成，耗时 {time.time() - start_time:.2f}秒")

    # 6. 评估
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in enumerate(test_loader):
            images, labels = labels[0].to(device), labels[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'网络在 10000 张测试图片上的准确率: {100 * correct / total:.2f}%')

    # 保存模型检查点
    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print("模型已保存至 lenet5_mnist.pth")

if __name__ == '__main__':
    main()
