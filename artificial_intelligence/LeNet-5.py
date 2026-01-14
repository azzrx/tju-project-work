import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import time

# =========================
# 1. 定义 LeNet-5 网络
# =========================
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)      # 输入1通道，输出6通道，卷积核5x5
        self.pool = nn.MaxPool2d(2, 2)       # 池化层 2x2
        self.conv2 = nn.Conv2d(6, 16, 5)     # 输入6通道，输出16通道
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))   # C1 -> S2
        x = self.pool(torch.relu(self.conv2(x)))   # C3 -> S4
        x = x.view(-1, 16 * 5 * 5)                 # 展平
        x = torch.relu(self.fc1(x))                # C5
        x = torch.relu(self.fc2(x))                # F6
        x = self.fc3(x)                            # 输出层
        return x

# =========================
# 2. 单张图片预测函数
# =========================
def predict_single_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 打开图片并转灰度
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)  # 增加 batch 维度

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# =========================
# 3. 主函数
# =========================
def main():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 超参数
    batch_size = 64
    learning_rate = 0.001
    num_epochs = 5

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    print("正在加载 MNIST 数据集...")
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

    # 初始化模型、损失函数、优化器
    model = LeNet5().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # =========================
    # 训练循环
    # =========================
    print(f"开始训练，共 {num_epochs} 个轮次...")
    start_time = time.time()

    for epoch in range(num_epochs):
        model.train()
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

            if (i+1) % 100 == 0:
                print(f"轮次 [{epoch+1}/{num_epochs}], 步数 [{i+1}/{len(train_loader)}], 损失: {loss.item():.4f}")

    print(f"训练完成，耗时 {time.time() - start_time:.2f} 秒")

    # =========================
    # 测试集评估
    # =========================
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"网络在 {total} 张测试图片上的准确率: {100 * correct / total:.2f}%")

    # 保存模型
    torch.save(model.state_dict(), 'lenet5_mnist.pth')
    print("模型已保存至 lenet5_mnist.pth")

    # =========================
    # 单张图片预测示例
    # =========================
    image_path = 'test.png'  # 替换为你的手写数字图片路径
    result = predict_single_image(image_path, model, device)
    print(f"单张图片 {image_path} 的预测结果: {result}")

# =========================
# 程序入口
# =========================
if __name__ == '__main__':
    main()
