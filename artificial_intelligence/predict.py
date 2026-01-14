import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms

# -------------------------
# 定义 LeNet-5 网络结构
# -------------------------
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# -------------------------
# 单张图片预测函数
# -------------------------
def predict_single_image(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    image = Image.open(image_path).convert('L')
    image = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
    return predicted.item()

# -------------------------
# 主函数
# -------------------------
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    model = LeNet5().to(device)

    # 直接加载已训练好的权重
    model.load_state_dict(torch.load('lenet5_mnist.pth', map_location=device))
    print("模型权重已加载")

    # 输入要预测的图片路径
    image_path = 'test.png'  # 替换为你的图片
    result = predict_single_image(image_path, model, device)
    print(f"单张图片 {image_path} 的预测结果: {result}")

if __name__ == '__main__':
    main()
