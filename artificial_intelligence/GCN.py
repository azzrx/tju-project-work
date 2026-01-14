import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv

# =========================
# 1. 设备选择
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# =========================
# 2. 加载 Cora 数据集
# =========================
dataset = Planetoid(root='./data', name='Cora')
data = dataset[0].to(device)

print("Dataset:", dataset)
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.num_edges)
print("Number of features:", dataset.num_node_features)
print("Number of classes:", dataset.num_classes)

# =========================
# 3. 定义 GCN 模型
# =========================
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()

        # 第一层 GCN
        self.conv1 = GCNConv(in_channels, hidden_channels)
        # 第二层 GCN
        self.conv2 = GCNConv(hidden_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一层：邻居聚合 + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        # Dropout（仅训练阶段启用）
        x = F.dropout(x, p=0.5, training=self.training)

        # 第二层：输出 logits
        x = self.conv2(x, edge_index)
        return x

# =========================
# 4. 初始化模型与优化器
# =========================
model = GCN(
    in_channels=dataset.num_node_features,  # 1433
    hidden_channels=16,                     # 原论文设置
    out_channels=dataset.num_classes        # 7
).to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
    weight_decay=5e-4
)

# =========================
# 5. 训练函数
# =========================
def train():
    model.train()
    optimizer.zero_grad()

    # 全图前向传播
    out = model(data.x, data.edge_index)

    # 只在训练节点上计算损失
    loss = F.cross_entropy(
        out[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    return loss.item()

# =========================
# 6. 测试 / 验证函数
# =========================
@torch.no_grad()
def evaluate(mask):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    correct = (pred[mask] == data.y[mask]).sum()
    acc = int(correct) / int(mask.sum())
    return acc

# =========================
# 7. 训练主循环
# =========================
num_epochs = 200

for epoch in range(1, num_epochs + 1):
    loss = train()
    train_acc = evaluate(data.train_mask)
    val_acc = evaluate(data.val_mask)

    if epoch % 10 == 0:
        print(
            f"Epoch {epoch:03d} | "
            f"Loss {loss:.4f} | "
            f"Train Acc {train_acc:.4f} | "
            f"Val Acc {val_acc:.4f}"
        )

# =========================
# 8. 最终测试集评估
# =========================
test_acc = evaluate(data.test_mask)
print("Test Accuracy:", test_acc)
