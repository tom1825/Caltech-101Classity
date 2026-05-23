import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils import get_dataloaders

#残差块
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))

# 自建CNN模型
class mineCNN(nn.Module):
    def __init__(self, num_classes=101):
        super().__init__()

        def stage(in_ch, out_ch, pool=True):
            layers = [
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                ResBlock(out_ch),
            ]
            if pool:
                layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            stage(3,   64),    # 128 → 64
            stage(64,  128),   # 64 → 32
            stage(128, 256),   # 32  → 16
            stage(256, 512),   # 16  → 8
            stage(512, 512, pool=False),  # 8 → 8（保留空间信息）
        )

        # → (B, 512, 1, 1) 
        self.gap = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(            #全连接层
            nn.Flatten(),                           # → (B, 512)
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# 评估
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            _, preds = torch.max(model(images), 1)
            total   += labels.size(0)
            correct += (preds == labels).sum().item()
    return correct / total


# 训练
def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{device}")

    #数据
    train_loader, test_loader, class_names = get_dataloaders(
        data_dir="data/101_ObjectCategories",
        batch_size=64,
    )
    num_classes = len(class_names)
    print(f"分类种类数量: {num_classes}")

    model = mineCNN(num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)                    #损失函数（Label Smoothing=0.1缓解过自信）
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4) #优化器

    epochs       = 40                                                       
    warmup_epochs = 5                                                       #学习率
    warmup_sched  = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.2, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched  = CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-5)

    best_acc = 0.0

    for epoch in range(epochs):
        # 训练
        model.train()
        total_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 梯度裁剪
            optimizer.step()
            total_loss += loss.item()

        #调度器步进 
        if epoch < warmup_epochs:
            warmup_sched.step()
        else:
            cosine_sched.step()

        #评估
        train_acc = evaluate(model, train_loader, device) if (epoch + 1) % 5 == 0 else float("nan")
        test_acc  = evaluate(model, test_loader,  device)
        lr_now    = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1:02d}/{epochs}]  "
              f"Loss: {total_loss:.4f}  "
              f"Train: {train_acc:.4f}  "
              f"Test: {test_acc:.4f}  "
              f"LR: {lr_now:.6f}")

        #保存最优
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_mine_cnn.pth")
            print(f"  ✓ 最优模型已保存(acc={best_acc:.4f})")

    print(f"\n最优准确率: {best_acc:.4f}")


if __name__ == "__main__":
    train()