import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models

from utils import get_dataloaders


# ResNet18 迁移
def build_resnet18(num_classes):
    # 加载 ImageNet 预训练权重
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

    # 只训练最后的全连接层
    for param in model.parameters():
        param.requires_grad = False
    # 替换最后的全连接层（1000类改成101类）
    in_features = model.fc.in_features   # 512
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )

    return model


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

    train_loader, test_loader, class_names = get_dataloaders(
        data_dir="data/101_ObjectCategories",
        batch_size=64,
    )
    num_classes = len(class_names)
    print(f"分类种类数量: {num_classes}")

    model = build_resnet18(num_classes).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)                   #损失函数（Label Smoothing=0.1缓解过自信）

    #stage1：只训练全连接层，前10 epoch 快速收敛
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr=1e-3, weight_decay=1e-4)   #优化器
    stage1_epochs = 10

    print("\nStage 1: 只训练全连接层")
    for epoch in range(stage1_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        test_acc = evaluate(model, test_loader, device)
        print(f"Epoch [{epoch+1:02d}/{stage1_epochs}]  "
              f"Loss: {total_loss:.4f}  Test: {test_acc:.4f}")

    #Stage 2全部层小学习率微调
    for param in model.parameters():
        param.requires_grad = True
    print("\n===== Stage 2: 全网络微调 =====")

    # 主干用更小的 lr，全连接层用大一点的 lr
    optimizer = optim.AdamW([
        {"params": list(model.parameters())[:-4], "lr":1e-4},   # 主干
        {"params": list(model.parameters())[-4:], "lr": 5e-4},   # 全连接层
    ], weight_decay=1e-4)

    stage2_epochs = 20
    scheduler = CosineAnnealingLR(optimizer, T_max=stage2_epochs, eta_min=1e-6)

    best_acc = 0.0

    for epoch in range(stage2_epochs):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item()

        scheduler.step()

        train_acc = evaluate(model, train_loader, device) if (epoch + 1) % 5 == 0 else float("nan")
        test_acc  = evaluate(model, test_loader,  device)
        lr_now    = optimizer.param_groups[0]["lr"]

        print(f"Epoch [{epoch+1:02d}/{stage2_epochs}]  "
              f"Loss: {total_loss:.4f}  "
              f"Train: {train_acc:.4f}  "
              f"Test: {test_acc:.4f}  "
              f"LR: {lr_now:.2e}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_resnet18.pth")
            print(f"  ✓ 最优模型已保存(acc={best_acc:.4f})")

    print(f"\n最优准确率: {best_acc:.4f}")


if __name__ == "__main__":
    train()