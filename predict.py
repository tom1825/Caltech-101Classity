import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from utils import get_dataloaders

# 复制模型定义
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
            stage(3,   64),
            stage(64,  128),
            stage(128, 256),
            stage(256, 512),
            stage(512, 512, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


def build_resnet18(num_classes):
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.4),
        nn.Linear(256, num_classes),
    )
    return model


# 推理
def predict(image_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #获取类别名称
    _, _, class_names = get_dataloaders(
        data_dir="data/101_ObjectCategories",
        batch_size=1,
    )
    num_classes = len(class_names)

    #预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    #读取图片
    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)  # (1, 3, 128, 128)

    #加载mineCNN/ResNet18
    cnn_model = mineCNN(num_classes=num_classes).to(device)
    cnn_model.load_state_dict(torch.load("saved_models/best_mine_cnn.pth", map_location=device,weights_only=True))
    cnn_model.eval()
    resnet_model = build_resnet18(num_classes=num_classes).to(device)
    resnet_model.load_state_dict(torch.load("saved_models/best_resnet18.pth", map_location=device,weights_only=True))
    resnet_model.eval()

    #推理
    with torch.no_grad():
        # mineCNN
        cnn_out = cnn_model(x)                          # (1, 101)
        cnn_prob = torch.softmax(cnn_out, dim=1)        # 转成概率
        cnn_conf, cnn_idx = torch.max(cnn_prob, dim=1)  # 最大概率和对应类别
        cnn_label = class_names[cnn_idx.item()]

        # ResNet18
        res_out = resnet_model(x)
        res_prob = torch.softmax(res_out, dim=1)
        res_conf, res_idx = torch.max(res_prob, dim=1)
        res_label = class_names[res_idx.item()]

    #打印结果
    print(f"图片：{image_path}")
    print(f"mineCNN  → {cnn_label:<20} 置信度: {cnn_conf.item():.4f}")
    print(f"ResNet18     → {res_label:<20} 置信度: {res_conf.item():.4f}")


if __name__ == "__main__":
    predict("umbrella.jpg")
