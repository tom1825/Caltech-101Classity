import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


def get_dataloaders(
        data_dir,
        batch_size=64,
        train_ratio=0.8
):

    trainAndtest_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(           #标准化
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    #读取
    full_dataset = datasets.ImageFolder(
        root=data_dir,
        transform=trainAndtest_transform
    )

    # 获取类别名称
    class_names = full_dataset.classes

    # 数据集划分
    train_size = int(train_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size

    train_dataset, test_dataset = random_split(
        full_dataset,
        [train_size, test_size]
    )

    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
    )

    return train_loader, test_loader, class_names


