import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split


# 自定义Dataset类
class MNISTDataset(Dataset):
    def __init__(self, data, labels=None, transform=None):
        """
        data: numpy数组，形状为 (n_samples, 784) 或 (n_samples, 28, 28)
        labels: 如果为None，则测试集没有标签
        transform: torchvision transforms
        """
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 如果是测试集，没有标签
        image = self.data[idx]
        # 确保图像是二维的 (28, 28)
        if image.ndim == 1:
            image = image.reshape(28, 28).astype(np.float32)
        else:
            image = image.astype(np.float32)

        # 添加通道维度 (H, W) -> (1, H, W) 因为灰度图
        image = np.expand_dims(image, axis=0)  # (1, 28, 28)

        # 应用transform
        if self.transform:
            # transforms要求输入是PIL或tensor，这里先转为tensor
            image = torch.from_numpy(image)
            image = self.transform(image)
        else:
            image = torch.from_numpy(image) / 255.0  # 归一化到[0,1]

        if self.labels is not None:
            label = self.labels[idx]
            label = torch.tensor(label, dtype=torch.long)
            return image, label
        else:
            return image


if __name__ == '__main__':
    pass