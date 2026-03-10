import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import DataLoader
from tools.dataset import MNISTDataset

# 2. 加载数据（Kaggle环境中的路径）
train_data = pd.read_csv("./data/train.csv")
test_data = pd.read_csv("./data/test.csv")


# 4. 准备数据
# 分离训练集的特征和标签
X_train_ = train_data.drop('label', axis=1).values
y_train_ = train_data['label'].values

X_test = test_data.values

# 划分训练集和验证集（70%训练，30%验证）
X_train, X_val, y_train, y_val = train_test_split(X_train_, y_train_,
                                                  test_size=0.3,
                                                  random_state=42,
                                                  stratify=y_train_)

# 可选的数据增强（如随机旋转）
train_transform = transforms.Compose([
    transforms.RandomRotation(10),  # 随机旋转±10度
    transforms.Normalize((0.5,), (0.5,))  # 标准化到[-1, 1] 因为之前/255是[0,1]，这里用(0.5,0.5)转换到[-1,1]
])

# 验证集和测试集不需要增强，只需要标准化
valid_transform = transforms.Compose([
    transforms.Normalize((0.5,), (0.5,))
])

# 创建Dataset对象
train_dataset = MNISTDataset(X_train, y_train, transform=train_transform)
val_dataset = MNISTDataset(X_val, y_val, transform=valid_transform)
test_dataset = MNISTDataset(X_test, labels=None, transform=valid_transform)

# 创建DataLoader
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

if __name__ == '__main__':
    pass