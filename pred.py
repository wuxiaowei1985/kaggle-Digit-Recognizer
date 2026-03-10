import torch
from model.cnn import CNN
from tools.utils import test_loader
from tqdm import tqdm
import pandas as pd
import numpy as np

# 检查是否有GPU可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main_pred():
    # 加载最佳模型
    model = CNN().to(device)
    model.load_state_dict(torch.load("./run/savemodel/best_model.pth"))
    # 对测试集进行预测
    model.eval()
    predictions = []
    with torch.no_grad():
        for images in tqdm(test_loader, desc='Predicting'):
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    # 生成提交文件
    submission = pd.DataFrame({
        'ImageId': np.arange(1, len(predictions) + 1),
        'Label': predictions
    })
    submission.to_csv("./run/submission/submission.csv", index=False)

if __name__ == '__main__':
    main_pred()