import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

def sample_stratified_data(csv_path, sample_ratio=0.05):
    """
    从CSV文件中按照保持类别比例的方式采样数据
    
    参数:
    csv_path: CSV文件路径
    sample_ratio: 采样比例，默认为5%
    
    返回:
    采样后的数据框
    """
    # 读取CSV文件
    df = pd.read_csv(csv_path)
    
    # 将one-hot编码转换为类别标签
    y = df.iloc[:, -7:].values.argmax(axis=1)  # 最后7列是one-hot编码的标签
    X = df.iloc[:, :-7]  # 特征列
    
    # 使用StratifiedShuffleSplit进行分层采样
    sss = StratifiedShuffleSplit(n_splits=1, test_size=sample_ratio, random_state=42)
    
    # 获取采样索引和剩余索引
    for train_idx, sample_idx in sss.split(X, y):
        sampled_df = df.iloc[sample_idx]
        rest_df = df.iloc[train_idx]  # 保存剩余的数据
    
    # 保存采样后的数据
    output_path = csv_path.replace('.csv', '_sampled.csv')
    rest_path = csv_path.replace('.csv', '_rest.csv')
    sampled_df.to_csv(output_path, index=False)
    rest_df.to_csv(rest_path, index=False)  # 保存剩余的数据
    
    print(f"原始数据大小: {len(df)}")
    print(f"采样后数据大小: {len(sampled_df)}")
    print(f"剩余数据大小: {len(rest_df)}")
    print(f"原始类别分布:\n{pd.Series(y).value_counts(normalize=True)}")
    print(f"采样后类别分布:\n{pd.Series(sampled_df.iloc[:, -7:].values.argmax(axis=1)).value_counts(normalize=True)}")
    
    return sampled_df

if __name__ == "__main__":
    csv_path = "/data_hdd/cyang/Code/MedSD/Classification/RAC-MT/data/skin/training.csv"
    sampled_data = sample_stratified_data(csv_path)
