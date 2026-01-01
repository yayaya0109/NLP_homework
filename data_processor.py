"""
欺诈对话数据处理模块
用于加载、预处理和处理CSV格式的中文欺诈对话数据集
"""

import pandas as pd
import re
import os
from typing import Tuple, List, Optional
from sklearn.model_selection import train_test_split


def clean_dialogue(text: str) -> str:
    """
    清洗对话文本
    """
    if pd.isna(text):
        return ""
        
    text = str(text)
    
    # 去除引号
    text = text.replace('"', '').replace("'", '')
    
    # 去除首尾空格
    text = text.strip()
    
    # 统一空白字符
    text = re.sub(r'\s+', ' ', text)
    
    # 去除 "left:" 和 "right:" 标签 (可选，保留可增加上下文信息)
    # text = re.sub(r'(left|right):\s*', '', text)
    
    return text


def parse_dialogue_content(text: str) -> List[Tuple[str, str]]:
    """
    解析对话内容，分离不同说话者
    """
    if not text:
        return []
        
    # 匹配 "left:" 或 "right:" 开头的内容
    pattern = r'(left|right):\s*([^left|right]+)'
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    dialogues = []
    for speaker, content in matches:
        content = content.strip()
        if content:
            dialogues.append((speaker.lower(), content))
            
    return dialogues


def load_fraud_dataset(csv_path: str, 
                       text_column: str = 'specific_dialogue_content',
                       label_column: str = 'is_fraud') -> pd.DataFrame:
    """
    加载欺诈对话数据集
    """
    # 尝试不同编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16']
    
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=encoding)
            print(f"成功使用 {encoding} 编码读取文件")
            break
        except Exception as e:
            continue
            
    if df is None:
        raise ValueError(f"无法读取文件: {csv_path}")
        
    print(f"数据集大小: {len(df)} 条")
    print(f"列名: {list(df.columns)}")
    
    # 清洗文本
    if text_column in df.columns:
        df['cleaned_text'] = df[text_column].apply(clean_dialogue)
    else:
        raise ValueError(f"找不到文本列: {text_column}")
        
    # 处理标签
    if label_column in df.columns:
        # 转换标签为二进制
        df['label'] = df[label_column].apply(
            lambda x: 1 if str(x).lower() in ['true', '1', 'yes'] else 0
        )
    else:
        raise ValueError(f"找不到标签列: {label_column}")
        
    # 统计信息
    fraud_count = df['label'].sum()
    normal_count = len(df) - fraud_count
    print(f"欺诈样本: {fraud_count} ({fraud_count/len(df)*100:.2f}%)")
    print(f"正常样本: {normal_count} ({normal_count/len(df)*100:.2f}%)")
    
    return df


def prepare_data_for_training(df: pd.DataFrame,
                              test_size: float = 0.2,
                              random_state: int = 42) -> Tuple:
    """
    准备训练数据
    """
    X = df['cleaned_text'].values
    y = df['label'].values
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"训练集: {len(X_train)} 条")
    print(f"测试集: {len(X_test)} 条")
    
    return X_train, X_test, y_train, y_test


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """保存处理后的数据"""
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"已保存到: {output_path}")


class FraudDataset:
    """
    PyTorch Dataset 包装类
    用于 BERT 等深度学习模型的训练
    """
    def __init__(self, texts: List[str], labels: List[int], 
                 tokenizer=None, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': label
            }
        else:
            return {'text': text, 'label': label}


if __name__ == '__main__':
    # 测试数据加载
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        # 使用示例路径
        csv_path = 'fraud_dialogue_dataset.csv'
        
    if os.path.exists(csv_path):
        df = load_fraud_dataset(csv_path)
        X_train, X_test, y_train, y_test = prepare_data_for_training(df)
        
        print("\n示例数据:")
        for i in range(min(3, len(X_train))):
            print(f"文本: {X_train[i][:100]}...")
            print(f"标签: {'欺诈' if y_train[i] == 1 else '正常'}")
            print("-" * 40)
    else:
        print(f"文件不存在: {csv_path}")
        print("请提供正确的CSV文件路径")
