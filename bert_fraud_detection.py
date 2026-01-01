"""
BERT 欺诈对话检测 + 对抗攻击实验
完整可运行版本
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# PyTorch
import torch
from torch.utils.data import DataLoader, Dataset

# Transformers
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 数据集类
class FraudDataset(Dataset):
    """欺诈对话数据集"""
    
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
        
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
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
            'label': torch.tensor(label, dtype=torch.long)
        }


# 对抗扰动类
class ChineseAdversarialAttack:
    """中文对抗攻击"""
    
    def __init__(self):
        # 同义词替换
        self.synonyms = {
            '转账': ['汇款', '打款', '划款', '转钱', '汇钱'],
            '账户': ['账号', '用户', '户头', '帐户'],
            '密码': ['口令', '暗号', '秘钥', '密钥'],
            '验证码': ['动态码', '校验码', '安全码', '确认码'],
            '银行': ['金融机构', '银号', '金融'],
            '客服': ['服务员', '工作人员', '客户代表'],
            '链接': ['网址', '地址', '网站', '连接'],
            '点击': ['打开', '访问', '进入', '查看'],
            '下载': ['安装', '获取', '保存'],
            '登录': ['进入', '访问', '登入', '签到'],
            '短信': ['信息', '消息', '通知', '简讯'],
            '手机': ['电话', '移动电话', '号码'],
            '贷款': ['借款', '借贷', '信贷', '融资'],
            '利息': ['利率', '回报', '收益'],
            '退款': ['返还', '退回', '返款'],
            '冻结': ['封停', '锁定', '暂停', '封存'],
            '风险': ['隐患', '危险', '问题', '威胁'],
            '安全': ['保障', '防护', '保护'],
            '免费': ['不收费', '零元', '白送', '赠送'],
            '中奖': ['获奖', '得奖', '幸运'],
            '优惠': ['折扣', '减免', '让利', '特惠'],
            '红包': ['奖励', '礼金', '现金券'],
            '支付': ['付款', '缴费', '结算'],
            '订单': ['单子', '订购单', '购买记录'],
            '系统': ['平台', '程序', '后台'],
            '异常': ['问题', '故障', '状况'],
            '立即': ['马上', '赶紧', '尽快', '立刻'],
        }
        
        # 形近字
        self.similar_chars = {
            '转': ['专', '砖', '传'], '账': ['帐', '胀', '张'],
            '密': ['蜜', '秘', '泌'], '银': ['很', '根', '跟'],
            '链': ['连', '联', '莲'], '点': ['店', '典', '电'],
            '冻': ['动', '栋', '洞'], '贷': ['代', '戴', '带'],
            '风': ['丰', '封', '峰'], '险': ['显', '现', '县'],
        }
        
        # 对话标记替换
        self.dialogue_markers = {
            'left:': ['A:', '客:', '甲:', '对方:', ''],
            'right:': ['B:', '用:', '乙:', '我方:', ''],
        }
        
    def synonym_attack(self, text, num_changes=10):
        """同义词替换攻击"""
        perturbed = text
        changes = 0
        
        for word, replacements in sorted(self.synonyms.items(), key=lambda x: len(x[0]), reverse=True):
            if word in perturbed and changes < num_changes:
                perturbed = perturbed.replace(word, random.choice(replacements), 1)
                changes += 1
                
        return perturbed
    
    def char_attack(self, text, ratio=0.1):
        """字符级扰动"""
        chars = list(text)
        num_changes = max(1, int(len(chars) * ratio))
        indices = [i for i, c in enumerate(chars) if c in self.similar_chars]
        random.shuffle(indices)
        
        for idx in indices[:num_changes]:
            chars[idx] = random.choice(self.similar_chars[chars[idx]])
            
        return ''.join(chars)
    
    def structure_attack(self, text):
        """结构扰动 - 改变对话标记"""
        perturbed = text
        for marker, replacements in self.dialogue_markers.items():
            if marker in perturbed:
                perturbed = perturbed.replace(marker, random.choice(replacements))
        return perturbed
    
    def combined_attack(self, text):
        """组合攻击"""
        perturbed = self.synonym_attack(text, num_changes=10)
        perturbed = self.structure_attack(perturbed)
        perturbed = self.char_attack(perturbed, ratio=0.05)
        return perturbed


# 训练函数
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(dataloader, desc='Training')
    for batch in progress_bar:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    """评估模型"""
    model.eval()
    predictions = []
    true_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
            
    return predictions, true_labels


def predict_single(model, tokenizer, text, device, max_length=256):
    """预测单条文本"""
    model.eval()
    
    encoding = tokenizer(
        text,
        max_length=max_length,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        pred = torch.argmax(outputs.logits, dim=1).item()
        
    return pred


# ==================== 主函数 ====================
def main():
    print("=" * 70)
    print("BERT 欺诈对话检测 + 对抗攻击实验")
    print("=" * 70)
    print(f"设备: {device}")
    
    # ========== 1. 数据加载 ==========
    print("\n[1] 加载数据...")
    
    # 尝试不同编码
    for encoding in ['utf-8', 'gbk', 'gb2312']:
        try:
            df = pd.read_csv('测试集结果.csv', encoding=encoding)
            print(f"使用 {encoding} 编码读取成功")
            break
        except:
            continue
    
    # 预处理
    df['text'] = df['specific_dialogue_content'].apply(
        lambda x: str(x).replace('"', '').strip()[:512]  # 限制长度
    )
    df['label'] = df['is_fraud'].apply(
        lambda x: 1 if str(x).upper() == 'TRUE' else 0
    )
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    
    print(f"总样本数: {len(df)}")
    print(f"欺诈样本: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.2, random_state=SEED, stratify=df['label']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1, random_state=SEED, stratify=y_train
    )
    
    print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")
    
    # ========== 2. 加载 BERT ==========
    print("\n[2] 加载 BERT 模型...")
    model_name = './bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    model.to(device)
    
    # ========== 3. 创建数据加载器 ==========
    BATCH_SIZE = 16
    MAX_LENGTH = 256
    
    train_dataset = FraudDataset(X_train, y_train, tokenizer, MAX_LENGTH)
    val_dataset = FraudDataset(X_val, y_val, tokenizer, MAX_LENGTH)
    test_dataset = FraudDataset(X_test, y_test, tokenizer, MAX_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # ========== 4. 训练设置 ==========
    EPOCHS = 3
    LEARNING_RATE = 2e-5
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    # ========== 5. 训练模型 ==========
    print("\n[3] 开始微调 BERT...")
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('='*50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
        print(f"训练损失: {train_loss:.4f}")
        
        # 验证
        val_preds, val_labels = evaluate(model, val_loader, device)
        val_acc = accuracy_score(val_labels, val_preds)
        print(f"验证准确率: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_bert_model.pt')
            print("✓ 保存最佳模型")
    
    # ========== 6. 测试评估 ==========
    print("\n[4] 测试集评估...")
    model.load_state_dict(torch.load('best_bert_model.pt'))
    test_preds, test_labels = evaluate(model, test_loader, device)
    
    original_acc = accuracy_score(test_labels, test_preds)
    original_prec = precision_score(test_labels, test_preds)
    original_rec = recall_score(test_labels, test_preds)
    original_f1 = f1_score(test_labels, test_preds)
    
    print(f"\n{'='*50}")
    print("BERT 微调后原始性能")
    print('='*50)
    print(f"准确率: {original_acc:.4f}")
    print(f"精确率: {original_prec:.4f}")
    print(f"召回率: {original_rec:.4f}")
    print(f"F1分数: {original_f1:.4f}")
    
    print("\n分类报告:")
    print(classification_report(test_labels, test_preds, target_names=['正常', '欺诈']))
    
    # ========== 7. 对抗攻击实验 ==========
    print("\n[5] 对抗攻击实验...")
    attacker = ChineseAdversarialAttack()
    
    # 找出预测正确的样本
    correct_indices = [i for i, (p, t) in enumerate(zip(test_preds, test_labels)) if p == t]
    print(f"原始正确预测: {len(correct_indices)}/{len(test_labels)}")
    
    # 不同攻击方法
    attack_methods = {
        '同义词替换': lambda t: attacker.synonym_attack(t, 10),
        '字符扰动': lambda t: attacker.char_attack(t, 0.1),
        '结构扰动': lambda t: attacker.structure_attack(t),
        '组合攻击': lambda t: attacker.combined_attack(t),
    }
    
    attack_results = {}
    
    for attack_name, attack_func in attack_methods.items():
        print(f"\n测试攻击方法: {attack_name}")
        
        success_count = 0
        test_samples = min(200, len(correct_indices))  # 限制测试数量
        
        for idx in tqdm(correct_indices[:test_samples], desc=f'{attack_name}'):
            text = X_test[idx]
            true_label = y_test[idx]
            
            # 生成对抗样本
            perturbed = attack_func(text)
            
            # 预测
            pert_pred = predict_single(model, tokenizer, perturbed, device)
            
            if pert_pred != true_label:
                success_count += 1
                
        attack_rate = success_count / test_samples
        attack_results[attack_name] = {
            'success_count': success_count,
            'total': test_samples,
            'attack_rate': attack_rate,
        }
        
        print(f"  攻击成功: {success_count}/{test_samples} = {attack_rate:.2%}")
    
    # ========== 8. 结果汇总 ==========
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n原始 BERT 性能:")
    print(f"  准确率: {original_acc:.4f}")
    print(f"  F1分数: {original_f1:.4f}")
    
    print(f"\n对抗攻击结果:")
    print(f"{'攻击方法':<15} {'成功数':<10} {'成功率':<10}")
    print("-" * 40)
    for name, result in attack_results.items():
        print(f"{name:<15} {result['success_count']:<10} {result['attack_rate']:.2%}")
    
    # 保存结果
    final_results = {
        'model': 'bert-base-chinese',
        'original_accuracy': original_acc,
        'original_precision': original_prec,
        'original_recall': original_rec,
        'original_f1': original_f1,
        'attack_results': attack_results,
    }
    
    with open('bert_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(final_results, f, ensure_ascii=False, indent=2)
    
    print("\n✓ 结果已保存到 bert_experiment_results.json")
    print("\n实验完成!")


if __name__ == '__main__':
    main()
