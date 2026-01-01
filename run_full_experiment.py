"""
中文欺诈对话对抗攻击实验
"""

import os
import re
import json
import random
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from collections import defaultdict
from typing import Dict, List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
TORCH_AVAILABLE = bool(torch)
# 对抗扰动类
class ChineseAdversarialAttack:
    """中文对抗攻击"""
    
    def __init__(self):
        # 同义词替换
        self.synonyms = {
            '转账': ['汇款', '打款', '划款', '转钱', '汇钱', '过账'],
            '账户': ['账号', '用户', '户头', '帐户', '帐号'],
            '密码': ['口令', '暗号', '秘钥', '密钥', '通行码'],
            '验证码': ['动态码', '校验码', '安全码', '确认码', '短信码'],
            '银行': ['金融机构', '银号', '金融', '银杭', '银航'],
            '客服': ['服务员', '工作人员', '客户代表', '服务人员'],
            '链接': ['网址', '地址', '网站', '连接', '入口'],
            '点击': ['打开', '访问', '进入', '查看', '浏览'],
            '下载': ['安装', '获取', '保存', '装载'],
            '登录': ['进入', '访问', '登入', '签到', '上线'],
            '短信': ['信息', '消息', '通知', '简讯', '讯息'],
            '手机': ['电话', '移动电话', '号码', '手提'],
            '贷款': ['借款', '借贷', '信贷', '融资', '放款'],
            '利息': ['利率', '回报', '收益', '利钱'],
            '退款': ['返还', '退回', '返款', '退钱'],
            '冻结': ['封停', '锁定', '暂停', '封存', '停用'],
            '风险': ['隐患', '危险', '问题', '威胁', '危机'],
            '安全': ['保障', '防护', '保护', '稳妥'],
            '免费': ['不收费', '零元', '白送', '赠送', '无偿'],
            '中奖': ['获奖', '得奖', '幸运', '抽中'],
            '优惠': ['折扣', '减免', '让利', '特惠', '促销'],
            '红包': ['奖励', '礼金', '现金券', '奖金'],
            '支付': ['付款', '缴费', '结算', '交费'],
            '订单': ['单子', '订购单', '购买记录', '交易单'],
            '系统': ['平台', '程序', '后台', '软件'],
            '异常': ['问题', '故障', '状况', '情况'],
            '立即': ['马上', '赶紧', '尽快', '立刻', '火速'],
            '紧急': ['急迫', '危急', '火急', '迫切'],
            '资金': ['钱款', '款项', '金额', '钱财'],
        }
        
        # 形近字
        self.similar_chars = {
            '转': ['专', '砖', '传', '赚'], '账': ['帐', '胀', '张', '障'],
            '密': ['蜜', '秘', '泌', '宓'], '银': ['很', '根', '跟', '垠'],
            '链': ['连', '联', '莲', '帘'], '点': ['店', '典', '电', '垫'],
            '冻': ['动', '栋', '洞', '侗'], '贷': ['代', '戴', '带', '待'],
            '风': ['丰', '封', '峰', '锋'], '险': ['显', '现', '县', '献'],
            '客': ['咳', '克', '刻', '课'], '服': ['福', '伏', '扶', '幅'],
            '验': ['检', '俭', '剑', '敛'], '码': ['玛', '妈', '吗', '骂'],
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


# BERT 数据集类
if TORCH_AVAILABLE:
    class FraudDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length=256):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
            
        def __len__(self):
            return len(self.texts)
            
        def __getitem__(self, idx):
            text = str(self.texts[idx])
            encoding = self.tokenizer(
                text, max_length=self.max_length, padding='max_length',
                truncation=True, return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'label': torch.tensor(self.labels[idx], dtype=torch.long)
            }


# SVM 分类器
class SVMClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
        self.model = SVC(kernel='rbf', C=1.0, probability=True)
        
    def train(self, texts, labels):
        print("训练 SVM 模型...")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        print("SVM 训练完成!")
        
    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()
        
    def predict_single(self, text):
        return self.predict([text])[0]


# BERT 分类器
class BERTClassifier:
    def __init__(self):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        self.model_name = './bert-base-chinese'
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.model = None
        self.max_length = 256
        
    def train(self, texts, labels, epochs=3, batch_size=16, lr=2e-5):
        """微调 BERT 模型"""
        print(f"微调 BERT 模型 ({epochs} epochs)...")
        
        # 初始化模型
        self.model = BertForSequenceClassification.from_pretrained(
            self.model_name, num_labels=2
        )
        self.model.to(self.device)
        
        # 划分训练/验证集
        X_train, X_val, y_train, y_val = train_test_split(
            texts, labels, test_size=0.1, random_state=SEED, stratify=labels
        )
        
        # 创建数据加载器
        train_dataset = FraudDataset(X_train, y_train, self.tokenizer, self.max_length)
        val_dataset = FraudDataset(X_val, y_val, self.tokenizer, self.max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        # 优化器
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=0, num_training_steps=total_steps
        )
        
        best_val_acc = 0
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            total_loss = 0
            
            for batch in train_loader:
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels_batch = batch['label'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            
            # 验证
            val_preds, val_labels = self._evaluate(val_loader)
            val_acc = accuracy_score(val_labels, val_preds)
            
            print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_bert.pt')
        
        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_bert.pt'))
        print(f"BERT 训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        
    def _evaluate(self, dataloader):
        self.model.eval()
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
                
        return predictions, true_labels
    
    def predict(self, texts):
        """批量预测"""
        self.model.eval()
        predictions = []
        
        for text in texts:
            pred = self.predict_single(text)
            predictions.append(pred)
            
        return predictions
    
    def predict_single(self, text):
        """单条预测"""
        self.model.eval()
        
        encoding = self.tokenizer(
            text, max_length=self.max_length, padding='max_length',
            truncation=True, return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            pred = torch.argmax(outputs.logits, dim=1).item()
            
        return pred


# 主函数
def main():
    parser = argparse.ArgumentParser(description='中文欺诈对话对抗攻击实验')
    parser.add_argument('--data', type=str, default='测试集结果.csv', help='数据集路径')
    parser.add_argument('--model', type=str, default='svm', choices=['svm', 'bert'], help='模型类型')
    parser.add_argument('--max_test', type=int, default=300, help='最大测试样本数')
    parser.add_argument('--epochs', type=int, default=3, help='BERT训练轮数')
    args = parser.parse_args()
    
    print("=" * 70)
    print(f"中文欺诈对话对抗攻击实验 - {args.model.upper()} 模型")
    print("=" * 70)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    for encoding in ['utf-8', 'gbk', 'gb2312', 'gb18030']:
        try:
            df = pd.read_csv(args.data, encoding=encoding)
            print(f"使用 {encoding} 编码读取成功")
            break
        except:
            continue
    
    df['text'] = df['specific_dialogue_content'].apply(lambda x: str(x).replace('"', '').strip()[:512])
    df['label'] = df['is_fraud'].apply(lambda x: 1 if str(x).upper() == 'TRUE' else 0)
    df = df[df['text'].str.len() > 10].reset_index(drop=True)
    
    print(f"总样本数: {len(df)}")
    print(f"欺诈样本: {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    
    # 2. 划分数据
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'].tolist(), df['label'].tolist(),
        test_size=0.2, random_state=SEED, stratify=df['label']
    )
    
    if args.max_test and len(X_test) > args.max_test:
        X_test = X_test[:args.max_test]
        y_test = y_test[:args.max_test]
    
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")
    
    # 3. 训练模型
    print(f"\n[2] 训练 {args.model.upper()} 模型...")
    
    if args.model == 'svm':
        classifier = SVMClassifier()
        classifier.train(X_train, y_train)
    else:
        classifier = BERTClassifier()
        classifier.train(X_train, y_train, epochs=args.epochs)
    
    # 4. 评估原始性能
    print("\n[3] 评估原始模型性能...")
    y_pred = classifier.predict(X_test)
    
    original_acc = accuracy_score(y_test, y_pred)
    original_prec = precision_score(y_test, y_pred)
    original_rec = recall_score(y_test, y_pred)
    original_f1 = f1_score(y_test, y_pred)
    
    print(f"\n{'='*50}")
    print(f"{args.model.upper()} 原始性能")
    print('='*50)
    print(f"准确率: {original_acc:.4f}")
    print(f"精确率: {original_prec:.4f}")
    print(f"召回率: {original_rec:.4f}")
    print(f"F1分数: {original_f1:.4f}")
    
    # 5. 对抗攻击实验
    print("\n[4] 对抗攻击实验...")
    attacker = ChineseAdversarialAttack()
    
    correct_indices = [i for i, (p, t) in enumerate(zip(y_pred, y_test)) if p == t]
    print(f"原始正确预测: {len(correct_indices)}/{len(y_test)}")
    
    attack_methods = {
        '同义词替换': lambda t: attacker.synonym_attack(t, 10),
        '字符扰动': lambda t: attacker.char_attack(t, 0.1),
        '结构扰动': lambda t: attacker.structure_attack(t),
        '组合攻击': lambda t: attacker.combined_attack(t),
    }
    
    attack_results = {}
    success_examples = []
    
    for attack_name, attack_func in attack_methods.items():
        print(f"\n测试: {attack_name}")
        success_count = 0
        
        for idx in correct_indices:
            text = X_test[idx]
            true_label = y_test[idx]
            
            perturbed = attack_func(text)
            pert_pred = classifier.predict_single(perturbed)
            
            if pert_pred != true_label:
                success_count += 1
                if len(success_examples) < 10 and attack_name == '组合攻击':
                    success_examples.append({
                        'original': text[:150],
                        'perturbed': perturbed[:150],
                        'true_label': '欺诈' if true_label == 1 else '正常',
                        'pred_change': f"{'欺诈' if true_label == 1 else '正常'} → {'正常' if true_label == 1 else '欺诈'}"
                    })
        
        attack_rate = success_count / len(correct_indices) if correct_indices else 0
        attack_results[attack_name] = {
            'success': success_count,
            'total': len(correct_indices),
            'rate': attack_rate
        }
        print(f"  成功: {success_count}/{len(correct_indices)} = {attack_rate:.2%}")
    
    # 6. 消融实验
    print("\n[5] 消融实验 - 不同扰动组件贡献分析...")
    
    ablation_components = {
        '仅同义词': lambda t: attacker.synonym_attack(t, 10),
        '仅形近字': lambda t: attacker.char_attack(t, 0.1),
        '仅结构扰动': lambda t: attacker.structure_attack(t),
        '同义词+形近字': lambda t: attacker.char_attack(attacker.synonym_attack(t, 10), 0.1),
        '同义词+结构': lambda t: attacker.structure_attack(attacker.synonym_attack(t, 10)),
        '形近字+结构': lambda t: attacker.structure_attack(attacker.char_attack(t, 0.1)),
        '全部组合': lambda t: attacker.combined_attack(t),
    }
    
    ablation_results = {}
    ablation_samples = min(200, len(correct_indices))  # 消融实验用200个样本
    
    print(f"\n使用 {ablation_samples} 个样本进行消融实验")
    print(f"{'扰动组件':<20} {'攻击成功':<10} {'成功率':<10}")
    print("-" * 45)
    
    for component_name, component_func in ablation_components.items():
        success = 0
        for idx in correct_indices[:ablation_samples]:
            text = X_test[idx]
            true_label = y_test[idx]
            perturbed = component_func(text)
            pert_pred = classifier.predict_single(perturbed)
            if pert_pred != true_label:
                success += 1
        
        rate = success / ablation_samples
        ablation_results[component_name] = {'success': success, 'rate': rate}
        print(f"{component_name:<20} {success:<10} {rate:.2%}")
    
    # 分析各组件贡献
    print("\n" + "-" * 45)
    print("组件贡献分析:")
    
    base_synonym = ablation_results['仅同义词']['rate']
    base_char = ablation_results['仅形近字']['rate']
    base_struct = ablation_results['仅结构扰动']['rate']
    full_rate = ablation_results['全部组合']['rate']
    
    print(f"  同义词替换贡献: {base_synonym:.2%}")
    print(f"  形近字替换贡献: {base_char:.2%}")
    print(f"  结构扰动贡献: {base_struct:.2%}")
    print(f"  组合后总效果: {full_rate:.2%}")
    
    if base_struct > base_synonym and base_struct > base_char:
        print(f"\n  结论: 结构扰动是最有效的攻击组件!")
    
    # 7. 结果汇总
    print("\n" + "=" * 70)
    print("实验结果汇总")
    print("=" * 70)
    
    print(f"\n【模型性能】")
    print(f"  模型: {args.model.upper()}")
    print(f"  准确率: {original_acc:.4f}")
    print(f"  精确率: {original_prec:.4f}")
    print(f"  召回率: {original_rec:.4f}")
    print(f"  F1分数: {original_f1:.4f}")
    
    print(f"\n【对抗攻击结果】")
    print(f"{'攻击方法':<15} {'成功数':<10} {'成功率':<10}")
    print("-" * 40)
    for name, result in attack_results.items():
        print(f"{name:<15} {result['success']:<10} {result['rate']:.2%}")
    
    print(f"\n【消融实验结果】")
    print(f"{'扰动组件':<20} {'成功率':<10}")
    print("-" * 35)
    for name, result in sorted(ablation_results.items(), key=lambda x: x[1]['rate'], reverse=True):
        print(f"{name:<20} {result['rate']:.2%}")
    
    # 找出最有效的攻击组件
    best_component = max(ablation_results.items(), key=lambda x: x[1]['rate'])
    print(f"\n【结论】最有效攻击: {best_component[0]} (成功率 {best_component[1]['rate']:.2%})")
    
    # 展示成功案例
    if success_examples:
        print("\n" + "=" * 50)
        print("攻击成功案例")
        print("=" * 50)
        for i, ex in enumerate(success_examples[:3], 1):
            print(f"\n【案例 {i}】 {ex['pred_change']}")
            print(f"原始: {ex['original']}...")
            print(f"扰动: {ex['perturbed']}...")
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    results = {
        'model': args.model,
        'original_accuracy': original_acc,
        'original_precision': original_prec,
        'original_recall': original_rec,
        'original_f1': original_f1,
        'attack_results': {k: {'success': v['success'], 'rate': v['rate']} 
                          for k, v in attack_results.items()},
        'ablation_study': {k: {'success': v['success'], 'rate': v['rate']} 
                          for k, v in ablation_results.items()},
        'success_examples': success_examples[:5]
    }
    
    with open(f'results/{args.model}_experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ 结果已保存到 results/{args.model}_experiment_results.json")
    print("\n实验完成!")


if __name__ == '__main__':
    main()
