"""
欺诈对话对抗攻击实验脚本
实现对 BERT 和 SVM 模型的对抗攻击，并评估攻击效果
"""

import os
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
from collections import defaultdict

# 导入自定义模块
from anthro_chinese import FraudDialoguePerturbation
from data_processor import load_fraud_dataset, FraudDataset

# 深度学习库
try:
    import torch
    from torch.utils.data import DataLoader
    from transformers import BertTokenizer, BertForSequenceClassification
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("警告: PyTorch 未安装，无法使用 BERT 模型")

# 传统机器学习库
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class AttackEvaluator:
    """
    对抗攻击评估器
    支持对 BERT 和 SVM 模型的攻击效果评估
    """
    
    def __init__(self, model_type: str = 'bert', model_path: Optional[str] = None):
        """
        初始化评估器
        """
        self.model_type = model_type
        self.model = None
        self.tokenizer = None
        self.vectorizer = None
        
        if model_type == 'bert' and TORCH_AVAILABLE:
            self._init_bert_model(model_path)
        elif model_type == 'svm':
            self._init_svm_model()
            
        # 初始化对抗攻击器
        self.attacker = FraudDialoguePerturbation()
        
    def _init_bert_model(self, model_path: Optional[str] = None):
        """初始化 BERT 模型"""
        model_name = model_path or './bert-base-chinese'
        
        try:
            self.tokenizer = BertTokenizer.from_pretrained(model_name)
            self.model = BertForSequenceClassification.from_pretrained(
                model_name, num_labels=2
            )
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model.to(self.device)
            self.model.eval()
            print(f"BERT 模型已加载: {model_name}")
            print(f"使用设备: {self.device}")
        except Exception as e:
            print(f"加载 BERT 模型失败: {e}")
            
    def _init_svm_model(self):
        """初始化 SVM 模型"""
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2)
        )
        self.model = SVC(kernel='rbf', probability=True)
        print("SVM 模型已初始化")
        
    def train_svm(self, X_train: List[str], y_train: List[int]):
        """训练 SVM 模型"""
        if self.model_type != 'svm':
            raise ValueError("当前模型类型不是 SVM")
            
        print("训练 SVM 模型...")
        X_vec = self.vectorizer.fit_transform(X_train)
        self.model.fit(X_vec, y_train)
        print("SVM 模型训练完成")
        
    def predict(self, texts: List[str]) -> List[int]:
        """
        预测文本类别
        """
        if self.model_type == 'bert' and TORCH_AVAILABLE:
            return self._predict_bert(texts)
        elif self.model_type == 'svm':
            return self._predict_svm(texts)
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")
            
    def _predict_bert(self, texts: List[str]) -> List[int]:
        """使用 BERT 预测"""
        predictions = []
        
        for text in texts:
            inputs = self.tokenizer(
                text,
                max_length=512,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
                predictions.append(pred)
                
        return predictions
        
    def _predict_svm(self, texts: List[str]) -> List[int]:
        """使用 SVM 预测"""
        X_vec = self.vectorizer.transform(texts)
        return self.model.predict(X_vec).tolist()
        
    def attack_single(self, text: str, true_label: int,
                      max_perturbations: int = 5) -> Dict:
        """
        对单条文本执行攻击
        """
        # 原始预测
        original_pred = self.predict([text])[0]
        
        # 生成对抗样本
        perturbed_text, _ = self.attacker.attack(text, max_perturbations=max_perturbations)
        
        # 对抗样本预测
        perturbed_pred = self.predict([perturbed_text])[0]
        
        # 判断攻击是否成功
        # 成功条件：原本正确分类，扰动后错误分类
        original_correct = (original_pred == true_label)
        attack_success = original_correct and (perturbed_pred != true_label)
        
        return {
            'original_text': text,
            'perturbed_text': perturbed_text,
            'true_label': true_label,
            'original_pred': original_pred,
            'perturbed_pred': perturbed_pred,
            'original_correct': original_correct,
            'attack_success': attack_success,
        }
        
    def evaluate_attack(self, texts: List[str], labels: List[int],
                        max_samples: int = None,
                        max_perturbations: int = 5) -> Dict:
        """
        批量评估攻击效果
        """
        if max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]
            
        results = []
        attack_success_count = 0
        original_correct_count = 0
        
        print(f"开始攻击评估，共 {len(texts)} 条样本...")
        
        for text, label in tqdm(zip(texts, labels), total=len(texts)):
            result = self.attack_single(text, label, max_perturbations)
            results.append(result)
            
            if result['original_correct']:
                original_correct_count += 1
            if result['attack_success']:
                attack_success_count += 1
                
        # 计算评估指标
        total = len(texts)
        original_accuracy = original_correct_count / total
        attack_success_rate = attack_success_count / original_correct_count if original_correct_count > 0 else 0
        
        # 攻击后的准确率
        perturbed_preds = [r['perturbed_pred'] for r in results]
        perturbed_accuracy = accuracy_score(labels, perturbed_preds)
        
        evaluation = {
            'total_samples': total,
            'original_correct': original_correct_count,
            'original_accuracy': original_accuracy,
            'attack_success_count': attack_success_count,
            'attack_success_rate': attack_success_rate,
            'perturbed_accuracy': perturbed_accuracy,
            'accuracy_drop': original_accuracy - perturbed_accuracy,
            'detailed_results': results,
        }
        
        return evaluation
        
    def print_evaluation_report(self, evaluation: Dict):
        """打印评估报告"""
        print("\n" + "=" * 60)
        print("对抗攻击评估报告")
        print("=" * 60)
        print(f"模型类型: {self.model_type.upper()}")
        print(f"总样本数: {evaluation['total_samples']}")
        print("-" * 40)
        print(f"原始准确率: {evaluation['original_accuracy']:.4f} ({evaluation['original_correct']} 条正确)")
        print(f"攻击成功率: {evaluation['attack_success_rate']:.4f} ({evaluation['attack_success_count']} 条攻击成功)")
        print(f"攻击后准确率: {evaluation['perturbed_accuracy']:.4f}")
        print(f"准确率下降: {evaluation['accuracy_drop']:.4f}")
        print("=" * 60)
        
        # 打印部分攻击成功的案例
        print("\n攻击成功案例示例:")
        success_cases = [r for r in evaluation['detailed_results'] if r['attack_success']]
        for i, case in enumerate(success_cases[:5]):
            print(f"\n案例 {i+1}:")
            print(f"  原始文本: {case['original_text'][:80]}...")
            print(f"  扰动文本: {case['perturbed_text'][:80]}...")
            print(f"  真实标签: {'欺诈' if case['true_label'] == 1 else '正常'}")
            print(f"  原始预测: {'欺诈' if case['original_pred'] == 1 else '正常'} -> 扰动后: {'欺诈' if case['perturbed_pred'] == 1 else '正常'}")


class AblationStudy:
    """
    消融实验，比较不同扰动策略的效果
    """
    
    def __init__(self, evaluator: AttackEvaluator):
        self.evaluator = evaluator
        self.attacker = FraudDialoguePerturbation()
        
    def run_ablation(self, texts: List[str], labels: List[int],
                     max_samples: int = 100) -> Dict:
        """
        运行消融实验
        
        比较扰动模式:
        1. 仅同音字扰动 (pinyin)
        2. 仅形近字扰动 (shape)
        3. 仅字符替代 (sub)
        4. 混合扰动 (all)
        """
        modes = ['pinyin', 'shape', 'sub', 'all']
        results = {}
        
        if max_samples:
            texts = texts[:max_samples]
            labels = labels[:max_samples]
            
        print("开始消融实验...")
        
        for mode in modes:
            print(f"\n测试扰动模式: {mode}")
            
            attack_success = 0
            original_correct = 0
            
            for text, label in tqdm(zip(texts, labels), total=len(texts)):
                # 原始预测
                orig_pred = self.evaluator.predict([text])[0]
                
                if orig_pred == label:
                    original_correct += 1
                    
                    # 生成特定模式的扰动
                    perturbed = self.attacker.generate_perturbation(
                        text, 
                        perturbation_ratio=0.3,
                        mode=mode
                    )
                    
                    # 扰动后预测
                    pert_pred = self.evaluator.predict([perturbed])[0]
                    
                    if pert_pred != label:
                        attack_success += 1
                        
            success_rate = attack_success / original_correct if original_correct > 0 else 0
            results[mode] = {
                'attack_success': attack_success,
                'original_correct': original_correct,
                'success_rate': success_rate,
            }
            
            print(f"  攻击成功率: {success_rate:.4f}")
            
        return results
        
    def print_ablation_report(self, results: Dict):
        """打印消融实验报告"""
        print("\n" + "=" * 60)
        print("消融实验报告")
        print("=" * 60)
        print(f"{'扰动模式':<15} {'攻击成功数':<12} {'成功率':<10}")
        print("-" * 40)
        
        for mode, data in results.items():
            mode_name = {
                'pinyin': '同音字扰动',
                'shape': '形近字扰动',
                'sub': '字符替代',
                'all': '混合扰动',
            }.get(mode, mode)
            
            print(f"{mode_name:<15} {data['attack_success']:<12} {data['success_rate']:.4f}")
            
        print("=" * 60)


def run_experiment(csv_path: str, model_type: str = 'svm',
                   max_train_samples: int = None,
                   max_test_samples: int = 500):
    """
    运行完整实验
    """
    print("=" * 60)
    print("欺诈对话对抗攻击实验")
    print("=" * 60)
    
    # 1. 加载数据
    print("\n[1] 加载数据...")
    df = load_fraud_dataset(csv_path)
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_text'].values,
        df['label'].values,
        test_size=0.2,
        random_state=42,
        stratify=df['label'].values
    )
    
    if max_train_samples:
        X_train = X_train[:max_train_samples]
        y_train = y_train[:max_train_samples]
        
    # 2. 初始化评估器
    print(f"\n[2] 初始化 {model_type.upper()} 模型...")
    evaluator = AttackEvaluator(model_type=model_type)
    
    # 3. 训练模型 (仅 SVM)
    if model_type == 'svm':
        print("\n[3] 训练模型...")
        evaluator.train_svm(X_train.tolist(), y_train.tolist())
        
    # 4. 评估原始性能
    print("\n[4] 评估原始模型性能...")
    original_preds = evaluator.predict(X_test.tolist()[:max_test_samples])
    original_accuracy = accuracy_score(y_test[:max_test_samples], original_preds)
    print(f"原始测试集准确率: {original_accuracy:.4f}")
    
    # 5. 对抗攻击评估
    print("\n[5] 执行对抗攻击...")
    attack_evaluation = evaluator.evaluate_attack(
        X_test.tolist(),
        y_test.tolist(),
        max_samples=max_test_samples
    )
    evaluator.print_evaluation_report(attack_evaluation)
    
    # 6. 消融实验
    print("\n[6] 运行消融实验...")
    ablation = AblationStudy(evaluator)
    ablation_results = ablation.run_ablation(
        X_test.tolist(),
        y_test.tolist(),
        max_samples=min(100, max_test_samples)
    )
    ablation.print_ablation_report(ablation_results)
    
    # 7. 保存结果
    print("\n[7] 保存实验结果...")
    save_results(attack_evaluation, ablation_results, model_type)
    
    return attack_evaluation, ablation_results


def save_results(attack_eval: Dict, ablation_results: Dict, model_type: str):
    """保存实验结果"""
    os.makedirs('results', exist_ok=True)
    
    # 保存攻击评估结果 (不含详细结果，太大)
    eval_summary = {k: v for k, v in attack_eval.items() if k != 'detailed_results'}
    with open(f'results/{model_type}_attack_evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(eval_summary, f, ensure_ascii=False, indent=2)
        
    # 保存消融实验结果
    with open(f'results/{model_type}_ablation_study.json', 'w', encoding='utf-8') as f:
        json.dump(ablation_results, f, ensure_ascii=False, indent=2)
        
    # 保存攻击成功案例
    success_cases = [r for r in attack_eval['detailed_results'] if r['attack_success']]
    with open(f'results/{model_type}_success_cases.json', 'w', encoding='utf-8') as f:
        json.dump(success_cases[:100], f, ensure_ascii=False, indent=2)
        
    print(f"结果已保存到 results/ 目录")


if __name__ == '__main__':
    import sys
    
    # 默认参数
    csv_path = sys.argv[1] if len(sys.argv) > 1 else 'fraud_dialogue_dataset.csv'
    model_type = sys.argv[2] if len(sys.argv) > 2 else 'svm'
    
    if os.path.exists(csv_path):
        run_experiment(
            csv_path=csv_path,
            model_type=model_type,
            max_train_samples=5000,
            max_test_samples=500
        )
    else:
        print(f"数据文件不存在: {csv_path}")
        print("使用方法: python attack_evaluation.py <csv_path> [model_type]")
        print("model_type: 'svm' 或 'bert'")
