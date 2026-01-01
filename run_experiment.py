"""
中文欺诈对话对抗攻击实验
直接适配用户的数据集格式
"""

import os
import re
import json
import random
import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 检查并安装依赖
def check_dependencies():
    """检查并提示安装依赖"""
    missing = []
    
    try:
        import jieba
    except ImportError:
        missing.append('jieba')
        
    try:
        from pypinyin import lazy_pinyin
    except ImportError:
        missing.append('pypinyin')
        
    if missing:
        print(f"缺少依赖: {', '.join(missing)}")
        print(f"请运行: pip install {' '.join(missing)}")
        return False
    return True

# 尝试导入，如果失败则使用简化版本
try:
    import jieba
    import jieba.posseg as pseg
    from pypinyin import lazy_pinyin
    CHINESE_NLP_AVAILABLE = True
except ImportError:
    CHINESE_NLP_AVAILABLE = False
    print("警告: jieba/pypinyin 未安装，将使用简化版分词")


# 简化版分词 (无依赖)
def simple_tokenize(text: str) -> List[str]:
    """简单的中文分词（字符级）"""
    # 移除标点，按字符分割
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    chars = list(text.replace(' ', ''))
    return chars

def tokenize(text: str) -> List[str]:
    """分词函数"""
    if CHINESE_NLP_AVAILABLE:
        return list(jieba.cut(text))
    else:
        return simple_tokenize(text)


# ==================== 中文扰动规则 ====================
class ChinesePerturbation:
    """中文对抗扰动生成器"""
    
    def __init__(self):
        # 形近字映射 (欺诈领域重点词汇)
        self.similar_chars = {
            '转': ['专', '砖', '传', '赚'],
            '账': ['帐', '胀', '张', '障'],
            '户': ['护', '沪', '互', '戸'],
            '款': ['欵', '宽'],
            '验': ['检', '俭', '剑'],
            '码': ['玛', '妈', '吗', '骂'],
            '银': ['很', '根', '跟', '垠'],
            '行': ['形', '型', '刑', '邢'],
            '卡': ['咔', '喀', '佧', '咖'],
            '号': ['呺', '昊', '浩', '耗'],
            '密': ['蜜', '秘', '泌', '宓'],
            '钱': ['浅', '钳', '潜', '谦'],
            '安': ['按', '案', '暗', '鞍'],
            '全': ['铨', '诠', '栓', '痊'],
            '输': ['书', '殊', '疏', '蔬'],
            '入': ['人', '仁', '忍', '刃'],
            '冻': ['动', '栋', '洞', '侗'],
            '结': ['洁', '捷', '节', '劫'],
            '退': ['腿', '褪', '蜕', '颓'],
            '链': ['连', '联', '莲', '怜'],
            '接': ['节', '捷', '洁', '劫'],
            '登': ['灯', '等', '邓', '蹬'],
            '录': ['路', '露', '鹿', '碌'],
            '客': ['咳', '克', '刻', '课'],
            '服': ['福', '伏', '扶', '幅'],
            '诈': ['炸', '榨', '乍', '咋'],
            '骗': ['偏', '篇', '编', '扁'],
            '警': ['景', '境', '竞', '井'],
            '领': ['令', '岭', '零', '铃'],
            '取': ['去', '趣', '娶', '曲'],
            '奖': ['将', '浆', '蒋', '桨'],
            '免': ['勉', '棉', '面', '眠'],
            '费': ['废', '肺', '沸', '吠'],
            '申': ['伸', '身', '深', '神'],
            '请': ['清', '青', '情', '晴'],
            '理': ['里', '礼', '李', '鲤'],
            '资': ['紫', '滋', '姿', '咨'],
            '金': ['斤', '今', '进', '近'],
            '信': ['心', '新', '欣', '芯'],
            '息': ['熄', '媳', '悉', '惜'],
            '贷': ['代', '戴', '带', '待'],
            '还': ['环', '坏', '换', '缓'],
            '风': ['丰', '封', '峰', '锋'],
            '险': ['显', '现', '县', '献'],
        }
        
        # 欺诈关键词变体
        self.keyword_variants = {
            '转账': ['转帐', '划转', '汇款', '过账', '转钱', 'zhuanzhang'],
            '账户': ['帐户', '账号', '帐号', '用户', 'zhanghu'],
            '密码': ['秘码', '口令', '暗码', 'mima', '秘碼'],
            '验证码': ['验证吗', '校验码', '确认码', 'yzm', '驗證碼'],
            '银行': ['银航', '银杭', '銀行', 'yinhang'],
            '信用卡': ['信用咔', '信用ka', 'xyk', '信用咖'],
            '客服': ['客副', '客fu', 'kefu', '咳服'],
            '安全': ['安权', 'anquan', '按全', '暗全'],
            '冻结': ['动结', '冻洁', '封存', '冻節'],
            '风险': ['风显', '危险', 'fengxian'],
            '链接': ['连接', '连结', '链jie', '鏈接'],
            '点击': ['点及', '点ji', '點擊'],
            '下载': ['下栽', '下zai', '下載'],
            '登录': ['登陆', '登入', 'denglu', '登錄'],
            '贷款': ['贷欵', '借款', '借钱', '貸款'],
            '利息': ['利率', '利钱', '利細'],
            '手续费': ['手续废', '费用', '工本费'],
            '中奖': ['中将', 'zhongjiang', '得奖', '中獎'],
            '返现': ['反现', '返xian', '返钱'],
            '红包': ['洪包', '红bao', '红宝', '紅包'],
            '优惠': ['优会', '优慧', '優惠'],
            '免费': ['勉费', '免废', '不要钱', '免費'],
            '警察': ['jingcha', '警查', '警茶', '員警'],
            '公安': ['公按', 'gongan', '公暗'],
            '快递': ['快di', '快地', 'kuaidi', '快遞'],
        }
        
        # 关键词重要性权重
        self.keyword_weights = {
            '转账': 5, '验证码': 5, '密码': 5, '冻结': 5, '警察': 5,
            '公安': 5, '银行': 4, '账户': 4, '安全': 4, '客服': 4,
            '链接': 4, '点击': 4, '下载': 4, '登录': 3, '中奖': 3,
            '返现': 3, '贷款': 3, '利息': 3, '手续费': 3, '红包': 2,
            '优惠': 2, '免费': 2, '快递': 2,
        }
        
    def get_similar_char(self, char: str) -> Optional[str]:
        """获取形近字"""
        if char in self.similar_chars:
            return random.choice(self.similar_chars[char])
        return None
        
    def get_keyword_variant(self, word: str) -> Optional[str]:
        """获取关键词变体"""
        if word in self.keyword_variants:
            return random.choice(self.keyword_variants[word])
        return None
        
    def find_keywords(self, text: str) -> List[Tuple[str, int, int]]:
        """查找文本中的关键词及其位置"""
        found = []
        for keyword, weight in self.keyword_weights.items():
            start = 0
            while True:
                pos = text.find(keyword, start)
                if pos == -1:
                    break
                found.append((keyword, pos, weight))
                start = pos + 1
        # 按权重降序排列
        found.sort(key=lambda x: x[2], reverse=True)
        return found
        
    def perturb_text(self, text: str, max_perturbations: int = 5, 
                     mode: str = 'all') -> str:
        """
        对文本进行扰动
        
        Args:
            text: 原始文本
            max_perturbations: 最大扰动次数
            mode: 扰动模式 ('keyword', 'char', 'all')
        """
        perturbed = text
        perturbation_count = 0
        
        if mode in ['keyword', 'all']:
            # 1. 关键词级扰动
            keywords = self.find_keywords(perturbed)
            for keyword, pos, weight in keywords:
                if perturbation_count >= max_perturbations:
                    break
                variant = self.get_keyword_variant(keyword)
                if variant and variant != keyword:
                    perturbed = perturbed.replace(keyword, variant, 1)
                    perturbation_count += 1
                    
        if mode in ['char', 'all'] and perturbation_count < max_perturbations:
            # 2. 字符级扰动 (针对未被关键词扰动的字符)
            chars = list(perturbed)
            indices = list(range(len(chars)))
            random.shuffle(indices)
            
            for idx in indices:
                if perturbation_count >= max_perturbations:
                    break
                char = chars[idx]
                similar = self.get_similar_char(char)
                if similar:
                    chars[idx] = similar
                    perturbation_count += 1
                    
            perturbed = ''.join(chars)
            
        return perturbed


# 数据处理
def clean_text(text: str) -> str:
    """清洗文本"""
    if pd.isna(text):
        return ""
    text = str(text)
    # 去除引号和首尾空格
    text = text.replace('"', '').replace("'", '').strip()
    # 统一空白字符
    text = re.sub(r'\s+', ' ', text)
    return text


def load_dataset(train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载训练集和测试集
    """
    # 尝试不同编码
    encodings = ['utf-8', 'gbk', 'gb2312', 'gb18030', 'utf-16', 'latin1']
    
    train_df = None
    test_df = None
    
    for encoding in encodings:
        try:
            if train_df is None:
                train_df = pd.read_csv(train_path, encoding=encoding)
                print(f"训练集使用 {encoding} 编码读取成功")
        except:
            continue
            
    for encoding in encodings:
        try:
            if test_df is None:
                test_df = pd.read_csv(test_path, encoding=encoding)
                print(f"测试集使用 {encoding} 编码读取成功")
        except:
            continue
            
    if train_df is None:
        raise ValueError(f"无法读取训练集: {train_path}")
    if test_df is None:
        raise ValueError(f"无法读取测试集: {test_path}")
        
    print(f"\n训练集: {len(train_df)} 条")
    print(f"测试集: {len(test_df)} 条")
    print(f"列名: {list(train_df.columns)}")
    
    return train_df, test_df


def prepare_data(df: pd.DataFrame, 
                 text_col: str = 'specific_dialogue_content',
                 label_col: str = 'is_fraud') -> Tuple[List[str], List[int]]:
    """
    准备训练/测试数据
    """
    # 清洗文本
    texts = df[text_col].apply(clean_text).tolist()
    
    # 处理标签
    labels = df[label_col].apply(
        lambda x: 1 if str(x).upper() in ['TRUE', '1', 'YES', 'T'] else 0
    ).tolist()
    
    # 统计
    fraud_count = sum(labels)
    normal_count = len(labels) - fraud_count
    print(f"  欺诈样本: {fraud_count} ({fraud_count/len(labels)*100:.1f}%)")
    print(f"  正常样本: {normal_count} ({normal_count/len(labels)*100:.1f}%)")
    
    return texts, labels


# 模型训练与评估
class FraudClassifier:
    """欺诈对话分类器 (SVM)"""
    
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            tokenizer=tokenize if CHINESE_NLP_AVAILABLE else None
        )
        self.model = SVC(kernel='rbf', probability=True, C=1.0)
        
    def train(self, texts: List[str], labels: List[int]):
        """训练模型"""
        print("正在训练 SVM 模型...")
        X = self.vectorizer.fit_transform(texts)
        self.model.fit(X, labels)
        print("训练完成!")
        
    def predict(self, texts: List[str]) -> List[int]:
        """预测"""
        X = self.vectorizer.transform(texts)
        return self.model.predict(X).tolist()
        
    def evaluate(self, texts: List[str], labels: List[int]) -> Dict:
        """评估模型"""
        preds = self.predict(texts)
        
        accuracy = accuracy_score(labels, preds)
        precision = precision_score(labels, preds, zero_division=0)
        recall = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'predictions': preds
        }


# 对抗攻击实验
def run_attack_experiment(classifier: FraudClassifier,
                          texts: List[str],
                          labels: List[int],
                          max_samples: int = None,
                          max_perturbations: int = 5) -> Dict:
    """
    运行对抗攻击实验
    """
    perturbator = ChinesePerturbation()
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        
    print(f"\n开始对抗攻击实验 (共 {len(texts)} 条样本)...")
    
    # 原始预测
    original_preds = classifier.predict(texts)
    original_correct = sum(1 for p, l in zip(original_preds, labels) if p == l)
    
    # 生成扰动样本并预测
    perturbed_texts = []
    attack_success_count = 0
    success_cases = []
    
    for i, (text, label) in enumerate(zip(texts, labels)):
        # 生成扰动
        perturbed = perturbator.perturb_text(text, max_perturbations=max_perturbations)
        perturbed_texts.append(perturbed)
        
        # 检查攻击是否成功
        orig_pred = original_preds[i]
        if orig_pred == label:  # 原本预测正确
            pert_pred = classifier.predict([perturbed])[0]
            if pert_pred != label:  # 扰动后预测错误
                attack_success_count += 1
                success_cases.append({
                    'original': text[:200],
                    'perturbed': perturbed[:200],
                    'true_label': '欺诈' if label == 1 else '正常',
                    'original_pred': '欺诈' if orig_pred == 1 else '正常',
                    'perturbed_pred': '欺诈' if pert_pred == 1 else '正常',
                })
                
        if (i + 1) % 100 == 0:
            print(f"  已处理 {i+1}/{len(texts)} 条...")
            
    # 扰动后评估
    perturbed_preds = classifier.predict(perturbed_texts)
    perturbed_correct = sum(1 for p, l in zip(perturbed_preds, labels) if p == l)
    
    # 计算指标
    original_accuracy = original_correct / len(labels)
    perturbed_accuracy = perturbed_correct / len(labels)
    attack_success_rate = attack_success_count / original_correct if original_correct > 0 else 0
    
    results = {
        'total_samples': len(labels),
        'original_accuracy': original_accuracy,
        'original_correct': original_correct,
        'perturbed_accuracy': perturbed_accuracy,
        'perturbed_correct': perturbed_correct,
        'attack_success_count': attack_success_count,
        'attack_success_rate': attack_success_rate,
        'accuracy_drop': original_accuracy - perturbed_accuracy,
        'success_cases': success_cases[:20],  # 保存前20个成功案例
    }
    
    return results


def run_ablation_study(classifier: FraudClassifier,
                       texts: List[str],
                       labels: List[int],
                       max_samples: int = 200) -> Dict:
    """
    消融实验：比较不同扰动策略
    """
    perturbator = ChinesePerturbation()
    
    if max_samples:
        texts = texts[:max_samples]
        labels = labels[:max_samples]
        
    print(f"\n开始消融实验 (共 {len(texts)} 条样本)...")
    
    modes = {
        'keyword': '仅关键词扰动',
        'char': '仅字符扰动',
        'all': '混合扰动',
    }
    
    results = {}
    original_preds = classifier.predict(texts)
    original_correct = sum(1 for p, l in zip(original_preds, labels) if p == l)
    
    for mode, mode_name in modes.items():
        print(f"  测试: {mode_name}...")
        
        attack_success = 0
        for i, (text, label) in enumerate(zip(texts, labels)):
            if original_preds[i] == label:  # 原本正确
                perturbed = perturbator.perturb_text(text, max_perturbations=5, mode=mode)
                pert_pred = classifier.predict([perturbed])[0]
                if pert_pred != label:
                    attack_success += 1
                    
        success_rate = attack_success / original_correct if original_correct > 0 else 0
        results[mode] = {
            'name': mode_name,
            'attack_success': attack_success,
            'success_rate': success_rate,
        }
        print(f"    攻击成功率: {success_rate:.4f}")
        
    return results


# 主函数
def main():
    """主函数"""
    print("=" * 70)
    print("中文欺诈对话对抗攻击实验")
    print("基于 ANTHRO 算法的中文适配版本")
    print("=" * 70)
    
    # 配置
    TRAIN_FILE = "训练集结果.csv"
    TEST_FILE = "测试集结果.csv"
    TEXT_COL = "specific_dialogue_content"
    LABEL_COL = "is_fraud"
    MAX_TRAIN = None  # None 表示使用全部
    MAX_TEST = 500    # 测试样本数
    
    # 检查文件
    if not os.path.exists(TRAIN_FILE):
        print(f"\n错误: 找不到训练集文件 '{TRAIN_FILE}'")
        print("请确保CSV文件在当前目录下")
        return
    if not os.path.exists(TEST_FILE):
        print(f"\n错误: 找不到测试集文件 '{TEST_FILE}'")
        print("请确保CSV文件在当前目录下")
        return
        
    # 1. 加载数据
    print("\n[1] 加载数据集...")
    train_df, test_df = load_dataset(TRAIN_FILE, TEST_FILE)
    
    print("\n处理训练集:")
    X_train, y_train = prepare_data(train_df, TEXT_COL, LABEL_COL)
    
    print("\n处理测试集:")
    X_test, y_test = prepare_data(test_df, TEXT_COL, LABEL_COL)
    
    if MAX_TRAIN:
        X_train = X_train[:MAX_TRAIN]
        y_train = y_train[:MAX_TRAIN]
        
    # 2. 训练模型
    print("\n[2] 训练分类模型...")
    classifier = FraudClassifier()
    classifier.train(X_train, y_train)
    
    # 3. 原始性能评估
    print("\n[3] 评估原始模型性能...")
    eval_results = classifier.evaluate(X_test[:MAX_TEST], y_test[:MAX_TEST])
    print(f"  准确率: {eval_results['accuracy']:.4f}")
    print(f"  精确率: {eval_results['precision']:.4f}")
    print(f"  召回率: {eval_results['recall']:.4f}")
    print(f"  F1分数: {eval_results['f1']:.4f}")
    
    # 4. 对抗攻击实验
    print("\n[4] 对抗攻击实验...")
    attack_results = run_attack_experiment(
        classifier, X_test, y_test, 
        max_samples=MAX_TEST, 
        max_perturbations=5
    )
    
    print("\n" + "=" * 50)
    print("对抗攻击结果")
    print("=" * 50)
    print(f"测试样本数: {attack_results['total_samples']}")
    print(f"原始准确率: {attack_results['original_accuracy']:.4f}")
    print(f"攻击后准确率: {attack_results['perturbed_accuracy']:.4f}")
    print(f"准确率下降: {attack_results['accuracy_drop']:.4f}")
    print(f"攻击成功数: {attack_results['attack_success_count']}")
    print(f"攻击成功率: {attack_results['attack_success_rate']:.4f}")
    
    # 5. 消融实验
    print("\n[5] 消融实验...")
    ablation_results = run_ablation_study(
        classifier, X_test, y_test, 
        max_samples=200
    )
    
    print("\n" + "=" * 50)
    print("消融实验结果")
    print("=" * 50)
    print(f"{'扰动策略':<15} {'攻击成功数':<12} {'成功率':<10}")
    print("-" * 40)
    for mode, data in ablation_results.items():
        print(f"{data['name']:<15} {data['attack_success']:<12} {data['success_rate']:.4f}")
        
    # 6. 展示攻击成功案例
    print("\n" + "=" * 50)
    print("攻击成功案例示例")
    print("=" * 50)
    for i, case in enumerate(attack_results['success_cases'][:5], 1):
        print(f"\n案例 {i}:")
        print(f"  原始: {case['original'][:80]}...")
        print(f"  扰动: {case['perturbed'][:80]}...")
        print(f"  真实标签: {case['true_label']}")
        print(f"  预测变化: {case['original_pred']} -> {case['perturbed_pred']}")
        
    # 7. 保存结果
    print("\n[6] 保存实验结果...")
    os.makedirs('results', exist_ok=True)
    
    # 保存攻击结果
    save_results = {
        'original_accuracy': attack_results['original_accuracy'],
        'perturbed_accuracy': attack_results['perturbed_accuracy'],
        'accuracy_drop': attack_results['accuracy_drop'],
        'attack_success_rate': attack_results['attack_success_rate'],
        'attack_success_count': attack_results['attack_success_count'],
        'ablation_study': {k: {'name': v['name'], 'success_rate': v['success_rate']} 
                          for k, v in ablation_results.items()}
    }
    
    with open('results/experiment_results.json', 'w', encoding='utf-8') as f:
        json.dump(save_results, f, ensure_ascii=False, indent=2)
        
    # 保存成功案例
    with open('results/success_cases.json', 'w', encoding='utf-8') as f:
        json.dump(attack_results['success_cases'], f, ensure_ascii=False, indent=2)
        
    print("结果已保存到 results/ 目录")
    print("\n实验完成!")


if __name__ == '__main__':
    main()
