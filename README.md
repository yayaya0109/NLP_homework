# 中文版 ANTHRO 对抗性文本扰动算法

## 项目简介

本项目基于 ACL 2022 论文"Perturbations in the Wild: Leveraging Human-Written Text Perturbations for Realistic Adversarial Attack and Defense" 进行中文适配，用于欺诈对话检测场景的对抗性数据改写。

原始项目：https://github.com/lethaiq/perturbations-in-the-wild

### 核心改进

由于原始 ANTHRO 算法基于英文 SOUNDEX++ 编码，无法直接用于中文，本项目进行了以下适配：

| 原英文方案  | 中文适配方案 |
|-----------|-------------|
| SOUNDEX++ 编码 |拼音编码(使用 pypinyin) |
| 英文同音词 | 中文同音字/词 |
| 字符级扰动 (拼写错误) | 形近字替换 + 数字/字母替代 |
| 空格分词 | jieba 中文分词 |

## 项目结构

```
chinese_anthro/
├──bert-base-chinese文件夹
├──result文件夹
├── anthro_chinese.py      # 核心算法实现 (中文ANTHRO)
├── data_processor.py      # 数据加载和预处理
├── attack_evaluation.py   # 攻击评估和消融实验
├── main.py                # 主入口脚本
├── requirements.txt       # 依赖列表
├── run_full_experiment.py # 全流程脚本
├──测试集结果.csv            # 数据集
├──训练集结果.csv            # 数据集
└── README.md              # 项目说明
```

## 安装依赖

```bash
pip install jieba pypinyin pandas numpy scikit-learn tqdm

# 如果使用 BERT 模型
pip install torch transformers
```

## 数据集格式
CSV 文件包含以下列：

| 列名 | 说明 | 示例 |
|------|------|--|
| `specific_dialogue_content` | 对话内容 | "left: 您好... right: 好的..." |
| `is_fraud` | 是否欺诈 | TRUE / FALSE |
| `fraud_type` | 欺诈类型 | 客服诈骗、贷款诈骗等 |
| `interaction_strategy` | 交互策略 |  |
| `call_type` | 通话类型 | 咨询客服、推销等 |

## 使用方法

### 1. 运行扰动演示

```bash
python main.py --demo
```

### 2. 运行完整实验

使用 SVM 模型：
```bash
python run_full_experiment.py --data 测试集结果.csv --model svm
or
python main.py --data 测试集结果.csv --model svm
```

使用 BERT 模型：
```bash
python run_full_experiment.py --data 测试集结果.csv --model bert
or
python main.py --data 测试集结果.csv --model bert

```

### 3. 在代码中使用

```python
from anthro_chinese import FraudDialoguePerturbation

# 初始化扰动器
anthro = FraudDialoguePerturbation()

# 原始欺诈对话
text = "您好，这里是银行客服中心，您的账户存在安全风险，请立即转账到安全账户。"

# 生成对抗扰动
perturbed, success = anthro.attack(text, max_perturbations=5)
print(f"原始: {text}")
print(f"扰动: {perturbed}")

# 获取单词的相似扰动
similars = anthro.get_similars('转账', level=1, distance=2, mode='all')
print(f"'转账' 的扰动词: {similars}")
```

## 中文扰动策略详解

### 1. 拼音编码 (替代 SOUNDEX++)

```python
def get_pinyin_code(word, level):
    """
    level=0: 仅声母 (最宽松)  "转账" -> "zhzh" -> "zz"
    level=1: 声母+韵母首    "转账" -> "zhazha" -> "zhazha"
    level=2: 完整拼音 (最严格) "转账" -> "zhuanzhang"
    """
```

### 2. 同音字扰动

基于拼音的同音字替换：
- 转 → 专、砖、传
- 账 → 帐、胀、张
- 验 → 检、俭

### 3. 形近字扰动

基于字形结构的相似字替换：
- 户 → 护、沪、互
- 码 → 玛、妈、吗
- 行 → 形、型、刑

### 4. 字符替代扰动

数字/字母替代汉字：
- 一 → 1、壹、yi
- 元 → ￥、圆、yuan
- 密码 → mima、秘码

### 5. 欺诈领域特定扰动

针对欺诈关键词的专门变体：
- 转账 → 转帐、划转、汇款、过账
- 验证码 → 验证吗、校验码、yzm
- 银行 → 银航、银杭
- 客服 → 客副、客fu、kefu

## 实验输出

运行实验后，结果保存在 `results/` 目录：

- `{model}_attack_evaluation.json`: 攻击评估结果
- `{model}_ablation_study.json`: 消融实验结果
- `{model}_success_cases.json`: 攻击成功案例

### 评估指标

| 指标 | 说明 |
|------|------|
| 原始准确率 | 模型在原始测试集上的准确率 |
| 攻击成功率 | 原本正确分类，扰动后错误分类的比例 |
| 攻击后准确率 | 模型在扰动测试集上的准确率 |
| 准确率下降 | 原始准确率 - 攻击后准确率 |

### 消融实验

比较不同扰动策略的效果：
- `pinyin`: 仅同音字扰动
- `shape`: 仅形近字扰动
- `sub`: 仅字符替代
- `all`: 混合扰动

## 与原始论文的对应关系

| 论文概念 | 本项目实现 |
|---------|-----------|
| SOUNDEX++ 编码 | `get_pinyin_code()` 函数 |
| 哈希表 H(c) | `pinyin_hash` / `initial_hash` 字典 |
| Levenshtein 距离 | `levenshtein_distance()` 函数 |
| 扰动检索公式 | `get_similars()` 方法 |
| ANTHRO 攻击算法 | `attack()` 方法 |
| Score 函数 (关键词重要性) | `get_fraud_keywords()` 方法 |

## 参考文献

```bibtex
@article{le2022perturbations,
  title={Perturbations in the Wild: Leveraging Human-Written Text Perturbations 
         for Realistic Adversarial Attack and Defense},
  author={Le, Thai and Lee, Jooyoung and Yen, Kevin and Hu, Yifan and Lee, Dongwon},
  journal={60th Annual Meeting of the Association for Computational Linguistics (ACL)},
  year={2022}
}
```

## License

MIT License
