"""
中文欺诈对话对抗攻击实验 - 主入口
基于 ANTHRO 算法的中文适配版本
"""

import argparse
import os
import sys

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anthro_chinese import ChineseANTHRO, FraudDialoguePerturbation
from data_processor import load_fraud_dataset, prepare_data_for_training
from attack_evaluation import run_experiment, AttackEvaluator, AblationStudy


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='中文欺诈对话对抗攻击实验',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 使用 SVM 模型进行攻击实验
    python main.py --data fraud_dialogue.csv --model svm
    
    # 使用 BERT 模型进行攻击实验
    python main.py --data fraud_dialogue.csv --model bert --max_test 500
    
    # 仅测试扰动功能
    python main.py --demo
        """
    )
    
    parser.add_argument('--data', type=str, default='fraud_dialogue.csv',
                        help='数据集CSV文件路径')
    parser.add_argument('--model', type=str, default='svm',
                        choices=['svm', 'bert'],
                        help='模型类型: svm 或 bert')
    parser.add_argument('--max_train', type=int, default=5000,
                        help='最大训练样本数')
    parser.add_argument('--max_test', type=int, default=500,
                        help='最大测试样本数')
    parser.add_argument('--demo', action='store_true',
                        help='运行扰动演示')
    parser.add_argument('--ablation', action='store_true',
                        help='运行消融实验')
    parser.add_argument('--output', type=str, default='results',
                        help='结果输出目录')
    
    return parser.parse_args()


def run_demo():
    """运行扰动演示"""
    print("=" * 60)
    print("中文 ANTHRO 对抗扰动演示")
    print("=" * 60)
    
    # 初始化扰动器
    anthro = FraudDialoguePerturbation()
    
    # 测试文本
    test_texts = [
        "您好，这里是银行客服中心，您的账户存在安全风险，请立即转账到安全账户。",
        "恭喜您中奖了，请点击链接领取红包，需要输入验证码完成验证。",
        "您的快递包裹有问题，请联系客服处理，可能需要支付手续费。",
        "我是公安局的警察，您涉嫌洗钱，需要将资金转到安全账户配合调查。",
        "您的贷款额度已审批通过，请点击链接下载APP提现，只需支付少量手续费。",
    ]
    
    print("\n" + "-" * 40)
    for i, text in enumerate(test_texts, 1):
        print(f"\n【测试 {i}】")
        print(f"原始文本: {text}")
        
        # 识别关键词
        keywords = anthro.get_fraud_keywords(text)
        print(f"识别关键词: {keywords[:5]}")  # 只显示前5个
        
        # 生成扰动 (不同模式)
        modes = {
            '同音字': 'pinyin',
            '形近字': 'shape', 
            '混合': 'all',
        }
        
        for mode_name, mode in modes.items():
            perturbed = anthro.generate_perturbation(text, mode=mode, perturbation_ratio=0.3)
            print(f"{mode_name}扰动: {perturbed}")
            
        print("-" * 40)
        
    # 测试单词扰动
    print("\n" + "=" * 60)
    print("单词扰动测试")
    print("=" * 60)
    
    test_words = ['转账', '验证码', '银行', '密码', '客服', '警察', '链接']
    
    for word in test_words:
        similars = anthro.get_similars(word, level=1, distance=2, mode='all', strict=False)
        print(f"'{word}' 的扰动: {similars if len(similars) <= 10 else list(similars)[:10]}")


def main():
    """主函数"""
    args = parse_args()
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    if args.demo:
        # 运行演示
        run_demo()
        return
        
    # 检查数据文件
    if not os.path.exists(args.data):
        print(f"错误: 数据文件不存在: {args.data}")
        print("\n请提供正确的CSV数据文件路径，数据格式应包含:")
        print("  - specific_dialogue_content: 对话内容")
        print("  - is_fraud: 是否欺诈 (TRUE/FALSE)")
        print("\n你也可以使用 --demo 参数运行扰动演示:")
        print("  python main.py --demo")
        sys.exit(1)
        
    # 运行完整实验
    print("\n" + "=" * 60)
    print(f"运行实验: {args.model.upper()} 模型")
    print(f"数据集: {args.data}")
    print(f"最大训练样本: {args.max_train}")
    print(f"最大测试样本: {args.max_test}")
    print("=" * 60)
    
    try:
        attack_eval, ablation_results = run_experiment(
            csv_path=args.data,
            model_type=args.model,
            max_train_samples=args.max_train,
            max_test_samples=args.max_test
        )
        
        print("\n实验完成！结果已保存到 results/ 目录")
        
    except Exception as e:
        print(f"\n实验运行出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
