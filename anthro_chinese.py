"""
中文版 ANTHRO 对抗性文本扰动算法
基于 ACL 2022 论文 "Perturbations in the Wild" 改编
适配中文欺诈对话检测场景
"""

import re
import json
import os
from collections import defaultdict
from typing import Dict, List, Set, Tuple, Optional

# 中文相关库
try:
    import jieba
    import jieba.posseg as pseg
except ImportError:
    print("请安装 jieba: pip install jieba")
    
try:
    from pypinyin import pinyin, Style, lazy_pinyin
except ImportError:
    print("请安装 pypinyin: pip install pypinyin")


class ChineseANTHRO:
    """
    中文版 ANTHRO 算法实现
    支持三种扰动模式:
    1. 同音字扰动 (基于拼音)
    2. 形近字扰动 (基于字形结构)
    3. 混合扰动 (字符级变换，如数字替代、繁简转换等)
    """
    
    def __init__(self):
        # 拼音哈希表: {拼音编码: {词汇集合}}
        self.pinyin_hash: Dict[str, Set[str]] = defaultdict(set)
        # 声母哈希表 (更宽松的匹配): {声母编码: {词汇集合}}
        self.initial_hash: Dict[str, Set[str]] = defaultdict(set)
        # 形近字映射表
        self.similar_shape_chars: Dict[str, List[str]] = {}
        # 同音字映射表
        self.homophone_chars: Dict[str, List[str]] = defaultdict(list)
        # 字符级扰动映射 (数字/字母替代汉字)
        self.char_substitution: Dict[str, List[str]] = {}
        # 确保同音字词典被构建
        self.build_homophone_dict()
        # 初始化扰动规则
        self._init_perturbation_rules()
        
    def _init_perturbation_rules(self):
        """初始化中文扰动规则"""
        
        # 形近字映射
        self.similar_shape_chars = {
            # 偏旁部首相近
            '转': ['专', '砖', '传'],
            '账': ['帐', '胀', '张'],
            '户': ['护', '沪', '互'],
            '款': ['歀', '欵'],
            '验': ['检', '验', '俭'],
            '码': ['玛', '妈', '吗'],
            '银': ['很', '根', '跟'],
            '行': ['形', '型', '刑'],
            '卡': ['咔', '喀', '佧'],
            '号': ['呺', '昊', '浩'],
            '密': ['蜜', '秘', '泌'],
            '钱': ['浅', '钳', '潜'],
            '安': ['按', '案', '暗'],
            '全': ['铨', '诠', '栓'],
            '输': ['书', '殊', '疏'],
            '入': ['人', '仁', '忍'],
            '冻': ['动', '栋', '洞'],
            '结': ['洁', '捷', '节'],
            '退': ['腿', '褪', '蜕'],
            '充': ['冲', '崇', '虫'],
            '值': ['植', '殖', '直'],
            '链': ['连', '联', '莲'],
            '接': ['节', '捷', '洁'],
            '登': ['灯', '等', '邓'],
            '录': ['路', '露', '鹿'],
            '客': ['咳', '克', '刻'],
            '服': ['福', '伏', '扶'],
            '诈': ['炸', '榨', '乍'],
            '骗': ['偏', '篇', '编'],
            '警': ['景', '境', '竞'],
            '察': ['查', '茶', '差'],
            '领': ['令', '岭', '零'],
            '取': ['去', '趣', '娶'],
            '奖': ['将', '浆', '蒋'],
            '免': ['勉', '棉', '面'],
            '费': ['废', '肺', '沸'],
            '申': ['伸', '身', '深'],
            '请': ['清', '青', '情'],
            '办': ['半', '拌', '伴'],
            '理': ['里', '礼', '李'],
            '资': ['紫', '滋', '姿'],
            '金': ['斤', '今', '进'],
            '期': ['其', '奇', '棋'],
            '限': ['险', '显', '现'],
            '信': ['心', '新', '欣'],
            '息': ['熄', '媳', '悉'],
            '贷': ['代', '戴', '带'],
            '还': ['环', '坏', '换'],
            '欠': ['歉', '倩', '芊'],
            '债': ['寨', '砦'],
        }
        
        # 数字/字母替代汉字
        self.char_substitution = {
            # 数字替代
            '一': ['1', '壹', 'yi'],
            '二': ['2', '贰', 'er'],
            '三': ['3', '叁', 'san'],
            '四': ['4', '肆', 'si'],
            '五': ['5', '伍', 'wu'],
            '六': ['6', '陆', 'liu'],
            '七': ['7', '柒', 'qi'],
            '八': ['8', '捌', 'ba'],
            '九': ['9', '玖', 'jiu'],
            '十': ['10', '拾', 'shi'],
            '零': ['0', '〇', 'ling'],
            '百': ['100', 'bai'],
            '千': ['1000', 'qian'],
            '万': ['10000', 'wan'],
            
            # 字母形近替代
            '人': ['Ren', '亻'],
            '口': ['kou', '囗'],
            '中': ['zhong', 'zhong1'],
            
            # 特殊符号替代
            '元': ['￥', '圆', 'yuan'],
        }
        
        # 常用欺诈领域同音字
        self._fraud_keywords = [
            '转账', '验证码', '安全', '账户', '密码', '银行', '客服',
            '退款', '充值', '链接', '登录', '冻结', '风险', '异常',
            '贷款', '利息', '额度', '审核', '提现', '手续费', '保证金',
            '中奖', '返现', '红包', '优惠', '免费', '赠送', '积分',
            '警察', '公安', '检察', '法院', '通缉', '涉嫌', '洗钱',
            '快递', '包裹', '物流', '签收', '代收', '货到付款',
        ]
        
    def get_pinyin_code(self, word: str, level: int = 1) -> str:
        """
        获取词汇的拼音编码 (类似 SOUNDEX++ 的功能)
        """
        try:
            pinyins = lazy_pinyin(word)
            if not pinyins:
                return word
                
            if level == 0:
                # 仅保留声母
                result = []
                for py in pinyins:
                    if py:
                        # 提取声母
                        initial = self._get_initial(py)
                        result.append(initial)
                return ''.join(result)
                
            elif level == 1:
                # 声母 + 韵母首字母
                result = []
                for py in pinyins:
                    if py:
                        initial = self._get_initial(py)
                        final_first = py[len(initial):len(initial)+1] if len(py) > len(initial) else ''
                        result.append(initial + final_first)
                return ''.join(result)
                
            else:  # level >= 2
                # 完整拼音
                return ''.join(pinyins)
                
        except Exception as e:
            return word
            
    def _get_initial(self, pinyin_str: str) -> str:
        """提取拼音的声母"""
        initials = ['zh', 'ch', 'sh', 'b', 'p', 'm', 'f', 'd', 't', 'n', 'l',
                   'g', 'k', 'h', 'j', 'q', 'x', 'r', 'z', 'c', 's', 'y', 'w']
        for ini in initials:
            if pinyin_str.startswith(ini):
                return ini
        return ''
        
    def levenshtein_distance(self, s1: str, s2: str) -> int:
        """计算 Levenshtein 编辑距离"""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
            
        prev_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            curr_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = prev_row[j + 1] + 1
                deletions = curr_row[j] + 1
                substitutions = prev_row[j] + (c1 != c2)
                curr_row.append(min(insertions, deletions, substitutions))
            prev_row = curr_row
        return prev_row[-1]
        
    def process(self, texts: List[str]) -> None:
        """
        处理文本语料库，提取潜在扰动并构建哈希表
        """
        for text in texts:
            # 分词
            words = list(jieba.cut(text))
            
            for word in words:
                if not word.strip():
                    continue
                    
                # 计算不同级别的拼音编码
                for level in range(3):
                    code = self.get_pinyin_code(word, level)
                    if level == 0:
                        self.initial_hash[code].add(word)
                    else:
                        self.pinyin_hash[f"{level}_{code}"].add(word)
                        
    def build_homophone_dict(self) -> None:
        """构建同音字词典"""
        # 常用汉字范围
        common_chars = []
        for i in range(0x4e00, 0x9fa5):  # CJK统一汉字
            common_chars.append(chr(i))
            
        # 按拼音分组
        pinyin_groups = defaultdict(list)
        for char in common_chars:
            try:
                py = lazy_pinyin(char)
                if py:
                    pinyin_groups[py[0]].append(char)
            except:
                continue
                
        self.homophone_chars = dict(pinyin_groups)
        
    def get_similar_chars(self, char: str, mode: str = 'all') -> Set[str]:
        """
        获取相似字符
        """
        results = set()
        
        if mode in ['pinyin', 'all']:
            # 同音字扰动
            try:
                py = lazy_pinyin(char)
                if py and py[0] in self.homophone_chars:
                    for similar in self.homophone_chars[py[0]]:
                        if similar != char:
                            results.add(similar)
            except:
                pass
                
        if mode in ['shape', 'all']:
            # 形近字扰动
            if char in self.similar_shape_chars:
                results.update(self.similar_shape_chars[char])
                
        if mode in ['sub', 'all']:
            # 字符替代扰动
            if char in self.char_substitution:
                results.update(self.char_substitution[char])
                
        return results
        
    def get_similars(self, word: str, level: int = 1, distance: int = 2, 
                     mode: str = 'all', strict: bool = True) -> Set[str]:
        """
        获取词汇的相似扰动词汇集合
        """
        results = set()
        
        # 基于拼音哈希表查找
        code = self.get_pinyin_code(word, level)
        hash_key = f"{level}_{code}" if level > 0 else code
        hash_table = self.pinyin_hash if level > 0 else self.initial_hash
        
        if hash_key in hash_table:
            for candidate in hash_table[hash_key]:
                if candidate != word:
                    dist = self.levenshtein_distance(word, candidate)
                    if dist <= distance:
                        results.add(candidate)
                        
        # 基于字符级扰动生成
        # 逐字符扰动
        for i, char in enumerate(word):
            similar_chars = self.get_similar_chars(char, mode)
            for sim_char in similar_chars:
                new_word = word[:i] + sim_char + word[i+1:]
                if new_word != word:
                    if strict:
                        # 严格模式：检查拼音相似度
                        if self.get_pinyin_code(new_word, 1) == self.get_pinyin_code(word, 1):
                            results.add(new_word)
                    else:
                        results.add(new_word)
                        
        return results
        
    def generate_perturbation(self, text: str, 
                              important_words: Optional[List[str]] = None,
                              perturbation_ratio: float = 0.3,
                              mode: str = 'all') -> str:
        """
        对文本生成对抗扰动
        """
        # 分词
        words = list(jieba.cut(text))
        
        # 确定需要扰动的词汇数量
        num_perturb = max(1, int(len(words) * perturbation_ratio))
        
        # 优先扰动重要词汇
        perturb_indices = []
        if important_words:
            for i, word in enumerate(words):
                if word in important_words:
                    perturb_indices.append(i)
                    
        # 如果重要词汇不够，随机补充
        import random
        remaining = num_perturb - len(perturb_indices)
        if remaining > 0:
            other_indices = [i for i in range(len(words)) 
                           if i not in perturb_indices and len(words[i].strip()) > 0]
            if other_indices:
                perturb_indices.extend(random.sample(other_indices, 
                                                     min(remaining, len(other_indices))))
                                                     
        # 执行扰动
        perturbed_words = words.copy()
        for idx in perturb_indices:
            word = words[idx]
            similars = self.get_similars(word, level=1, distance=2, mode=mode, strict=False)
            if similars:
                perturbed_words[idx] = random.choice(list(similars))
                
        return ''.join(perturbed_words)
        
    def save(self, path: str) -> None:
        """保存哈希表到本地"""
        os.makedirs(path, exist_ok=True)
        
        # 转换 set 为 list 以便 JSON 序列化
        pinyin_data = {k: list(v) for k, v in self.pinyin_hash.items()}
        initial_data = {k: list(v) for k, v in self.initial_hash.items()}
        homophone_data = dict(self.homophone_chars)
        
        with open(os.path.join(path, 'pinyin_hash.json'), 'w', encoding='utf-8') as f:
            json.dump(pinyin_data, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(path, 'initial_hash.json'), 'w', encoding='utf-8') as f:
            json.dump(initial_data, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(path, 'homophone_chars.json'), 'w', encoding='utf-8') as f:
            json.dump(homophone_data, f, ensure_ascii=False, indent=2)
            
        with open(os.path.join(path, 'similar_shape_chars.json'), 'w', encoding='utf-8') as f:
            json.dump(self.similar_shape_chars, f, ensure_ascii=False, indent=2)
            
        print(f"已保存到 {path}")
        
    def load(self, path: str) -> None:
        """从本地加载哈希表"""
        with open(os.path.join(path, 'pinyin_hash.json'), 'r', encoding='utf-8') as f:
            pinyin_data = json.load(f)
            self.pinyin_hash = defaultdict(set, {k: set(v) for k, v in pinyin_data.items()})
            
        with open(os.path.join(path, 'initial_hash.json'), 'r', encoding='utf-8') as f:
            initial_data = json.load(f)
            self.initial_hash = defaultdict(set, {k: set(v) for k, v in initial_data.items()})
            
        if os.path.exists(os.path.join(path, 'homophone_chars.json')):
            with open(os.path.join(path, 'homophone_chars.json'), 'r', encoding='utf-8') as f:
                self.homophone_chars = json.load(f)
                
        if os.path.exists(os.path.join(path, 'similar_shape_chars.json')):
            with open(os.path.join(path, 'similar_shape_chars.json'), 'r', encoding='utf-8') as f:
                self.similar_shape_chars = json.load(f)
                
        print(f"已从 {path} 加载")


# 针对欺诈对话的特殊扰动规则
class FraudDialoguePerturbation(ChineseANTHRO):
    """
    针对欺诈对话场景的扰动生成器，继承自 ChineseANTHRO，增加了欺诈领域特有的扰动规则
    """
    
    def __init__(self):
        super().__init__()
        self._init_fraud_specific_rules()
        self.build_homophone_dict()

    def _init_fraud_specific_rules(self):
        """初始化欺诈领域特有的扰动规则"""
        
        # 欺诈关键词及其扰动变体
        self.fraud_keyword_variants = {
            # 金融类
            '转账': ['转帐', '划转', '汇款', '过账', '转钱'],
            '账户': ['帐户', '账号', '帐号', '用户'],
            '密码': ['秘码', '口令', '暗码', 'mima'],
            '验证码': ['验证吗', '校验码', '确认码', 'yzm'],
            '银行': ['银航', '银杭', '银行卡'],
            '信用卡': ['信用咔', '信用ka', 'xyk'],
            '贷款': ['贷欵', '借款', '借钱'],
            '利息': ['利率', '利钱'],
            '额度': ['额渡', '限额'],
            '提现': ['提款', '取现', '取款'],
            '手续费': ['手续废', '费用', '工本费'],
            
            # 安全类
            '安全': ['安全', '安权', 'anquan'],
            '风险': ['风险', '危险'],
            '冻结': ['动结', '冻洁', '封存'],
            '异常': ['异长', '问题'],
            '账号被盗': ['账号被dao', '被盗号'],
            
            # 身份伪装类
            '客服': ['客副', '客fu', 'kefu'],
            '工作人员': ['工作人圆', '工zuo人员'],
            '警察': ['jingcha', '警查', '警茶'],
            '公安': ['公按', 'gongan'],
            '检察院': ['检查院', '检察圆'],
            '法院': ['法圆', '法远'],
            
            # 诱导类
            '中奖': ['中将', 'zhongjiang', '得奖'],
            '返现': ['反现', '返xian', '返钱'],
            '红包': ['洪包', '红bao', '红宝'],
            '优惠': ['优会', '优慧', '优惠券'],
            '免费': ['勉费', '免废', '不要钱'],
            '赠送': ['增送', '赠song'],
            '积分': ['集分', '积份'],
            '升级': ['声级', '升ji'],
            
            # 操作类
            '链接': ['连接', '连结', '链jie'],
            '点击': ['点及', '点ji'],
            '下载': ['下栽', '下zai'],
            '登录': ['登陆', '登入', 'denglu'],
            '注册': ['注侧', '注ce'],
            '确认': ['确人', '确仁'],
            '操作': ['cao作', '草作'],
            
            # 快递物流类
            '快递': ['快di', '快地', 'kuaidi'],
            '包裹': ['包guo', '包锅'],
            '签收': ['签受', '签shou'],
            '物流': ['物刘', '物liu'],
        }
        
        # 合并到形近字映射
        for word, variants in self.fraud_keyword_variants.items():
            if word not in self.similar_shape_chars:
                self.similar_shape_chars[word] = []
            self.similar_shape_chars[word].extend(variants)
            
    def get_fraud_keywords(self, text: str) -> List[Tuple[str, int]]:
        """
        识别文本中的欺诈关键词及其重要性得分
        
        Args:
            text: 输入文本
            
        Returns:
            (关键词, 重要性得分) 列表，按得分降序排列
        """
        # 分词
        words = list(jieba.cut(text))
        
        # 关键词权重
        keyword_weights = {
            # 高危关键词 (权重 5)
            '转账': 5, '验证码': 5, '密码': 5, '冻结': 5, '警察': 5,
            '公安': 5, '检察': 5, '法院': 5, '通缉': 5, '洗钱': 5,
            
            # 中危关键词 (权重 3)
            '银行': 3, '账户': 3, '安全': 3, '客服': 3, '链接': 3,
            '点击': 3, '下载': 3, '登录': 3, '中奖': 3, '返现': 3,
            '贷款': 3, '利息': 3, '额度': 3, '手续费': 3,
            
            # 低危关键词 (权重 1)
            '红包': 1, '优惠': 1, '免费': 1, '赠送': 1, '积分': 1,
            '快递': 1, '包裹': 1, '签收': 1, '物流': 1,
        }
        
        # 识别关键词并计算得分
        results = []
        for word in words:
            for keyword, weight in keyword_weights.items():
                if keyword in word:
                    results.append((word, weight))
                    break
                    
        # 按得分降序排列
        results.sort(key=lambda x: x[1], reverse=True)
        return results
        
    def attack(self, text: str, target_model=None, 
               max_perturbations: int = 5) -> Tuple[str, bool]:
        """
        对欺诈对话执行对抗攻击
        """
        # 识别关键词
        keywords = self.get_fraud_keywords(text)
        
        # 按重要性顺序扰动
        perturbed_text = text
        attack_success = False
        
        for word, score in keywords[:max_perturbations]:
            # 获取扰动词汇
            variants = self.get_similars(word, level=1, distance=2, 
                                        mode='all', strict=False)
            
            if not variants and word in self.fraud_keyword_variants:
                variants = set(self.fraud_keyword_variants[word])
                
            if variants:
                import random
                new_word = random.choice(list(variants))
                perturbed_text = perturbed_text.replace(word, new_word, 1)
                
                # 3. 验证攻击效果
                if target_model is not None:
                    try:
                        # 假设模型有 predict 方法
                        original_pred = target_model.predict([text])[0]
                        perturbed_pred = target_model.predict([perturbed_text])[0]
                        
                        if original_pred != perturbed_pred:
                            attack_success = True
                            break
                    except:
                        pass
                        
        return perturbed_text, attack_success
        

if __name__ == '__main__':
    # 测试示例
    anthro = FraudDialoguePerturbation()
    
    # 测试文本
    test_texts = [
        "您好，这里是银行客服中心，您的账户存在安全风险，请立即转账到安全账户。",
        "恭喜您中奖了，请点击链接领取红包，需要输入验证码完成验证。",
        "您的快递包裹有问题，请联系客服处理，可能需要支付手续费。",
    ]
    
    print("=" * 60)
    print("中文 ANTHRO 对抗扰动测试")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\n原始文本: {text}")
        
        # 识别关键词
        keywords = anthro.get_fraud_keywords(text)
        print(f"识别到的关键词: {keywords}")
        
        # 生成扰动
        perturbed, _ = anthro.attack(text, max_perturbations=3)
        print(f"扰动文本: {perturbed}")
        
        print("-" * 40)
