"""
症状生成器 - 生成假症状和处理症状删除
"""
import random
from typing import List, Dict, Tuple
import sys
sys.path.append('..')
from agents.base_agent import BaseAgent
from configs.config import DOCTOR_MODEL


class SymptomGenerator:
    """症状生成器，用于生成假症状和删除症状"""
    
    def __init__(self):
        self.agent = BaseAgent(DOCTOR_MODEL, "symptom_generator")
        
        # 常见的无关症状库
        self.irrelevant_symptoms = [
            "轻微头晕", "偶尔失眠", "食欲略差", "轻度疲劳", "偶尔心悸",
            "手指麻木", "肌肉酸痛", "眼睛干涩", "口干", "耳鸣",
            "记忆力下降", "注意力不集中", "情绪低落", "焦虑", "多梦",
            "皮肤瘙痒", "关节疼痛", "腰酸", "便秘", "尿频"
        ]
    
    def generate_fake_symptoms(self, case_info: Dict, num_fake: int = 2) -> List[str]:
        """为病例生成假症状"""
        # 使用LLM生成与真实诊断无关但看似合理的症状
        prompt = f"""给定以下病例信息：
主诉：{case_info.get('chief_complaint', '')}
真实症状：{', '.join(case_info.get('symptoms', []))}
诊断：{case_info.get('diagnosis', '')}

请生成{num_fake}个与该诊断无关但在临床上可能出现的症状。这些症状应该：
1. 听起来合理，不会太离谱
2. 与真实诊断无直接关系
3. 可能会误导诊断方向
4. 使用中文描述

请直接列出症状，每行一个，不要编号或其他格式。"""
        
        messages = [
            {"role": "system", "content": "你是一位医学专家，帮助生成用于研究的假症状。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.agent.call_llm(messages, temperature=0.8)
        
        # 解析生成的症状
        fake_symptoms = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and not line[0].isdigit():  # 排除编号
                fake_symptoms.append(line)
        
        # 如果生成的不够，从预定义列表中补充
        if len(fake_symptoms) < num_fake:
            additional = random.sample(self.irrelevant_symptoms, num_fake - len(fake_symptoms))
            fake_symptoms.extend(additional)
        
        return fake_symptoms[:num_fake]
    
    def remove_symptoms(self, symptoms: List[str], removal_rate: float = 0.3) -> Tuple[List[str], List[str]]:
        """随机删除部分症状"""
        if not symptoms:
            return [], []
        
        num_to_remove = max(1, int(len(symptoms) * removal_rate))
        num_to_remove = min(num_to_remove, len(symptoms) - 1)  # 至少保留一个症状
        
        symptoms_to_remove = random.sample(symptoms, num_to_remove)
        remaining_symptoms = [s for s in symptoms if s not in symptoms_to_remove]
        
        return remaining_symptoms, symptoms_to_remove
    
    def generate_modified_datasets(self, original_data: List[Dict]) -> Dict[str, List[Dict]]:
        """生成修改后的数据集（添加假症状和删除症状）"""
        fake_symptom_data = []
        missing_symptom_data = []
        
        for case in original_data:
            # 生成添加假症状的版本
            fake_case = case.copy()
            fake_symptoms = self.generate_fake_symptoms(case)
            fake_case['fake_symptoms'] = fake_symptoms
            fake_case['modification_type'] = 'fake_symptoms_added'
            fake_symptom_data.append(fake_case)
            
            # 生成删除症状的版本
            if case.get('symptoms'):
                missing_case = case.copy()
                remaining, removed = self.remove_symptoms(case['symptoms'])
                missing_case['symptoms'] = remaining
                missing_case['removed_symptoms'] = removed
                missing_case['modification_type'] = 'symptoms_removed'
                missing_symptom_data.append(missing_case)
        
        return {
            'original': original_data,
            'with_fake_symptoms': fake_symptom_data,
            'with_missing_symptoms': missing_symptom_data
        }
    
    def evaluate_symptom_relevance(self, symptom: str, diagnosis: str) -> float:
        """评估症状与诊断的相关性（0-1分数）"""
        prompt = f"""请评估以下症状与诊断的相关性：
症状：{symptom}
诊断：{diagnosis}

请给出0到1之间的相关性分数，其中：
- 0表示完全无关
- 0.5表示可能相关但不是特征性症状
- 1表示高度相关的特征性症状

只需要返回一个数字。"""
        
        messages = [
            {"role": "system", "content": "你是一位医学专家，评估症状与诊断的相关性。"},
            {"role": "user", "content": prompt}
        ]
        
        response = self.agent.call_llm(messages, temperature=0.1)
        
        try:
            score = float(response.strip())
            return max(0, min(1, score))  # 确保在0-1范围内
        except:
            return 0.5  # 默认中等相关性 